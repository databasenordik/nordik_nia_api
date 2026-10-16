"""Capture real, scoped research-bank responses without changing runtime behavior.

Results are evidence, not an automated claim of semantic correctness. The existing
prose grader is retained as an advisory signal; an independent source review must
decide accuracy, unsupported expectations, and conversational quality.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import dataclasses
import hashlib
import json
import logging
import math
import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_calls: contextvars.ContextVar[list | None] = contextvars.ContextVar('calls', default=None)


def serializable(value: Any) -> Any:
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if dataclasses.is_dataclass(value):
        return {f.name: serializable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def percentiles(values: list[float]) -> dict:
    ordered = sorted(values)
    return {'n': len(values), 'median_ms': statistics.median(values),
            'p95_ms': ordered[math.ceil(.95 * len(values)) - 1]} if values else {'n': 0}


async def run(args):
    # Import after selecting the preserved or candidate application root.
    import asyncpg
    import app
    from app.config import get_settings
    from app.data_gateway.repository import AssistantDataGateway
    from app.db.privilege_selftest import assert_runtime_isolation
    from app.execution.turn import AssistantTurnService
    from app.llm.factory import get_reasoning_provider
    from app.memory.store import InMemoryMemoryStore
    from app.security.access_scope import AccessScope
    from benchmarks.grading import grade

    os.environ['PLANNER_MODE'] = 'ai'
    settings = get_settings()
    dsn = settings.assistant_database_url.rsplit('/', 1)[0] + '/' + args.database
    os.environ['ASSISTANT_DATABASE_URL'] = dsn
    get_settings.cache_clear()
    settings = get_settings()
    await assert_runtime_isolation(dsn)
    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=max(4, args.concurrency * 2))
    scope = AccessScope(principal_id='principal-researcher', allowed_file_ids=(49, 91, 93, 94),
                        can_use_private_files=True)
    gateway = AssistantDataGateway(pool)
    counts = {f: await gateway.count_records(scope, (f,), []) for f in scope.allowed_file_ids}
    provider = get_reasoning_provider()
    if provider is None or settings.reasoning_provider == 'fake':
        raise RuntimeError('This evaluation requires the configured real provider')
    if hasattr(provider, '_post'):
        original_post = provider._post

        async def recorded_post(body):
            started = time.perf_counter()
            call = {'request': body}
            try:
                result = await original_post(body)
                call['response'] = result
                return result
            except Exception as exc:
                call['error_type'] = type(exc).__name__
                call['error'] = str(exc)[:500]
                raise
            finally:
                call['wall_ms'] = (time.perf_counter() - started) * 1000
                if _calls.get() is not None:
                    _calls.get().append(call)

        provider._post = recorded_post

    bank_bytes = args.bank.read_bytes()
    questions = json.loads(bank_bytes)['questions']
    if args.only:
        selected = set(args.only.split(','))
        questions = [q for q in questions if q['id'] in selected]
    args.out.mkdir(parents=True, exist_ok=True)
    result_file = args.out / 'responses.jsonl'
    if result_file.exists():
        raise RuntimeError('Preserve existing evidence; choose a new output directory')
    app_root = Path(app.__file__).parent
    hashes = {str(p.relative_to(app_root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in app_root.rglob('*.py')}
    metadata = {
        'started_at': datetime.now(UTC).isoformat(), 'python': platform.python_version(),
        'database': args.database, 'counts': counts, 'runtime_isolation': 'passed',
        'provider': settings.reasoning_provider, 'model': settings.reasoning_model,
        'temperature': settings.model_temperature, 'seed': settings.model_seed,
        'concurrency': args.concurrency, 'runs': args.runs, 'app_hashes': hashes,
        'bank_sha256': hashlib.sha256(bank_bytes).hexdigest(),
        'latency_policy': 'wall clock, all outcomes included; nearest-rank p95; no hidden retries',
        'grading_policy': 'automated grade advisory only; independent review required',
    }
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    semaphore = asyncio.Semaphore(args.concurrency)
    results = []

    async def answer(q, repeat, service=None, conversation_id=None):
        async with semaphore:
            service = service or AssistantTurnService(gateway, store=InMemoryMemoryStore(), reasoner=provider)
            calls = []
            token = _calls.set(calls)
            started = time.perf_counter()
            record = {'id': q['id'], 'repeat': repeat, 'file_id': q['file_id'],
                      'question': q['question'], 'expected_answer': q.get('correct_answer', '')}
            try:
                result = await asyncio.wait_for(service.answer(
                    scope, q['question'], selected_file_id=q['file_id'],
                    conversation_id=conversation_id), timeout=args.timeout)
                record['result'] = serializable(result)
                record['advisory_grade'] = serializable(grade(q.get('correct_answer', ''), result.answer))
                conversation_id = result.conversation_id
            except Exception as exc:
                record['exception'] = type(exc).__name__
                record['error'] = str(exc)[:500]
            finally:
                record['wall_ms'] = (time.perf_counter() - started) * 1000
                record['calls'] = calls
                _calls.reset(token)
            results.append(record)
            with result_file.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            result = record.get('result', {})
            print(json.dumps({'id': q['id'], 'repeat': repeat, 'ms': round(record['wall_ms']),
                              'status': result.get('status', record.get('exception')),
                              'answer': result.get('answer', '')[:120]}, ensure_ascii=False), flush=True)
            return conversation_id

    try:
        # Fixed interleaved order and isolated conversations on each repetition.
        for repeat in range(1, args.runs + 1):
            await asyncio.gather(*(answer(q, repeat) for q in questions))
        if args.history:
            threads = json.loads(args.history.read_text(encoding='utf-8'))
            for index, thread in enumerate(threads):
                service = AssistantTurnService(gateway, store=InMemoryMemoryStore(), reasoner=provider)
                conversation_id = None
                for turn, prompt in enumerate(thread.get('prompts', [])):
                    conversation_id = await answer(
                        {'id': f'H{index + 1:02d}.{turn + 1}', 'file_id': thread['selected_file_id'],
                         'question': prompt}, 1, service, conversation_id)
    finally:
        await pool.close()
        if getattr(provider, '_client', None) is not None:
            await provider._client.aclose()
    summary = {
        'all_outcomes': percentiles([r['wall_ms'] for r in results]),
        'answered': percentiles([r['wall_ms'] for r in results if r.get('result', {}).get('status') == 'answered']),
        'statuses': {status: sum(r.get('result', {}).get('status', r.get('exception')) == status for r in results)
                     for status in {r.get('result', {}).get('status', r.get('exception')) for r in results}},
        'provider_errors': sum('error_type' in c for r in results for c in r['calls']),
        'completed_at': datetime.now(UTC).isoformat(),
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, default=Path('database_review_outputs/testing_questions.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--database', required=True)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--only')
    parser.add_argument('--history', type=Path)
    parser.add_argument('--app-root', type=Path)
    args = parser.parse_args()
    if args.app_root:
        sys.path.insert(0, str(args.app_root.resolve()))
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
