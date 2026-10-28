"""Ask Nia questions from a terminal, against a real database and the real provider.

A console for trying prompts out: pick a source list, type questions, read the answer
and -- the part a prompt change is actually judged on -- what the planner made of it.

There is no sign-in. The access scope is built here, the way the benchmark harnesses
build it, because this tool runs as whoever can already read the database it is pointed
at. That also means it is not a way around the runtime's own boundary: the pool refuses
to open unless the role is isolated from the raw tables and the Potential release gate is
installed, exactly as the application requires.

    python backend/benchmarks/ask.py                      # the configured database
    python backend/benchmarks/ask.py --database nia_v6    # another database on that host
    python backend/benchmarks/ask.py --file-id 49         # skip the list prompt

Latency is reported split into local and provider time, because the upstream provider
dominates the wall clock and moves by the hour: only the local half says anything about
a change made here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
# This file is synced into the deployed API repository, where app/ is the repository root
# rather than a backend/ directory inside it, so the repository is whichever of the two
# holds the .env -- and neither, when there is none.
REPO_ROOT = next(
    (path for path in (BACKEND_ROOT.parent, BACKEND_ROOT) if (path / ".env").is_file()),
    BACKEND_ROOT.parent,
)
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def load_repository_env() -> None:
    """Read the repository's .env regardless of where this was run from.

    Settings name their env files by relative path, which resolves against the working
    directory: from the repository root or from backend/ the file is found, and from
    backend/benchmarks/ -- next to this script, where someone would naturally run it --
    nothing is found and every setting falls back to its default. The symptom was an
    empty database URL and a DSN of "/nia_eval_20261016". Anything already exported
    wins, exactly as it would have.
    """
    try:
        from dotenv import dotenv_values  # noqa: PLC0415 - optional at import time
    except ModuleNotFoundError:  # pragma: no cover - python-dotenv is a dependency
        return
    for name, value in dotenv_values(REPO_ROOT / ".env").items():
        if value is not None:
            os.environ.setdefault(name, value)


load_repository_env()

import asyncpg  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.data_gateway.repository import AssistantDataGateway  # noqa: E402
from app.db.pool import close_pool, get_pool, init_pool  # noqa: E402
from app.execution.turn import AssistantTurnService, select_dataset_for_question  # noqa: E402
from app.llm.factory import get_reasoning_provider  # noqa: E402
from app.memory.store import InMemoryMemoryStore  # noqa: E402
from app.security.access_scope import AccessScope  # noqa: E402

# The database holding the most recent version of the data: the current source rows plus
# the normalization and access migrations applied after them. The configured
# ASSISTANT_DATABASE_URL still points at an older snapshot, which the runtime's own
# boundary check refuses, so asking questions would otherwise start with a flag. Any name
# on that host can be passed to --database, and if this one is gone the refusal names the
# databases that would serve.
NEWEST_DATABASE = "nia_eval_20261016"

HELP = """Commands
  :lists         the authorized lists and their record counts
  :use <file_id> ask the next questions about that list
  :auto          let Nia choose the list per question
  :new           start a fresh conversation (drops follow-up context)
  :plan          the compiled query plan for the last answer
  :raw           what the planner and reviewer actually exchanged
  :trace         per-layer timings and statuses for the last answer
  :save <path>   write this session's turns to a JSON file
  :help          this list
  :quit          leave
"""


class Recorder:
    """Keeps each provider request and response, with its own wall time.

    The console reports provider time separately and `:raw` prints these payloads,
    which is the only way to see whether a prompt change reached the model and what
    came back. Wrapping the transport is how the research-bank harness does it, so
    the numbers here and there mean the same thing.
    """

    def __init__(self, provider: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self._provider = provider
        self._original = getattr(provider, "_post", None)
        if self._original is not None:
            provider._post = self._post

    async def _post(self, body: Any) -> Any:
        started = time.perf_counter()
        call: dict[str, Any] = {"request": body}
        try:
            call["response"] = result = await self._original(body)
            return result
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            call["wall_ms"] = (time.perf_counter() - started) * 1000
            self.calls.append(call)

    def take(self) -> list[dict[str, Any]]:
        calls, self.calls = self.calls, []
        return calls


def dsn_for(args: argparse.Namespace) -> str:
    if args.database_url:
        return args.database_url
    configured = get_settings().assistant_database_url
    if not args.database:
        return configured
    return configured.rsplit("/", 1)[0] + "/" + args.database


def with_host(dsn: str, host: str) -> str:
    head, sep, tail = dsn.rpartition("@")
    if not sep:
        return dsn
    _, _, rest = tail.partition("/")
    port = tail.partition("/")[0].rpartition(":")[2]
    return f"{head}@{host}:{port}/{rest}" if port.isdigit() else f"{head}@{host}/{rest}"


def wsl_hosts() -> list[str]:
    """Addresses of the WSL virtual machine, for a database published only inside it.

    Docker Engine running in WSL publishes its ports on the VM's interfaces, and
    Windows does not always forward localhost to them -- so the configured
    localhost DSN is refused from a Windows shell while the database is running
    and reachable one address away. Finding it beats telling someone to run
    `hostname -I` themselves.
    """
    if not sys.platform.startswith("win"):
        return []
    try:
        import subprocess  # noqa: PLC0415 - only needed on this path

        output = subprocess.run(
            ["wsl", "hostname", "-I"],
            capture_output=True,
            # wsl.exe drains whatever it inherits, and this console's stdin is the
            # questions: without this the first prompt read EOF and the session ended
            # before a word of it was asked.
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=20,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [item for item in output.split() if item.count(".") == 3]


async def probe_pool(connections: int) -> None:
    """Hold several pool connections at once, so a fresh one has to be made."""

    async def one() -> None:
        async with get_pool().acquire() as connection:
            await connection.fetchval("SELECT 1")
            await asyncio.sleep(0.05)

    await asyncio.gather(*(one() for _ in range(max(connections, 2))))


async def working_databases(dsn: str) -> list[str]:
    """The databases on this host that would serve the runtime, for a failure message.

    Pointing the console at a database that has not had the access migrations applied
    is the easy mistake -- the configured one is often an older snapshot -- and the
    answer is a name, so the message may as well carry the names that work.
    """
    import asyncpg  # noqa: PLC0415 - import cost only on the failure path

    from app.db.privilege_selftest import assert_runtime_isolation  # noqa: PLC0415

    base, _, current = dsn.rpartition("/")
    try:
        connection = await asyncpg.connect(dsn=dsn, timeout=10)
    except Exception:  # noqa: BLE001 - the caller already reported why
        return []
    try:
        names = [
            str(row["datname"])
            for row in await connection.fetch(
                "SELECT datname FROM pg_database WHERE NOT datistemplate ORDER BY datname"
            )
        ]
    except Exception:  # noqa: BLE001
        return []
    finally:
        await connection.close()
    usable = []
    for name in names:
        if name == current:
            continue
        try:
            await assert_runtime_isolation(f"{base}/{name}")
        except Exception:  # noqa: BLE001 - this one simply does not qualify
            continue
        usable.append(name)
    return usable


def redacted(dsn: str) -> str:
    head, _, tail = dsn.rpartition("@")
    if not head:
        return dsn
    user = head.rpartition("//")[2].split(":", 1)[0]
    return f"{head.rpartition('//')[0]}//{user}:***@{tail}"


def serializable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [serializable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def show_lists(datasets: list[tuple[int, str, int]], selected: int | None) -> None:
    for position, (file_id, label, count) in enumerate(datasets, start=1):
        mark = "*" if file_id == selected else " "
        print(f" {mark} [{position}] {file_id:<4} {label:<28} {count:>6,} records")
    if selected is None:
        print("   (no list selected; Nia chooses one per question)")


def resolve_choice(reply: str, datasets: list[tuple[int, str, int]]) -> int | None:
    reply = reply.strip()
    if not reply:
        return None
    by_id = {file_id for file_id, _, _ in datasets}
    if reply.isdigit():
        number = int(reply)
        if number in by_id:
            return number
        if 1 <= number <= len(datasets):
            return datasets[number - 1][0]
        raise ValueError(f"no list numbered {number}")
    wanted = reply.casefold()
    hits = [item for item in datasets if wanted in item[1].casefold()]
    if len(hits) == 1:
        return hits[0][0]
    raise ValueError(f"{reply!r} matches {len(hits)} lists; use a file id")


def attempts_for(args: argparse.Namespace, base: str) -> list[str]:
    """Addresses to try for the chosen database, best first.

    Calling this again re-reads the WSL address, which is the point: the distro shuts
    down when idle and comes back with the database -- and possibly a new address --
    the moment anything talks to it.
    """
    if not is_local(args, base):
        return [base]
    # The direct address first: where the database is published inside WSL, the
    # forwarded localhost port accepts a first connection and refuses a later one,
    # which surfaces as a turn failing halfway through a session. The address used is
    # printed, so this is a better default rather than a hidden one.
    return [with_host(base, host) for host in wsl_hosts()] + [base]


async def connect(args: argparse.Namespace, base: str) -> tuple[str | None, str]:
    """Open the pool on the first address that serves, or say why none did."""
    attempts = attempts_for(args, base)
    last = ""
    for candidate in attempts:
        os.environ["ASSISTANT_DATABASE_URL"] = candidate
        get_settings.cache_clear()
        try:
            await init_pool()
            # Opening the pool is not the same as being able to use it: a port forwarded
            # into this host can accept the first connections and refuse the next, and
            # the symptom is then a turn failing halfway through a session. Hold as many
            # connections at once as the pool allows, so one has to be made here and now,
            # before settling on this address.
            await probe_pool(get_settings().db_pool_max)
            return candidate, ""
        except PermissionError as exc:
            # The pool's own boundary check, not this tool's: this database will not
            # serve whichever address it is reached by, so stop rather than retry.
            await close_pool()
            return None, f"boundary:{exc}"
        except OSError as exc:
            await close_pool()
            last = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 - the address is the useful part here
            await close_pool()
            last = f"{type(exc).__name__}: {exc}"
    return None, last


def is_local(args: argparse.Namespace, base: str) -> bool:
    """Whether the chosen database is the one on this machine, by its original address.

    Taken from the address the console started with, never from the environment: opening
    the pool rewrites ASSISTANT_DATABASE_URL to whichever address answered, and reading
    that back makes a database reached at the WSL machine's own address look remote.
    """
    return not args.database_url and ("@localhost" in base or "@127.0.0.1" in base)


def wake_local_stack(args: argparse.Namespace, base: str) -> bool:
    """Ask Docker in WSL for its container list, which brings the engine back up.

    A WSL distro shuts down when idle and stops the Docker engine with it, so a
    database that is supposed to be running is simply not there. Any command against
    the engine starts it, and containers the user has already marked to restart come
    back on their own -- so this reads the container list and changes nothing itself.
    Only for a database on this machine: a remote address is nothing to do with WSL.
    """
    if not is_local(args, base) or not sys.platform.startswith("win"):
        return False
    try:
        import subprocess  # noqa: PLC0415 - only needed on this path

        subprocess.run(
            ["wsl", "docker", "ps", "--quiet"],
            capture_output=True,
            stdin=subprocess.DEVNULL,  # wsl.exe would otherwise eat the next question
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


async def wait_for_database(
    args: argparse.Namespace, base: str, *, seconds: int = 90
) -> tuple[str | None, str]:
    """Keep trying until the database answers or the wait runs out.

    The distro, the Docker engine and PostgreSQL each take their own time coming back
    -- around forty seconds here -- and a single attempt lands in the middle of that.
    """
    deadline = time.monotonic() + seconds
    woken = False
    while True:
        dsn, failure = await connect(args, base)
        if dsn is not None or failure.startswith("boundary:"):
            return dsn, failure
        if not woken:
            woken = True
            if wake_local_stack(args, base):
                print(f"  waking the local stack, waiting up to {seconds}s for it")
            else:
                print(f"  waiting up to {seconds}s for the database")
        if time.monotonic() >= deadline:
            return None, failure
        await asyncio.sleep(4)


async def run(args: argparse.Namespace) -> int:
    if args.planner_mode:
        os.environ["PLANNER_MODE"] = args.planner_mode

    base = dsn_for(args)
    if not base.startswith(("postgresql://", "postgres://")):
        print("There is no database address to connect to.")
        print(f"  ASSISTANT_DATABASE_URL is {get_settings().assistant_database_url or 'empty'}")
        print(f"  Expected it in {REPO_ROOT / '.env'}, or pass --database-url.")
        return 2
    dsn, failure = await connect(args, base)
    if dsn is None and not failure.startswith("boundary:"):
        # Not there on the first try: it may simply not be running yet.
        dsn, failure = await wait_for_database(args, base)
    if dsn is None:
        if failure.startswith("boundary:"):
            print(f"This database will not serve the runtime role: {failure[9:]}")
            print(f"  database: {redacted(base)}")
            usable = await working_databases(base)
            if usable:
                print(f"  On that host these would: {', '.join(usable)}")
                print(f"  Try --database {usable[0]}")
            else:
                print("  Apply the migrations through 073 there, as assistant_runtime.")
            return 2
        print(f"Could not connect: {failure}")
        print(f"  database: {redacted(base)}")
        if len(attempts_for(args, base)) > 1:
            print("  Tried the WSL virtual machine's address and the configured host.")
        return 2
    settings = get_settings()

    provider = get_reasoning_provider()
    if provider is None or settings.reasoning_provider == "fake":
        print("No real reasoning provider is configured, so there are no prompts to test.")
        print("  Set REASONING_PROVIDER, REASONING_BASE_URL, REASONING_MODEL and the key.")
        await close_pool()
        return 2

    gateway = AssistantDataGateway(get_pool())
    scope = AccessScope(
        principal_id=args.principal,
        allowed_file_ids=tuple(int(item) for item in args.file_ids.split(",") if item.strip()),
        can_use_private_files=True,
    )
    catalog = await gateway.load_catalog(scope)
    datasets = [
        (
            dataset.file_id,
            dataset.user_facing_label,
            await gateway.count_records(scope, (dataset.file_id,), []),
        )
        for dataset in catalog.datasets
    ]
    if not datasets:
        print("That database exposes no authorized lists for this scope.")
        await close_pool()
        return 2

    print(f"Nia console · {redacted(dsn)} · {settings.reasoning_model}")
    print()
    show_lists(datasets, args.file_id)
    print()

    selected = args.file_id
    if selected is None:
        while True:
            reply = await asyncio.to_thread(
                input, "Which list? (number, file id, name, or blank for automatic) "
            )
            try:
                selected = resolve_choice(reply, datasets)
            except ValueError as exc:
                print(f"  {exc}")
                continue
            break
    elif selected not in {file_id for file_id, _, _ in datasets}:
        print(f"--file-id {selected} is not one of the authorized lists.")
        await close_pool()
        return 2

    label = next((item[1] for item in datasets if item[0] == selected), "automatic")
    print(f"Asking about: {label}" if selected else "Asking about: whichever list fits")
    print()
    print(HELP)

    recorder = Recorder(provider)
    store = InMemoryMemoryStore()
    service = AssistantTurnService(gateway, store=store, reasoner=provider)
    previous_dsn = dsn
    lost = False

    async def revive() -> tuple[AssistantDataGateway, AssistantTurnService] | None:
        """Reopen the pool and rebuild what holds it, or explain why not."""
        nonlocal previous_dsn
        await close_pool()
        address, failure = await wait_for_database(args, base)
        if address is None:
            print(f"  still unreachable: {failure.removeprefix('boundary:')}")
            print("  ask again once the database is back")
            return None
        if address != previous_dsn:
            print(f"  now at {redacted(address)}")
            previous_dsn = address
        fresh = AssistantDataGateway(get_pool())
        return fresh, AssistantTurnService(fresh, store=store, reasoner=provider)
    conversation_id: str | None = None
    last: dict[str, Any] | None = None
    turns: list[dict[str, Any]] = []

    while True:
        try:
            line = await asyncio.to_thread(input, "> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        question = line.strip()
        if not question:
            continue

        if question.startswith(":"):
            command, _, argument = question[1:].partition(" ")
            command, argument = command.casefold(), argument.strip()
            if command in {"quit", "exit", "q"}:
                break
            if command in {"help", "h", "?"}:
                print(HELP)
            elif command == "lists":
                show_lists(datasets, selected)
            elif command == "use":
                try:
                    chosen = resolve_choice(argument, datasets)
                except ValueError as exc:
                    print(f"  {exc}")
                    continue
                selected = chosen
                label = next((item[1] for item in datasets if item[0] == selected), "automatic")
                conversation_id = None
                print(f"  asking about {label}; conversation reset")
            elif command == "auto":
                selected, conversation_id = None, None
                print("  Nia will choose the list per question; conversation reset")
            elif command == "new":
                conversation_id = None
                print("  new conversation")
            elif command in {"plan", "raw", "trace"}:
                if last is None:
                    print("  nothing asked yet")
                elif command == "plan":
                    if last["plan"] is None:
                        print("  that turn compiled no plan; :trace says how far it got")
                    else:
                        print(json.dumps(last["plan"], indent=2, ensure_ascii=False))
                elif command == "trace":
                    print(json.dumps(last["trace"], indent=2, ensure_ascii=False))
                elif not last["calls"]:
                    print("  that turn made no provider calls")
                else:
                    for index, call in enumerate(last["calls"]):
                        print(f"--- provider call {index + 1} "
                              f"({round(call.get('wall_ms', 0))} ms) ---")
                        print(json.dumps(call, indent=2, ensure_ascii=False)[: args.raw_chars])
            elif command == "save":
                if not argument:
                    print("  :save needs a path")
                    continue
                target = Path(argument)
                target.write_text(
                    json.dumps(turns, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(f"  wrote {len(turns)} turn(s) to {target}")
            else:
                print(f"  unknown command {command!r}; :help lists them")
            continue

        if lost:
            rebuilt = await revive()
            if not rebuilt:
                continue
            gateway, service = rebuilt
            lost = False

        # A turn needs a list; the application's own selector is what picks one when the
        # user has not, and saying which it chose is the point of asking automatically.
        asking = selected or select_dataset_for_question(
            question, catalog, scope, catalog.default_people_file_id
        )
        if selected is None:
            chosen = next((item[1] for item in datasets if item[0] == asking), asking)
            print(f"  (asking the {chosen})")

        recorder.take()
        started = time.perf_counter()
        try:
            result = await service.answer(
                scope, question, selected_file_id=asking, conversation_id=conversation_id
            )
        except (OSError, asyncpg.InterfaceError) as exc:
            # A WSL distro shuts down when it has been idle and takes the database with
            # it, and the address can change when it comes back. Reconnecting re-reads
            # the address -- and the asking itself wakes the distro -- so the question
            # costs a wait rather than being lost. Everything holding the pool is
            # rebuilt, keeping this conversation's memory.
            recorder.take()
            print(f"  database unreachable ({type(exc).__name__}); reconnecting")
            lost = True
            rebuilt = await revive()
            if not rebuilt:
                continue
            gateway, service = rebuilt
            lost = False
            started = time.perf_counter()
            try:
                result = await service.answer(
                    scope, question, selected_file_id=asking, conversation_id=conversation_id
                )
            except Exception as retried:  # noqa: BLE001 - report and keep the console alive
                print(f"  that question still failed: {type(retried).__name__}: {retried}")
                recorder.take()
                continue
        except Exception as exc:  # noqa: BLE001 - a console should outlive one bad turn
            print(f"  turn failed: {type(exc).__name__}: {exc}")
            recorder.take()
            continue
        wall_ms = (time.perf_counter() - started) * 1000
        calls = recorder.take()
        provider_ms = sum(call.get("wall_ms", 0.0) for call in calls)
        conversation_id = result.conversation_id

        print()
        print(result.answer or "(no answer)")
        print()
        detail = f" · {result.detail}" if result.detail else ""
        print(
            f"  {result.status}{detail} · {wall_ms / 1000:.1f}s"
            f" (local {max(wall_ms - provider_ms, 0) / 1000:.2f}s,"
            f" provider {provider_ms / 1000:.1f}s over {len(calls)} call"
            f"{'' if len(calls) == 1 else 's'})"
        )
        if result.status != "answered":
            print(f"  review: {result.review_status} · planner: {result.planner_type}")
        print()

        last = {
            "plan": serializable(result.plan),
            "trace": serializable(result.trace),
            "calls": calls,
        }
        turns.append(
            {
                "question": question,
                "file_id": result.selected_file_id or asking,
                "status": result.status,
                "answer": result.answer,
                "wall_ms": round(wall_ms),
                "provider_ms": round(provider_ms),
                "local_ms": round(max(wall_ms - provider_ms, 0)),
                "plan": last["plan"],
                "trace": last["trace"],
            }
        )

    if getattr(provider, "_client", None) is not None:
        await provider._client.aclose()
    await close_pool()
    print(f"{len(turns)} question(s) asked.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-level",
        default="CRITICAL",
        help=(
            "application logging. The default keeps the conversation readable; "
            "the turn's own status and :trace carry what went wrong"
        ),
    )
    parser.add_argument(
        "--database",
        default=NEWEST_DATABASE,
        help=(
            f"database name on the configured host (default {NEWEST_DATABASE}, the most "
            "recent data; any other name on that host works, and a name without the access "
            "migrations is refused with the ones that would serve)"
        ),
    )
    parser.add_argument("--database-url", help="full DSN, for a host that is not the configured one")
    parser.add_argument("--file-id", type=int, help="source list to ask about, skipping the prompt")
    parser.add_argument("--file-ids", default="49,91,93,94", help="the authorized scope")
    parser.add_argument("--principal", default="principal-researcher")
    parser.add_argument("--planner-mode", default="ai", help="ai, or empty to leave configured")
    parser.add_argument("--raw-chars", type=int, default=6000, help=":raw payload cut-off")
    return parser


def main() -> None:
    import logging

    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.CRITICAL))
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
