"""One-command offline release qualification for NIA.

Run from ``backend``::

    python -m benchmarks.release_check

The command runs compileall, the complete pytest suite, the offline planner release evaluator,
and Ruff when Ruff is installed. External PostgreSQL/LiveKit tests may be skipped by pytest in an
offline environment; the final report preserves pytest's exit code and output.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> dict[str, object]:
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    env = dict(os.environ)
    backend_text = str(BACKEND_ROOT)
    current = env.get("PYTHONPATH", "")
    if backend_text not in current.split(os.pathsep):
        env["PYTHONPATH"] = backend_text + (os.pathsep + current if current else "")

    steps: dict[str, dict[str, object]] = {}
    steps["compileall"] = _run(
        [sys.executable, "-m", "compileall", "-q", "backend/app", "backend/benchmarks", "tests"],
        cwd=REPO_ROOT,
        env=env,
    )
    steps["pytest"] = _run([sys.executable, "-m", "pytest", "-q"], cwd=REPO_ROOT, env=env)
    steps["release_evaluation"] = _run(
        [
            sys.executable,
            "-m",
            "benchmarks.release_evaluation",
            "--iterations",
            str(args.iterations),
        ],
        cwd=BACKEND_ROOT,
        env=env,
    )
    ruff = shutil.which("ruff")
    if ruff:
        steps["ruff"] = _run(
            [ruff, "check", "backend/app", "backend/benchmarks", "tests"],
            cwd=REPO_ROOT,
            env=env,
        )
    else:
        steps["ruff"] = {
            "command": ["ruff", "check", "backend/app", "backend/benchmarks", "tests"],
            "exit_code": None,
            "stdout": "",
            "stderr": "Ruff is not installed; lint gate not executed.",
            "status": "skipped",
        }

    required = ("compileall", "pytest", "release_evaluation")
    passed = all(steps[name]["exit_code"] == 0 for name in required)
    if ruff:
        passed = passed and steps["ruff"]["exit_code"] == 0
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "offline_release_gate": "pass" if passed else "fail",
        "steps": steps,
        "note": "Live model/database qualification is intentionally separate from this offline gate.",
    }
    text = json.dumps(report, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
