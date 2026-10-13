from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend" / "app"

FORBIDDEN_IMPORTS = (
    "standalone_jwt",
    "WebsiteAuthAdapter",
    "pyjwt",
    "jwt.encode",
)

CORE_DIRS = (
    "data_gateway",
    "planning",
    "retrieval",
    "memory",
    "llm",
    "tts",
    "stt",
    "realtime",
    "execution",
    "concurrency",
    "observability",
)


def test_core_layers_do_not_import_standalone_auth():
    offenders: list[str] = []
    for directory in CORE_DIRS:
        path = BACKEND / directory
        if not path.exists():
            continue
        for file in path.rglob("*.py"):
            text = file.read_text(encoding="utf-8")
            for needle in FORBIDDEN_IMPORTS:
                if needle in text:
                    offenders.append(f"{file.relative_to(ROOT)} contains {needle}")
    assert offenders == []
