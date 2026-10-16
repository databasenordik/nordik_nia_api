"""The prompt console's own plumbing, which is all it is allowed to decide by itself.

The console builds no scope rules and relaxes no boundary: it resolves which database and
which list to talk to, and the runtime refuses the rest for itself. These are the parts
where a mistake would point someone at the wrong data or print a password.
"""

from __future__ import annotations

import os
import sys

import pytest

from repo_paths import BACKEND_ROOT

sys.path.insert(0, str(BACKEND_ROOT))

from benchmarks.ask import dsn_for, redacted, resolve_choice, serializable, with_host  # noqa: E402

DATASETS = [
    (49, "Student master list", 2814),
    (91, "Confirmed deaths", 84),
    (93, "Additional Deaths", 29),
]


class Args:
    def __init__(self, database=None, database_url=None):
        self.database = database
        self.database_url = database_url


def test_a_database_name_keeps_the_configured_host_and_credentials(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("ASSISTANT_DATABASE_URL", "postgresql://runtime:pw@db.example:5432/nia")
    get_settings.cache_clear()
    try:
        assert dsn_for(Args(database="nia_v6")) == "postgresql://runtime:pw@db.example:5432/nia_v6"
        assert dsn_for(Args()) == "postgresql://runtime:pw@db.example:5432/nia"
        # An explicit DSN is used as given, so another host stays reachable.
        other = "postgresql://someone:else@10.0.0.9:6543/live"
        assert dsn_for(Args(database="ignored", database_url=other)) == other
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "dsn,host,expected",
    [
        (
            "postgresql://runtime:pw@localhost:5432/nia_v6",
            "172.17.0.2",
            "postgresql://runtime:pw@172.17.0.2:5432/nia_v6",
        ),
        (
            "postgresql://runtime:pw@localhost/nia_v6",
            "172.17.0.2",
            "postgresql://runtime:pw@172.17.0.2/nia_v6",
        ),
        ("postgresql:///nia_v6", "172.17.0.2", "postgresql:///nia_v6"),
    ],
)
def test_only_the_host_moves_when_an_address_is_substituted(dsn, host, expected):
    assert with_host(dsn, host) == expected


def test_a_printed_address_never_carries_the_password():
    printed = redacted("postgresql://runtime:s3cr3t@localhost:5432/nia_v6")
    assert "s3cr3t" not in printed
    assert printed == "postgresql://runtime:***@localhost:5432/nia_v6"
    # Nothing to redact, nothing changed.
    assert redacted("postgresql:///nia_v6") == "postgresql:///nia_v6"


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("49", 49),  # a file id
        ("2", 91),  # the position in the printed list
        ("confirmed", 91),  # part of the label
        ("CONFIRMED DEATHS", 91),
        ("", None),  # blank means let Nia choose
        ("  ", None),
    ],
)
def test_a_list_can_be_chosen_by_id_position_or_name(reply, expected):
    assert resolve_choice(reply, DATASETS) == expected


@pytest.mark.parametrize("reply", ["7", "123", "deaths", "nothing like this"])
def test_an_ambiguous_or_absent_choice_is_refused_rather_than_guessed(reply):
    """"deaths" matches two lists, and the console must not pick one of them."""
    with pytest.raises(ValueError):
        resolve_choice(reply, DATASETS)


def test_a_file_id_outside_the_printed_lists_is_refused():
    """94 is a real list elsewhere; it is not one of these, so it is not offered."""
    with pytest.raises(ValueError):
        resolve_choice("94", DATASETS)


def test_saved_turns_survive_json(tmp_path):
    from app.planning.plan_schema import PlanOp, PlanScope, PlanStep, QueryPlan

    plan = QueryPlan(
        scope=PlanScope(file_ids=(49,), version_mode="current", authorized_only=True),
        goals=["count"],
        steps=[PlanStep(id="scope", op=PlanOp.USE_CURRENT_VERSION, input="current_records")],
    )
    import json

    encoded = json.dumps(serializable({"plan": plan, "when": tmp_path}), ensure_ascii=False)
    assert "USE_CURRENT_VERSION" in encoded
    assert json.loads(encoded)["plan"]["goals"] == ["count"]


def test_the_default_database_is_the_most_recent_one_not_the_configured_snapshot(monkeypatch):
    """Asking a question should not start with a flag.

    ASSISTANT_DATABASE_URL points at an older snapshot that the runtime's own boundary
    check refuses, so the console defaults to the database holding the current source
    rows and the migrations applied after them.
    """
    from app.config import get_settings
    from benchmarks.ask import NEWEST_DATABASE, build_parser

    monkeypatch.setenv("ASSISTANT_DATABASE_URL", "postgresql://runtime:pw@db.example:5432/nia")
    get_settings.cache_clear()
    try:
        default = build_parser().parse_args([])
        assert default.database == NEWEST_DATABASE
        assert dsn_for(default) == f"postgresql://runtime:pw@db.example:5432/{NEWEST_DATABASE}"
        # And any other name on that host is still one flag away.
        assert dsn_for(build_parser().parse_args(["--database", "nia_v6"])) == (
            "postgresql://runtime:pw@db.example:5432/nia_v6"
        )
    finally:
        get_settings.cache_clear()


def test_the_repository_env_is_found_from_any_working_directory(monkeypatch, tmp_path):
    """Run from backend/benchmarks, the console used to find no configuration at all.

    Settings name their env files by relative path, so from the directory the script
    lives in nothing resolved, every setting fell back to its default, and the DSN came
    out as "/nia_eval_20261016".
    """
    from benchmarks.ask import REPO_ROOT, load_repository_env

    if not (REPO_ROOT / ".env").exists():
        pytest.skip("no .env in this checkout")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ASSISTANT_DATABASE_URL", raising=False)
    load_repository_env()
    assert os.environ["ASSISTANT_DATABASE_URL"].startswith(("postgresql://", "postgres://"))


def test_an_exported_setting_still_wins_over_the_file(monkeypatch, tmp_path):
    from benchmarks.ask import REPO_ROOT, load_repository_env

    if not (REPO_ROOT / ".env").exists():
        pytest.skip("no .env in this checkout")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSISTANT_DATABASE_URL", "postgresql://mine:pw@elsewhere:5432/chosen")
    load_repository_env()
    assert os.environ["ASSISTANT_DATABASE_URL"] == "postgresql://mine:pw@elsewhere:5432/chosen"
