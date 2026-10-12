"""Slice 6 release-candidate contracts."""

from __future__ import annotations

import asyncio
import sys

import pytest

from repo_paths import BACKEND_ROOT, monorepo_file  # noqa: E402

sys.path.insert(0, str(BACKEND_ROOT))

from app.config import Settings
from benchmarks.release_evaluation import SCOPED_SUITE, _suite_summary, evaluate


def test_release_defaults_keep_ai_and_mandatory_review():
    fields = Settings.model_fields
    assert fields["planner_mode"].default == "ai"
    assert fields["planner_self_review"].default is True
    assert fields["planner_requirement_check"].default is True
    assert fields["planner_review_mode"].default == "always"


def test_release_scoped_suite_is_all_critical_and_covers_all_four_lists():
    if not SCOPED_SUITE.exists():
        pytest.skip(f"{SCOPED_SUITE.name} is published with the standalone repository")
    summary = _suite_summary()
    assert summary["cases"] == 12
    assert summary["all_critical"] is True
    assert summary["all_have_semantic_contract"] is True
    assert summary["selected_file_ids"] == [49, 91, 93, 94]


def test_offline_release_evaluator_passes_and_separates_live_evidence():
    if not SCOPED_SUITE.exists():
        pytest.skip(f"{SCOPED_SUITE.name} is published with the standalone repository")
    report = asyncio.run(evaluate(2))
    assert report["release_gate"] == "pass"
    assert report["model_call_policy"]["greeting"]["model_calls"] == 0
    assert report["model_call_policy"]["next_page"]["model_calls"] == 0
    assert report["model_call_policy"]["verify"]["model_calls"] == 0
    assert report["model_call_policy"]["fresh_query"]["model_calls"] == 2
    assert report["live_qualification"]["current_code_model_accuracy"] == "unmeasured"
    assert report["live_qualification"]["historical_evidence_is_not_current_qualification"] is True


def test_readme_describes_primary_plus_review_not_single_call():
    readme = monorepo_file("README.md").read_text(encoding="utf-8")
    assert "primary planner call plus mandatory semantic review" in readme
    assert "one structured planner call, selected-list scope only" not in readme
