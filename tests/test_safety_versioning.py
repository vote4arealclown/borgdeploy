"""Smoke tests for safety gates and versioning workflow."""
from __future__ import annotations

import pytest

from borg.safety import ActionKind, SafetyGate
from borg.versioning import Versioning


@pytest.fixture
def fresh_db(tmp_path):
    from borg.db import Database

    db_url = f"sqlite:///{tmp_path / 'safety_test.db'}"
    database = Database(db_url)
    yield database


@pytest.fixture
def safety_gate(fresh_db, monkeypatch):
    monkeypatch.setattr("borg.config.settings.require_confirmation_for", ["self_modify", "delete"])
    return SafetyGate(database=fresh_db)


def test_safety_requires_confirmation_for_self_modify(safety_gate) -> None:
    assert safety_gate.needs_confirmation(ActionKind.SELF_MODIFY) is True
    decision = safety_gate.check(ActionKind.SELF_MODIFY, {"path": "borg/brain.py"})
    assert decision["requires_confirmation"] is True
    assert decision["approved"] is False


def test_safety_auto_approves_forecast(safety_gate) -> None:
    assert safety_gate.needs_confirmation(ActionKind.FORECAST) is False
    decision = safety_gate.check(ActionKind.FORECAST)
    assert decision["requires_confirmation"] is False
    assert decision["approved"] is True


def test_versioning_propose_and_apply(fresh_db, tmp_path, monkeypatch) -> None:
    from borg.safety import SafetyGate

    safety_gate = SafetyGate(database=fresh_db)
    safety_gate.required = set()  # auto-approve for test
    versioner = Versioning(database=fresh_db, safety_gate=safety_gate)
    # Point project root at tmp_path so we don't modify real source
    versioner.project_root = tmp_path
    target = tmp_path / "test_module.py"
    original = "# original\n"
    target.write_text(original, encoding="utf-8")

    new_content = "# original\n# improved\n"
    proposal = versioner.propose("test", "test_module.py", new_content)
    assert proposal["requires_confirmation"] is False

    result = versioner.apply(proposal["version_id"])
    assert result["status"] == "applied"
    assert target.read_text(encoding="utf-8") == new_content


def test_versioning_reject(fresh_db, tmp_path, monkeypatch) -> None:
    from borg.safety import SafetyGate

    safety_gate = SafetyGate(database=fresh_db)
    safety_gate.required = set()
    versioner = Versioning(database=fresh_db, safety_gate=safety_gate)
    versioner.project_root = tmp_path
    target = tmp_path / "test_module.py"
    target.write_text("# original\n", encoding="utf-8")

    proposal = versioner.propose("test", "test_module.py", "# changed\n")
    result = versioner.reject(proposal["version_id"], reason="test")
    assert result["status"] == "rejected"
