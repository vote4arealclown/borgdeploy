"""Tests for the SMB inventory and assimilation pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from borg.db import Database
from borg.modules.smb_inventory import SMBInventory
from borg.safety import ActionKind


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    database = Database(db_url)
    yield database


class FakeSMB:
    """In-memory SMB filesystem used to monkeypatch SMBInventory primitives."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.registered = False

    def register_session(self, *args, **kwargs) -> None:
        self.registered = True

    def unregister_session(self, host: str) -> None:
        self.registered = False

    def listdir(self, smb_path: str) -> list[dict[str, Any]]:
        # smb_path looks like \\host\share\rel
        rel = self._extract_rel(smb_path)
        target = self.root / rel
        out = []
        for entry in sorted(target.iterdir()):
            st = entry.stat()
            out.append(
                {
                    "name": entry.name,
                    "is_dir": entry.is_dir(),
                    "is_file": entry.is_file(),
                    "size": st.st_size if entry.is_file() else 0,
                    "mtime": st.st_mtime,
                }
            )
        return out

    def read_file(self, smb_path: str, max_bytes: int) -> bytes:
        rel = self._extract_rel(smb_path)
        target = self.root / rel
        return target.read_bytes()[:max_bytes]

    def _extract_rel(self, smb_path: str) -> str:
        # Strip leading \\host\share
        parts = smb_path.lstrip("\\").split("\\", 2)
        if len(parts) < 3 or not parts[2]:
            return "."
        return parts[2].replace("\\", "/")


@pytest.fixture
def fake_smb(tmp_path):
    root = tmp_path / "smb_root"
    root.mkdir()
    (root / "strategies").mkdir()
    (root / "strategies" / "rsi_strategy.py").write_text(
        "class RSIStrategy:\n    def analyze(self, data):\n        return []\n", encoding="utf-8"
    )
    (root / "README.md").write_text("# Projects\n", encoding="utf-8")
    (root / "big.bin").write_bytes(b"\x00" * 2_000_000)
    return FakeSMB(root)


@pytest.fixture
def inventory(fresh_db, fake_smb, monkeypatch):
    from borg.versioning import Versioning

    inv = SMBInventory(database=fresh_db, versioner=Versioning(database=fresh_db))
    monkeypatch.setattr(inv, "_register_session", fake_smb.register_session)
    monkeypatch.setattr(inv, "_unregister_session", lambda host: fake_smb.unregister_session(host))
    monkeypatch.setattr(inv, "_listdir", fake_smb.listdir)
    monkeypatch.setattr(inv, "_read_file", fake_smb.read_file)
    return inv


@pytest.mark.asyncio
async def test_scan_creates_inventory_entries(inventory, fresh_db) -> None:
    result = await inventory.scan(host="10.0.0.100", share="projects", username="u", password="p")
    assert result["status"] == "ok"
    assert result["files"] == 3
    assert result["dirs"] == 1

    entries = fresh_db.get_inventory_entries(source="smb://10.0.0.100/projects")
    assert len(entries) == 4  # 3 files + 1 dir
    paths = {e["rel_path"] for e in entries}
    assert "README.md" in paths
    assert "strategies" in paths
    assert "strategies/rsi_strategy.py" in paths
    assert "big.bin" in paths


@pytest.mark.asyncio
async def test_scan_skips_oversized_binary_files(inventory) -> None:
    await inventory.scan(host="10.0.0.100", share="projects", username="u", password="p")
    big = next(
        e for e in inventory.db.get_inventory_entries() if e["rel_path"] == "big.bin"
    )
    assert big["content"] is None or big["content"] == ""


@pytest.mark.asyncio
async def test_score_candidates_updates_status(inventory, fresh_db, monkeypatch) -> None:
    entry_id = fresh_db.insert_inventory_entry(
        {
            "source": "smb://10.0.0.100/projects",
            "rel_path": "strategies/rsi_strategy.py",
            "name": "rsi_strategy.py",
            "entry_type": "file",
            "size_bytes": 100,
            "content_hash": "abc",
            "content": "class RSIStrategy:\n    pass\n",
            "metadata": {},
        }
    )

    async def offline():
        return False

    from borg.llm import llm

    monkeypatch.setattr(llm, "_check_ollama", offline)

    result = await inventory.score_candidates(source="smb://10.0.0.100/projects")
    assert result["scored"] == 1
    updated = inventory.db.get_inventory_entry(entry_id)
    assert updated["assimilation_status"] == "scored"
    assert updated["assimilation_score"] is not None
    assert updated["assimilation_score"] > 0


@pytest.mark.asyncio
async def test_stage_candidate_creates_version_and_safety_check(inventory, fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(
        inventory.db,
        "get_inventory_entries",
        lambda *args, **kwargs: [],
    )
    entry_id = fresh_db.insert_inventory_entry(
        {
            "source": "smb://10.0.0.100/projects",
            "rel_path": "strategies/rsi_strategy.py",
            "name": "rsi_strategy.py",
            "entry_type": "file",
            "size_bytes": 100,
            "content_hash": "abc",
            "content": "class RSIStrategy:\n    pass\n",
            "metadata": {},
        }
    )

    result = await inventory.stage_candidate(entry_id)
    assert result["status"] == "staged"
    assert result["version_id"] > 0
    assert "borg/assimilated/" in result["target_path"]

    entry = fresh_db.get_inventory_entry(entry_id)
    assert entry["assimilation_status"] == "staged"
    assert entry["version_id"] == result["version_id"]

    versions = fresh_db.list_versions()
    assert any(v["id"] == result["version_id"] and v["status"] == "proposed" for v in versions)


@pytest.mark.asyncio
async def test_apply_candidate_requires_staged_status(inventory, fresh_db, monkeypatch) -> None:
    entry_id = fresh_db.insert_inventory_entry(
        {
            "source": "smb://10.0.0.100/projects",
            "rel_path": "strategies/rsi_strategy.py",
            "name": "rsi_strategy.py",
            "entry_type": "file",
            "size_bytes": 100,
            "content_hash": "abc",
            "content": "class RSIStrategy:\n    pass\n",
            "metadata": {},
        }
    )
    with pytest.raises(ValueError, match="not staged"):
        await inventory.apply_candidate(entry_id)


def test_heuristic_score_prefers_strategies_and_python(inventory) -> None:
    strat = {"name": "rsi_strategy.py", "rel_path": "strategies/rsi_strategy.py"}
    score = inventory._heuristic_score(strat, "class X: pass")
    assert score > 50

    cache = {"name": "foo.pyc", "rel_path": "__pycache__/foo.pyc"}
    assert inventory._heuristic_score(cache, "abc") <= 0


def test_parse_score_response(inventory) -> None:
    score, reason = inventory._parse_score_response("SCORE: 75 REASON: Useful strategy module")
    assert score == 75.0
    assert "Useful" in (reason or "")

    assert inventory._parse_score_response("garbage") == (None, None)
