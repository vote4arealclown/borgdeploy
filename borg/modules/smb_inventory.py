"""SMB share inventory and assimilation pipeline.

Scans an SMB share, inventories files and code, scores candidates for
assimilation, and stages approved candidates through the versioning/safety
system before any code is copied into Borg.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Optional

from borg.config import settings
from borg.db import Database, db
from borg.events import EventLog, event_log
from borg.llm import llm
from borg.memory import Memory, memory
from borg.safety import ActionKind, SafetyGate, safety
from borg.versioning import Versioning, versioning


class SMBInventory:
    """Connects to an SMB share, inventories content, and manages assimilation candidates."""

    DEFAULT_CODE_EXTENSIONS: frozenset[str] = frozenset(
        [".py", ".js", ".ts", ".sql", ".yaml", ".yml", ".json", ".md", ".txt", ".sh"]
    )

    def __init__(
        self,
        database: Database = db,
        events: EventLog = event_log,
        mem: Memory = memory,
        safety_gate: SafetyGate = safety,
        versioner: Versioning = versioning,
    ) -> None:
        self.db = database
        self.events = events
        self.memory = mem
        self.safety = safety_gate
        self.versioning = versioner
        self._session_registered = False

    # ------------------------------------------------------------------
    # SMB primitives (extracted so tests can monkeypatch easily)
    # ------------------------------------------------------------------
    def _register_session(
        self,
        host: str,
        username: str,
        password: str,
        domain: str = "",
        port: int = 445,
    ) -> None:
        try:
            import smbclient
        except ImportError as exc:
            raise RuntimeError("smbprotocol is not installed") from exc
        # smbclient.register_session does not accept a domain keyword; embed it
        # in the username when provided (DOMAIN\user or user@DOMAIN).
        if domain:
            user = f"{domain}\\{username}"
        else:
            user = username
        smbclient.register_session(
            host,
            username=user,
            password=password,
            port=port,
        )
        self._session_registered = True

    def _unregister_session(self, host: str) -> None:
        try:
            import smbclient

            smbclient.delete_session(host)
        except Exception:
            pass
        self._session_registered = False

    def _smb_path(self, host: str, share: str, rel_path: str) -> str:
        rel = rel_path.replace("/", "\\").strip("\\")
        base = f"\\\\{host}\\{share}"
        if rel:
            return f"{base}\\{rel}"
        return base

    def _listdir(self, smb_path: str) -> list[dict[str, Any]]:
        import smbclient

        out = []
        for entry in smbclient.scandir(smb_path):
            out.append(
                {
                    "name": entry.name,
                    "is_dir": entry.is_dir(),
                    "is_file": entry.is_file(),
                    "size": entry.stat().st_size if entry.is_file() else 0,
                    "mtime": entry.stat().st_mtime if hasattr(entry.stat(), "st_mtime") else None,
                }
            )
        return out

    def _read_file(self, smb_path: str, max_bytes: int) -> bytes:
        import smbclient

        with smbclient.open_file(smb_path, mode="rb", share_access="r") as f:
            return f.read(max_bytes)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------
    def _is_code_file(self, name: str) -> bool:
        if "." not in name:
            return False
        ext = name[name.rfind(".") :].lower()
        return ext in set(settings.smb_code_extensions) or ext in self.DEFAULT_CODE_EXTENSIONS

    def _should_skip(self, rel_path: str) -> bool:
        """Return True if the path matches a configured skip pattern."""
        parts = rel_path.replace("\\", "/").split("/")
        for pattern in settings.smb_skip_patterns:
            if pattern in parts:
                return True
        return False

    def _hash_content(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _metadata(self, entry: dict[str, Any]) -> dict[str, Any]:
        meta: dict[str, Any] = {"is_dir": entry["is_dir"], "is_file": entry["is_file"]}
        if entry.get("mtime"):
            try:
                meta["mtime"] = datetime.fromtimestamp(entry["mtime"], tz=timezone.utc).isoformat()
            except Exception:
                meta["mtime"] = str(entry["mtime"])
        return meta

    async def scan(
        self,
        host: Optional[str] = None,
        share: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        domain: Optional[str] = None,
        root_path: str = "",
    ) -> dict[str, Any]:
        """Recursively scan the SMB share and persist inventory entries."""
        host = host or settings.smb_host
        share = share or settings.smb_share
        username = username or settings.smb_username
        password = password or settings.smb_password
        domain = domain if domain is not None else settings.smb_domain
        source = f"smb://{host}/{share}"

        self.events.emit(
            f"Starting SMB inventory scan of {source}",
            category="inventory",
            phase="observe",
        )

        self._register_session(host, username, password, domain)
        try:
            scanned = await self._scan_recursive(source, host, share, root_path)
        finally:
            self._unregister_session(host)

        summary = {
            "status": "ok",
            "source": source,
            "files": scanned["files"],
            "dirs": scanned["dirs"],
            "errors": scanned["errors"],
        }
        self.events.emit(
            f"SMB inventory scan complete: {scanned['files']} files, {scanned['dirs']} dirs, {len(scanned['errors'])} errors",
            category="inventory",
            phase="idle",
            metadata=summary,
        )
        return summary

    async def _scan_recursive(
        self,
        source: str,
        host: str,
        share: str,
        rel_path: str,
    ) -> dict[str, Any]:
        totals = {"files": 0, "dirs": 0, "errors": []}
        smb_path = self._smb_path(host, share, rel_path)

        try:
            entries = self._listdir(smb_path)
        except Exception as exc:
            err = f"listdir failed for {smb_path}: {exc}"
            totals["errors"].append(err)
            self.events.emit(err, category="inventory", phase="idle", level="ERROR")
            return totals

        for entry in entries:
            name = entry["name"]
            if name in (".", ".."):
                continue

            child_rel = f"{rel_path}/{name}" if rel_path else name
            if self._should_skip(child_rel):
                continue

            child_smb = self._smb_path(host, share, child_rel)

            if entry["is_dir"]:
                self.db.insert_inventory_entry(
                    {
                        "source": source,
                        "rel_path": child_rel.replace("\\", "/"),
                        "name": name,
                        "entry_type": "dir",
                        "size_bytes": 0,
                        "content_hash": None,
                        "content": None,
                        "metadata": self._metadata(entry),
                    }
                )
                totals["dirs"] += 1
                child_totals = await self._scan_recursive(source, host, share, child_rel)
                totals["files"] += child_totals["files"]
                totals["dirs"] += child_totals["dirs"]
                totals["errors"].extend(child_totals["errors"])
                continue

            content: Optional[str] = None
            content_hash: Optional[str] = None
            size = entry.get("size") or 0
            if entry["is_file"] and self._is_code_file(name) and size <= settings.smb_max_file_size_bytes:
                try:
                    raw = self._read_file(child_smb, settings.smb_max_file_size_bytes)
                    content_hash = self._hash_content(raw)
                    content = raw.decode("utf-8", errors="replace")
                except Exception as exc:
                    err = f"read failed for {child_smb}: {exc}"
                    totals["errors"].append(err)
                    self.events.emit(err, category="inventory", phase="idle", level="ERROR")

            self.db.insert_inventory_entry(
                {
                    "source": source,
                    "rel_path": child_rel.replace("\\", "/"),
                    "name": name,
                    "entry_type": "file",
                    "size_bytes": size,
                    "content_hash": content_hash,
                    "content": content,
                    "metadata": self._metadata(entry),
                }
            )
            totals["files"] += 1

        return totals

    # ------------------------------------------------------------------
    # Scoring / assimilation decision
    # ------------------------------------------------------------------
    async def score_candidates(
        self,
        source: Optional[str] = None,
        limit: int = 100,
        heuristic_only: bool = False,
    ) -> dict[str, Any]:
        """Score pending inventory entries and mark the most useful as candidates."""
        entries = self.db.get_inventory_entries(source=source, status="pending", limit=limit)
        files = [
            e
            for e in entries
            if e["entry_type"] == "file"
            and e.get("content")
            and not self._should_skip(e.get("rel_path", ""))
        ]
        if not files:
            return {"status": "ok", "scored": 0, "message": "no pending files to score"}

        ollama_up = await llm._check_ollama() and not heuristic_only
        scored = 0
        for entry in files:
            score, reason = await self._score_entry(entry, ollama_up)
            self.db.update_inventory_status(
                entry["id"],
                "scored" if score is not None else "pending",
                score=score,
                reason=reason,
            )
            scored += 1

        return {"status": "ok", "scored": scored}

    async def _score_entry(self, entry: dict[str, Any], ollama_up: bool) -> tuple[Optional[float], Optional[str]]:
        content = entry.get("content") or ""
        snippet = content[:2000]
        prompt = (
            f"Borg is evaluating whether to assimilate an external file into its own codebase.\n"
            f"File path: {entry['rel_path']}\n"
            f"Size: {entry.get('size_bytes', 0)} bytes\n"
            f"Snippet:\n{snippet}\n\n"
            f"Rate how useful this file would be to Borg on a scale of 0 (useless) to 100 (highly useful). "
            f"Respond with exactly one line in this format:\n"
            f"SCORE: <number> REASON: <short reason>"
        )

        if ollama_up:
            try:
                raw = (await llm.generate(prompt, timeout=30.0)).strip()
                return self._parse_score_response(raw)
            except Exception as exc:
                self.events.emit(
                    f"LLM scoring failed for {entry['rel_path']}: {exc}",
                    category="inventory",
                    phase="idle",
                    level="ERROR",
                )

        # Deterministic fallback heuristic
        score = self._heuristic_score(entry, content)
        reason = f"Fallback heuristic score based on size and code signals."
        return score, reason

    def _parse_score_response(self, raw: str) -> tuple[Optional[float], Optional[str]]:
        try:
            if "SCORE:" in raw:
                score_part = raw.split("SCORE:", 1)[1]
                if "REASON:" in score_part:
                    score_str, reason = score_part.split("REASON:", 1)
                else:
                    score_str, reason = score_part, ""
                score = float(score_str.strip().split()[0])
                score = max(0.0, min(100.0, score))
                return score, reason.strip()
        except Exception:
            pass
        return None, None

    def _heuristic_score(self, entry: dict[str, Any], content: str) -> float:
        score = 30.0
        name = (entry.get("name") or "").lower()
        rel = (entry.get("rel_path") or "").lower()
        signals = [
            (name.endswith(".py"), 25),
            ("strategy" in rel, 15),
            ("borg" in rel, 15),
            ("test" in name, -10),
            ("__pycache__" in rel, -100),
            ("node_modules" in rel, -100),
            (".git/" in rel, -100),
            (len(content) > 500, 10),
            (len(content) > 5000, 10),
            ("def " in content or "class " in content, 10),
        ]
        for active, delta in signals:
            if active:
                score += delta
        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # Staging & applying assimilation
    # ------------------------------------------------------------------
    async def stage_candidate(self, entry_id: int) -> dict[str, Any]:
        """Stage an inventory entry for assimilation. Creates a version proposal."""
        entry = self.db.get_inventory_entry(entry_id)
        if not entry:
            raise ValueError(f"Inventory entry {entry_id} not found")
        if entry["entry_type"] != "file":
            raise ValueError("Only file entries can be assimilated")
        if not entry.get("content"):
            raise ValueError("Entry has no readable content")

        # Compute a Borg-relative path. By default place assimilated modules under
        # borg/assimilated/<source>/<rel_path> to keep them isolated.
        rel = entry["rel_path"].replace("\\", "/").lstrip("/")
        safe_rel = rel.replace("../", "").replace("..\\", "")
        target_path = f"borg/assimilated/{entry['source'].replace('://', '_').replace('/', '_')}/{safe_rel}"

        proposal = self.versioning.propose(
            module=f"assimilated:{entry['name']}",
            relative_path=target_path,
            new_content=entry["content"],
        )

        self.db.update_inventory_status(
            entry_id,
            "staged",
            reason=f"Staged for assimilation as {target_path}",
            version_id=proposal["version_id"],
        )

        decision = self.safety.check(
            ActionKind.ASSIMILATE,
            {
                "entry_id": entry_id,
                "version_id": proposal["version_id"],
                "target_path": target_path,
            },
        )

        self.events.emit(
            f"Staged assimilation candidate {entry_id} -> {target_path} (version {proposal['version_id']})",
            category="inventory",
            phase="idle",
            metadata={"requires_confirmation": decision["requires_confirmation"]},
        )

        return {
            "status": "staged",
            "entry_id": entry_id,
            "version_id": proposal["version_id"],
            "target_path": target_path,
            "requires_confirmation": decision["requires_confirmation"],
        }

    async def apply_candidate(self, entry_id: int, actor: str = "user") -> dict[str, Any]:
        """Apply a staged assimilation after safety approval."""
        entry = self.db.get_inventory_entry(entry_id)
        if not entry:
            raise ValueError(f"Inventory entry {entry_id} not found")
        if entry.get("assimilation_status") != "staged":
            raise ValueError(f"Entry {entry_id} is {entry.get('assimilation_status')}, not staged")

        version_id = entry.get("version_id")
        if not version_id:
            raise ValueError(f"Entry {entry_id} has no associated version")

        result = self.versioning.apply(version_id, actor=actor)
        self.db.update_inventory_status(entry_id, "applied", reason=f"Applied by {actor}")
        await self.memory.observe(
            f"Assimilated {entry['rel_path']} from {entry['source']} into {result['path']}",
            kind="episode",
            source="assimilation",
        )
        return {"status": "applied", "entry_id": entry_id, "version_id": version_id, "path": result["path"]}

    async def reject_candidate(self, entry_id: int, actor: str = "user", reason: str = "") -> dict[str, Any]:
        """Reject a staged assimilation."""
        entry = self.db.get_inventory_entry(entry_id)
        if not entry:
            raise ValueError(f"Inventory entry {entry_id} not found")
        version_id = entry.get("version_id")
        if version_id:
            self.versioning.reject(version_id, actor=actor, reason=reason)
        self.db.update_inventory_status(entry_id, "rejected", reason=reason or "Rejected by user")
        return {"status": "rejected", "entry_id": entry_id}


inventory = SMBInventory()
