"""Track proposed, applied, and rejected code changes."""
from __future__ import annotations

import difflib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from borg.config import settings
from borg.db import Database, db
from borg.events import EventLog, event_log
from borg.safety import ActionKind, SafetyGate, safety


class Versioning:
    """Manages self-improvement diffs with explicit approval gates."""

    def __init__(
        self,
        database: Database = db,
        events: EventLog = event_log,
        safety_gate: SafetyGate = safety,
    ) -> None:
        self.db = database
        self.events = events
        self.safety = safety_gate
        self.project_root = settings.sqlite_path.parent.parent if settings.sqlite_path else Path(__file__).resolve().parent.parent

    def _read_file(self, relative_path: str) -> str:
        path = self.project_root / relative_path
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _write_file(self, relative_path: str, content: str) -> None:
        path = self.project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _compute_diff(self, original: str, updated: str) -> str:
        return "\n".join(difflib.unified_diff(original.splitlines(), updated.splitlines(), lineterm=""))

    def propose(
        self,
        module: str,
        relative_path: str,
        new_content: str,
        version: Optional[str] = None,
    ) -> dict[str, Any]:
        """Propose a code change; does NOT apply it."""
        original = self._read_file(relative_path)
        diff = self._compute_diff(original, new_content)
        version_str = version or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        version_id = self.db.insert_version(
            {
                "module": module,
                "version": version_str,
                "path": relative_path,
                "diff": diff,
                "new_content": new_content,
            }
        )
        decision = self.safety.check(ActionKind.SELF_MODIFY, {"version_id": version_id, "path": relative_path})
        self.events.emit(
            f"Proposed version {version_str} for {module} (id={version_id})",
            category="versioning",
            phase="idle",
            metadata={"path": relative_path, "requires_confirmation": decision["requires_confirmation"]},
        )
        return {
            "version_id": version_id,
            "module": module,
            "version": version_str,
            "path": relative_path,
            "requires_confirmation": decision["requires_confirmation"],
        }

    def apply(self, version_id: int, actor: str = "user") -> dict[str, Any]:
        """Apply a previously proposed version after safety approval."""
        row = next((v for v in self.db.list_versions(limit=1000) if v["id"] == version_id), None)
        if not row:
            raise ValueError(f"Version {version_id} not found")
        if row["status"] != "proposed":
            raise ValueError(f"Version {version_id} is {row['status']}, not proposed")

        self.safety.approve(ActionKind.SELF_MODIFY, actor=actor, detail={"version_id": version_id})

        updated = row.get("new_content")
        if not updated:
            original = self._read_file(row["path"])
            updated = self._apply_diff(original, row["diff"])
        self._write_file(row["path"], updated)
        self.db.update_version_status(version_id, "applied")
        self.events.emit(
            f"Applied version {row['version']} to {row['path']} (id={version_id})",
            category="versioning",
            phase="idle",
            metadata={"actor": actor},
        )
        self.db.audit(actor, "version:apply", {"version_id": version_id, "path": row["path"]})
        return {"status": "applied", "version_id": version_id, "path": row["path"]}

    async def record_learning_update(
        self,
        updates: dict[str, Any],
        performance_before: dict[str, Any],
        performance_after: dict[str, Any],
    ) -> dict[str, Any]:
        """Record autonomous learning updates in version history for rollback."""
        version_id = self.db.insert_version(
            {
                "module": "learning",
                "version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
                "path": "",
                "diff": "",
                "new_content": "",
            }
        )
        # Enrich the version row with learning-specific metadata via audit log.
        self.db.audit(
            "learning",
            "learning_update",
            {
                "version_id": version_id,
                "updates": updates,
                "performance": {
                    "before": performance_before,
                    "after": performance_after,
                },
            },
        )
        self.events.emit(
            f"Learning update v{version_id} recorded",
            category="versioning",
            phase="idle",
            metadata={"version_id": version_id, "updates_count": len(updates)},
        )
        return {"version_id": version_id, "status": "deployed"}

    def reject(self, version_id: int, actor: str = "user", reason: str = "") -> dict[str, Any]:
        row = next((v for v in self.db.list_versions(limit=1000) if v["id"] == version_id), None)
        if not row:
            raise ValueError(f"Version {version_id} not found")
        self.safety.reject(ActionKind.SELF_MODIFY, actor=actor, reason=reason)
        self.db.update_version_status(version_id, "rejected")
        self.events.emit(f"Rejected version {row['version']} (id={version_id})", category="versioning", phase="idle")
        return {"status": "rejected", "version_id": version_id}

    def _apply_diff(self, original: str, diff: str) -> str:
        """Best-effort application of a unified diff. Falls back to whole-file replacement if patch fails."""
        try:
            lines = original.splitlines()
            # Use PatchSet if available
            try:
                import whatthepatch
                patches = list(whatthepatch.parse_patch(diff))
                if patches:
                    for patch in patches:
                        _, new = whatthepatch.apply_diff(patch, lines)
                        lines = new
                    return "\n".join(lines)
            except Exception:
                pass
            # Fallback: simple naive patch
            result = self._naive_apply(lines, diff)
            return "\n".join(result)
        except Exception:
            # Last resort: if diff looks like a whole file, return diff content
            return diff

    def _naive_apply(self, lines: list[str], diff: str) -> list[str]:
        """Naive unified-diff applier for simple whole-file or hunk replacements."""
        result = list(lines)
        diff_lines = diff.splitlines()
        i = 0
        while i < len(diff_lines):
            line = diff_lines[i]
            if line.startswith("@@"):
                # Parse @@ -start,count +start,count @@
                try:
                    parts = line.split()
                    new_info = parts[1][1:]
                    if "," in new_info:
                        new_start, _ = map(int, new_info.split(","))
                    else:
                        new_start = int(new_info)

                    removed: list[str] = []
                    added: list[str] = []
                    i += 1
                    while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
                        dl = diff_lines[i]
                        if dl.startswith("-"):
                            removed.append(dl[1:])
                        elif dl.startswith("+"):
                            added.append(dl[1:])
                        elif dl.startswith(" "):
                            removed.append(dl[1:])
                            added.append(dl[1:])
                        i += 1

                    # Find the removed block in result
                    if removed and all(r in result for r in removed):
                        start_idx = -1
                        for j in range(len(result) - len(removed) + 1):
                            if result[j : j + len(removed)] == removed:
                                start_idx = j
                                break
                        if start_idx >= 0:
                            result = result[:start_idx] + added + result[start_idx + len(removed) :]
                    else:
                        # Insert at new_start if no context
                        insert_at = max(0, new_start - 1)
                        result = result[:insert_at] + added + result[insert_at:]
                    continue
                except Exception:
                    pass
            i += 1
        return result


versioning = Versioning()
