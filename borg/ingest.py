"""File ingestion watcher for /borg/input and local input path."""
from __future__ import annotations

import asyncio
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from borg.config import settings
from borg.db import Database, db
from borg.events import EventLog, event_log
from borg.schemas import CandleInput
from borg.safety import ActionKind, safety


class IngestWatcher:
    """Polls the input directory for CSV/JSON files and imports them."""

    def __init__(
        self,
        database: Database = db,
        events: EventLog = event_log,
    ) -> None:
        self.db = database
        self.events = events
        self.input_paths = [Path(settings.input_path).resolve()]
        # Also watch legacy /borg/input if it exists and is different
        legacy = Path("/borg/input").resolve()
        if legacy.exists() and legacy not in self.input_paths:
            self.input_paths.append(legacy)
        self.processed_dir = Path(settings.output_path).resolve() / "processed"
        self._running = False
        self._interval = 5

    def _ensure_dirs(self) -> None:
        for p in self.input_paths:
            p.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _parse_csv(self, path: Path) -> tuple[int, list[tuple[int, str]]]:
        inserted = 0
        errors: list[tuple[int, str]] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=1):
                try:
                    candle = CandleInput(**row)
                    self.db.insert_candle(candle.model_dump())
                    inserted += 1
                except Exception as exc:
                    errors.append((row_num, str(exc)))
        return inserted, errors

    def _parse_json(self, path: Path) -> tuple[int, list[tuple[int, str]]]:
        inserted = 0
        errors: list[tuple[int, str]] = []
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        for row_num, row in enumerate(data, start=1):
            try:
                candle = CandleInput(**row)
                self.db.insert_candle(candle.model_dump())
                inserted += 1
            except Exception as exc:
                errors.append((row_num, str(exc)))
        return inserted, errors

    async def process_file(self, path: Path) -> dict[str, Any]:
        safety.check(ActionKind.INGEST, {"path": str(path), "size": path.stat().st_size})

        ext = path.suffix.lower()
        if ext == ".csv":
            inserted, errors = self._parse_csv(path)
        elif ext == ".json":
            inserted, errors = self._parse_json(path)
        elif ext == ".jsonl":
            inserted, errors = 0, []
            for row_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    candle = CandleInput(**json.loads(line))
                    self.db.insert_candle(candle.model_dump())
                    inserted += 1
                except Exception as exc:
                    errors.append((row_num, str(exc)))
        else:
            return {"status": "skipped", "path": str(path), "reason": "unsupported extension"}

        # Move to processed dir with timestamp prefix to avoid collisions
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        dest = self.processed_dir / f"{timestamp}-{path.name}"
        shutil.move(str(path), str(dest))

        result = {
            "status": "ok" if not errors else "partial",
            "path": str(path),
            "inserted": inserted,
            "errors": errors[:20],
        }
        self.events.emit(
            f"Ingested {path.name}: {inserted} rows, {len(errors)} errors",
            category="ingest",
            phase="observe",
            metadata=result,
        )
        return result

    async def scan(self) -> list[dict[str, Any]]:
        self._ensure_dirs()
        results: list[dict[str, Any]] = []
        for input_path in self.input_paths:
            for entry in sorted(input_path.iterdir()):
                if entry.is_file() and entry.suffix.lower() in (".csv", ".json", ".jsonl"):
                    try:
                        results.append(await self.process_file(entry))
                    except Exception as exc:
                        self.events.emit(f"Ingest failed for {entry.name}: {exc}", category="ingest", phase="idle", level="ERROR")
                        results.append({"status": "error", "path": str(entry), "error": str(exc)})
        return results

    async def run(self) -> None:
        self._running = True
        while self._running:
            await self.scan()
            await asyncio.sleep(self._interval)

    def stop(self) -> None:
        self._running = False


watcher = IngestWatcher()
