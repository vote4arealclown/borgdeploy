"""Entry point for the Borg prototype."""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone

import uvicorn

from borg.backtest_engine import BacktestEngine
from borg.brain import brain
from borg.config import settings
from borg.coordinator import StrategyCoordinator
from borg.db import db
from borg.learning import LearningEngine
from borg.llm import llm
from borg.memory import memory
from borg.modules.smb_inventory import inventory
from borg.web.app import app

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("borg")


async def _brain_loop() -> None:
    logger.info("Starting brain loop; interval=%ss", settings.brain_interval_seconds)
    await brain.seed()
    await brain.run()


async def _run_backtest(start: str, end: str) -> None:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    coordinator = StrategyCoordinator(database=db, llm=llm, mem=memory)
    engine = BacktestEngine(coordinator=coordinator, database=db, memory=memory)
    results = await engine.run_backtest(start_dt, end_dt)
    logger.info(
        "Backtest %s to %s: %d trades, %.1f%% win rate, Sharpe %.2f, Max DD %.2f%%",
        start,
        end,
        results["trade_count"],
        results["win_rate"] * 100,
        results["sharpe"],
        results["max_dd"] * 100,
    )
    print(results)


async def _run_learn(dry_run: bool, validate: bool) -> None:
    coordinator = StrategyCoordinator(database=db, llm=llm, mem=memory)
    engine = BacktestEngine(coordinator=coordinator, database=db, memory=memory)
    learner = LearningEngine(coordinator=coordinator)

    train_end = datetime.now(timezone.utc)
    train_start = train_end - timedelta(days=90)
    val_start = train_end
    val_end = val_start + timedelta(days=10)

    logger.info("Training backtest: %s to %s", train_start, train_end)
    train_results = await engine.run_backtest(train_start, train_end, record_episodes=True)
    logger.info(
        "Training: %d trades, %.1f%% win, Sharpe %.2f",
        train_results["trade_count"],
        train_results["win_rate"] * 100,
        train_results["sharpe"],
    )

    current_weights = coordinator.get_weights()
    proposed = learner.propose_weight_updates(train_results, current_weights)
    logger.info("Proposed weight updates: %s", proposed)

    if dry_run:
        print({"proposed_updates": proposed, "train_results": train_results})
        return

    coordinator.apply_weights(proposed)

    if validate:
        logger.info("Validation backtest: %s to %s", val_start, val_end)
        val_results = await engine.run_backtest(val_start, val_end, record_episodes=False)
        logger.info(
            "Validation: %d trades, %.1f%% win, Sharpe %.2f",
            val_results["trade_count"],
            val_results["win_rate"] * 100,
            val_results["sharpe"],
        )
        safe = learner.validate_updates(proposed, train_results, val_results)
        if not safe:
            logger.warning("Learning updates failed validation; rolling back")
            coordinator.apply_weights(current_weights)
            print({"status": "rolled_back", "proposed_updates": proposed})
            return
        print({"status": "deployed", "proposed_updates": proposed, "validation_results": val_results})
    else:
        print({"status": "deployed_without_validation", "proposed_updates": proposed})


def main() -> None:
    parser = argparse.ArgumentParser(description="Borg Prototype")
    parser.add_argument(
        "command",
        choices=["web", "brain", "all", "scan-smb", "backtest", "learn"],
        default="all",
        nargs="?",
        help="Run web server, brain loop, both, scan SMB share, run backtest, or run learning",
    )
    parser.add_argument("--host", default=settings.web_host)
    parser.add_argument("--port", type=int, default=settings.web_port)
    parser.add_argument("--smb-host", default=settings.smb_host)
    parser.add_argument("--smb-share", default=settings.smb_share)
    parser.add_argument("--smb-username", default=settings.smb_username)
    parser.add_argument("--smb-password", default=settings.smb_password)
    parser.add_argument("--smb-domain", default=settings.smb_domain)
    parser.add_argument("--smb-root", default="", help="Root path inside the SMB share to scan")
    parser.add_argument("--score-limit", type=int, default=100, help="Max files to score after scan")
    parser.add_argument("--heuristic", action="store_true", help="Use fast heuristic scoring instead of LLM")
    parser.add_argument("--start", default=(datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat())
    parser.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--dry-run", action="store_true", help="Propose learning updates without applying")
    parser.add_argument("--validate", action="store_true", help="Validate learning updates out-of-sample")
    args = parser.parse_args()

    if args.command == "brain":
        asyncio.run(_brain_loop())
    elif args.command == "web":
        uvicorn.run(app, host=args.host, port=args.port, log_level=settings.log_level.lower())
    elif args.command == "backtest":
        asyncio.run(_run_backtest(args.start, args.end))
    elif args.command == "learn":
        asyncio.run(_run_learn(args.dry_run, args.validate))
    elif args.command == "scan-smb":
        # smbprotocol is very chatty at INFO; raise it to WARNING for scans.
        logging.getLogger("smbprotocol").setLevel(logging.WARNING)
        result = asyncio.run(
            inventory.scan(
                host=args.smb_host,
                share=args.smb_share,
                username=args.smb_username,
                password=args.smb_password,
                domain=args.smb_domain,
                root_path=args.smb_root,
            )
        )
        logger.info("SMB scan result: %s", result)
        if args.score_limit > 0:
            score_result = asyncio.run(
                inventory.score_candidates(
                    source=f"smb://{args.smb_host}/{args.smb_share}",
                    limit=args.score_limit,
                    heuristic_only=args.heuristic,
                )
            )
            logger.info("SMB score result: %s", score_result)
        return
    else:
        # Run brain loop in background and web server in foreground
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        brain_task = loop.create_task(_brain_loop())
        config = uvicorn.Config(app, host=args.host, port=args.port, log_level=settings.log_level.lower())
        server = uvicorn.Server(config)
        try:
            loop.run_until_complete(server.serve())
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            brain.stop()
            brain_task.cancel()
            try:
                loop.run_until_complete(brain_task)
            except asyncio.CancelledError:
                pass
            loop.close()


if __name__ == "__main__":
    main()
