"""Consciousness report generator: human-readable self-reflection."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from borg.llm import llm
from borg.self_analysis import SelfAnalysisEngine


class ConsciousnessReporter:
    """Generate daily/weekly consciousness reports from self-analysis."""

    def __init__(
        self,
        self_analysis: SelfAnalysisEngine,
        llm_client: Any = llm,
    ) -> None:
        self.self_analysis = self_analysis
        self.llm = llm_client

    def _format_section(self, title: str, data: Any) -> str:
        return f"{title}:\n{json.dumps(data, indent=2, default=str)}"

    async def generate_daily_report(self) -> str:
        """Generate a daily consciousness report."""
        analysis = self.self_analysis.analyze_performance("daily")
        insights = self.self_analysis.identify_insights(analysis)
        risks = self.self_analysis.surface_risks(analysis)

        if analysis.get("trades_count", 0) == 0:
            return f"""# BORG Daily Consciousness Report
Generated: {datetime.now(timezone.utc).isoformat()}

## Performance
No trades in the last 24 hours.

## Self-Reflection
Borg was observant but did not act.
"""

        prompt = f"""You are an autonomous trading agent reflecting on your daily performance.

PERFORMANCE SUMMARY:
- Trades: {analysis['trades_count']} ({analysis['win_count']} wins, {analysis['loss_count']} losses)
- Win Rate: {analysis['win_rate'] * 100:.1f}%
- Total P&L (avg per trade * count): ${analysis['total_pnl']:.4f}
- Sharpe: {analysis['sharpe']:.2f}
- Max Drawdown: {analysis['max_dd'] * 100:.1f}%

{self._format_section('BY STRATEGY', analysis.get('by_strategy', {}))}

{self._format_section('BY REGIME', analysis.get('by_regime', {}))}

INSIGHTS:
{"\n".join(f"- {i}" for i in insights)}

RISKS:
{"\n".join(f"- {r}" for r in risks)}

Write a brief (3-5 paragraphs) consciousness report reflecting on:
1. What did I do well?
2. What didn't work?
3. What did I learn?
4. What am I uncertain about?
5. What will I do differently tomorrow?

Be objective, humble, and specific. Reference the data above."""

        if await self.llm._check_ollama():
            reflection = (await self.llm.generate(prompt)).strip()
        else:
            reflection = (
                f"Borg traded {analysis['trades_count']} times with a "
                f"{analysis['win_rate'] * 100:.1f}% win rate. "
                f"{'Strengths outweighed weaknesses.' if analysis['win_rate'] > 0.5 else 'Performance was mixed.'} "
                f"Key uncertainty: whether current regime persistence will continue."
            )

        return f"""# BORG Daily Consciousness Report
Generated: {datetime.now(timezone.utc).isoformat()}

## Performance
- Trades: {analysis['trades_count']} (Win rate: {analysis['win_rate'] * 100:.1f}%)
- P&L: ${analysis['total_pnl']:.4f}
- Sharpe: {analysis['sharpe']:.2f}
- Max Drawdown: {analysis['max_dd'] * 100:.1f}%

## Self-Reflection
{reflection}

## Top Insight
{insights[0] if insights else "No clear pattern"}

## Key Risk
{risks[0] if risks else "No identified risks"}
"""

    async def generate_weekly_report(self) -> str:
        """Generate a weekly consciousness report."""
        analysis = self.self_analysis.analyze_performance("weekly")
        insights = self.self_analysis.identify_insights(analysis)
        risks = self.self_analysis.surface_risks(analysis)

        if analysis.get("trades_count", 0) == 0:
            return f"""# BORG Weekly Consciousness Report
Generated: {datetime.now(timezone.utc).isoformat()}

## Overview
No trades this week.

## Reflection
Borg observed markets but remained on the sidelines.
"""

        prompt = f"""You are an autonomous trading agent reflecting on your weekly performance.

WEEK SUMMARY:
- Trades: {analysis['trades_count']}
- Win Rate: {analysis['win_rate'] * 100:.1f}%
- Total P&L: ${analysis['total_pnl']:.4f}
- Sharpe: {analysis['sharpe']:.2f}

{self._format_section('BY STRATEGY', analysis.get('by_strategy', {}))}

{self._format_section('BY REGIME', analysis.get('by_regime', {}))}

INSIGHTS:
{"\n".join(f"- {i}" for i in insights)}

Write a weekly consciousness report:
1. Overall assessment of the week
2. Which strategies outperformed?
3. Which regimes were challenging?
4. What patterns did I notice?
5. What's my confidence for next week?

Be honest about weaknesses and uncertainties."""

        if await self.llm._check_ollama():
            reflection = (await self.llm.generate(prompt)).strip()
        else:
            reflection = (
                f"Weekly summary: {analysis['trades_count']} trades, "
                f"{analysis['win_rate'] * 100:.1f}% win rate. "
                f"Best performers and regimes are listed above."
            )

        return f"""# BORG Weekly Consciousness Report
Generated: {datetime.now(timezone.utc).isoformat()}

## Overview
{analysis['trades_count']} trades, {analysis['win_rate'] * 100:.1f}% win rate, ${analysis['total_pnl']:.4f} P&L

## Reflection
{reflection}

## Key Risks
{"\n".join(f"- {r}" for r in risks) if risks else "- No major risks identified"}
"""
