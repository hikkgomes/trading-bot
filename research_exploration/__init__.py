"""Exploratory multi-timeframe strategy research workflow.

This package is a *clean-room* research layer for discovering BTC/USDT intraday,
day-trading, swing and scalping strategies from structured multi-timeframe
hypotheses. It deliberately does **not** depend on (or feed) the old
condition-grid searches, ``active_strategies*.json`` or the live paper bots.

Flow:  market idea -> precise hypothesis -> controlled test -> keep/reject -> log

Modules
-------
feature_inventory   What data/features exist across every timeframe (lightweight).
hypothesis_schema   The REGIME + SETUP + TRIGGER + EXIT + RISK grammar.
strategy_families   Named market-behaviour families (trend, breakout, reversion, ...).
hypothesis_generator  Concrete candidate hypotheses built from the families.
evaluate            Adapter: align timeframes, turn a hypothesis into trades.
experiment_log      Append-only record of what was tested and the verdict.
"""
