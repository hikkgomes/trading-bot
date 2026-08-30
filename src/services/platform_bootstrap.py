"""Idempotent first-run authority for a new platform database."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import Any

from sqlalchemy import insert, select

from src.data.database import (
    account,
    account_snapshot,
    balance_snapshot,
    cost_model_manifest,
    feature_manifest,
    feature_set,
    platform_bootstrap,
    strategy_definition,
    strategy_version,
)
from src.data.universe import InstrumentObservation, SqlUniverseStore, UniverseEligibilityPolicy
from src.domain._codec import canonical_hash, timestamp, to_primitive
from src.domain.instruments import Instrument, MarketType
from src.domain.strategies import StrategyDefinition, StrategySourceType
from src.research.artefacts import StrategyArtefact
from src.research.canonical import (
    SqlActiveStrategyAssignmentRepository,
    SqlStrategyArtefactRepository,
)
from src.research.datasets import (
    RESEARCH_BUNDLE_ROLES,
    CanonicalResearchDatasetBuilder,
)
from src.risk.engine import SqlRiskPolicyStore
from src.risk.policies import install_product_risk_policies
from src.services.scheduler import DatabaseJobQueue


def _record(connection, table, values: Mapping[str, Any]) -> bool:
    identity = str(values["id"])
    existing = connection.execute(select(table).where(table.c.id == identity)).mappings().first()
    if existing is not None:
        if any(existing[key] != value for key, value in values.items()):
            raise ValueError(f"bootstrap identity collision in {table.name}: {identity}")
        return False
    connection.execute(insert(table).values(**dict(values)))
    return True


def _instrument(product_id: str) -> Instrument:
    spot = product_id == "btc_accumulation"
    return Instrument(
        venue="binance",
        market_type=MarketType.SPOT if spot else MarketType.FUTURES,
        base_asset="BTC",
        quote_asset="USDT",
        settlement_asset=None if spot else "USDT",
        exchange_symbol="BTCUSDT",
        price_precision=2,
        quantity_precision=6,
        minimum_quantity=0.000001,
        minimum_notional=5.0,
    )


def _diagnostic_dataset(
    *, product_id: str, instrument: Instrument, start: dt.datetime
) -> dict[str, Any]:
    bars: list[dict[str, Any]] = []
    returns = []
    bar_steps = []
    prior_close = 100_000.0
    for index in range(240):
        available_at = (start + dt.timedelta(hours=index + 1)).replace(microsecond=0).isoformat()
        close = 100_000.0 + index * 10.0 + (25.0 if index % 7 == 0 else 0.0)
        bars.append(
            {
                "available_at": available_at,
                "open": prior_close,
                "high": max(prior_close, close) + 20.0,
                "low": min(prior_close, close) - 20.0,
                "close": close,
                "volume": 100.0 + index,
            }
        )
        returns.append(close / prior_close - 1.0)
        bar_steps.append(
            {
                "timestamp": available_at,
                "prices": {instrument.instrument_id: close},
                "target_fractions": {instrument.instrument_id: 0.1 if index % 24 < 12 else 0.0},
                "funding_rates": {
                    instrument.instrument_id: 0.0001
                    if instrument.market_type is MarketType.FUTURES
                    else 0.0
                },
            }
        )
        prior_close = close
    event_start = start.replace(microsecond=0)
    replay_events = [
        {
            "event_time": (event_start + dt.timedelta(seconds=index)).isoformat(),
            "receive_time": (
                event_start + dt.timedelta(seconds=index, milliseconds=100)
            ).isoformat(),
            "instrument_id": instrument.instrument_id,
            "best_bid": 99_999.0 + index,
            "best_ask": 100_001.0 + index,
            "bid_depth": 10.0,
            "ask_depth": 10.0,
            "traded_at_ask": 1.0 if index == 1 else 0.0,
            "mark_price": 100_000.0 + index,
            "funding_rate": 0.0001 if instrument.market_type is MarketType.FUTURES else 0.0,
        }
        for index in range(3)
    ]
    ml_rows = [
        {
            "available_at": bar["available_at"],
            "return_1": returns[index],
            "range_fraction": (float(bar["high"]) - float(bar["low"])) / float(bar["close"]),
            "label": float(returns[index] > 0),
        }
        for index, bar in enumerate(bars[:80])
    ]
    return {
        "schema": "platform.diagnostic_dataset/v1",
        "diagnostic": True,
        "promotable": False,
        "synthetic": True,
        "product_id": product_id,
        "market_frame": bars,
        "returns": returns,
        "fee_bps": 5.0,
        "slippage_bps": 2.0,
        "funding_rate": 0.0001 if instrument.market_type is MarketType.FUTURES else 0.0,
        "features_valid": True,
        "causality_valid": True,
        "symbol_returns": {instrument.exchange_symbol: returns},
        "bar_steps": {
            "steps": bar_steps,
            "initial_equity": 10_000.0,
            "fee_bps": 5.0,
        },
        "event_replay": {
            "events": replay_events,
            "orders": [
                {
                    "order_id": f"diagnostic-replay:{product_id}",
                    "instrument_id": instrument.instrument_id,
                    "side": "buy",
                    "quantity": 0.1,
                    "limit_price": 100_001.0,
                    "submitted_at": event_start.isoformat(),
                    "expires_at": (event_start + dt.timedelta(seconds=10)).isoformat(),
                }
            ],
            "cancel_latency_seconds": 0.25,
            "impact_bps_per_depth_fraction": 5.0,
        },
        "ml_rows": {
            "rows": ml_rows,
            "model_name": "logistic_regression",
            "feature_names": ["return_1", "range_fraction"],
            "target_name": "label",
            "train_fraction": 0.7,
            "embargo_rows": 1,
        },
    }


def _diagnostic_stage_payloads(data: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Partition the deterministic bootstrap sample into role-local inputs."""

    bars = list(data.get("market_frame", ()))
    returns = list(data.get("returns", ()))
    steps = data.get("bar_steps")
    step_rows = list(steps.get("steps", ())) if isinstance(steps, Mapping) else []
    rows = data.get("ml_rows")
    ml_rows = list(rows.get("rows", ())) if isinstance(rows, Mapping) else []
    chunks: dict[str, Mapping[str, Any]] = {}
    size = max(1, len(bars) // len(RESEARCH_BUNDLE_ROLES))
    for index, role in enumerate(RESEARCH_BUNDLE_ROLES):
        start = index * size
        end = len(bars) if index == len(RESEARCH_BUNDLE_ROLES) - 1 else (index + 1) * size
        payload = dict(data)
        payload["role"] = role
        payload["market_frame"] = bars[start:end]
        payload["returns"] = returns[start:end]
        step_payload = dict(steps) if isinstance(steps, Mapping) else {}
        payload["bar_steps"] = {
            **step_payload,
            "steps": step_rows[start:end],
        }
        ml_payload = dict(rows) if isinstance(rows, Mapping) else {}
        payload["ml_rows"] = {
            **ml_payload,
            "rows": ml_rows[start:end],
        }
        event_replay = data.get("event_replay")
        if isinstance(event_replay, Mapping):
            payload["event_replay"] = {
                **dict(event_replay),
                "events": list(event_replay.get("events", ()))[start:end],
            }
        chunks[role] = payload
    return chunks


class PlatformBootstrap:
    """Create the minimum safe graph needed for autonomous platform progress.

    Every record is content-addressed or immutable. Re-running this class is
    safe and returns the same identities after the first successful run.
    """

    def __init__(
        self,
        *,
        engine,
        configuration: Mapping[str, Mapping[str, Any]],
        node_id: str = "linux-optiplex",
    ) -> None:
        self.engine = engine
        self.configuration = configuration
        self.node_id = node_id
        self.products = {
            str(item["product_id"]): dict(item) for item in configuration["products"]["products"]
        }
        self.accounts = {
            str(item["account_id"]): dict(item) for item in configuration["accounts"]["accounts"]
        }

    def ensure(self, *, now: str | None = None) -> dict[str, Any]:
        observed_at = timestamp(now or dt.datetime.now(dt.UTC), field="now")
        product_ids = tuple(sorted(self.products))
        config_hash = canonical_hash({"schema": "platform.bootstrap/v1", "node_id": self.node_id})
        with self.engine.begin() as connection:
            marker = (
                connection.execute(
                    select(platform_bootstrap).where(
                        platform_bootstrap.c.id == "platform-bootstrap:v1"
                    )
                )
                .mappings()
                .first()
            )
            if marker is not None:
                if marker["content_hash"] != config_hash:
                    raise ValueError(
                        "platform bootstrap configuration changed after initialisation"
                    )
                observed_at = timestamp(str(marker["created_at"]), field="bootstrap.created_at")
            else:
                _record(
                    connection,
                    platform_bootstrap,
                    {
                        "id": "platform-bootstrap:v1",
                        "created_at": observed_at,
                        "content_hash": config_hash,
                        "payload": {
                            "schema": "platform.bootstrap/v1",
                            "configuration_hash": config_hash,
                            "node_id": self.node_id,
                            "products": list(product_ids),
                        },
                    },
                )
            for account_id, raw_account in self.accounts.items():
                if (
                    connection.execute(
                        select(account.c.id).where(account.c.id == account_id)
                    ).first()
                    is None
                ):
                    connection.execute(
                        insert(account).values(
                            id=account_id,
                            created_at=observed_at,
                            payload={
                                **raw_account,
                                "account_state_authority": "authenticated_rest"
                                if raw_account.get("environment") == "production"
                                else "paper_config",
                            },
                        )
                    )

        result: dict[str, Any] = {
            "reason_code": "platform_bootstrap_completed",
            "configuration_hash": config_hash,
            "products": {},
        }
        for product_id in product_ids:
            result["products"][product_id] = self._ensure_product(product_id, observed_at)
        return result

    def _ensure_product(self, product_id: str, observed_at: str) -> dict[str, Any]:
        product = self.products[product_id]
        account_config = self.accounts[str(product["account_id"])]
        instrument = _instrument(product_id)
        universe_id = str(product["universe_id"])
        universe_store = SqlUniverseStore(self.engine)
        policy = UniverseEligibilityPolicy()
        universe_snapshot_id = universe_store.record_snapshot(
            universe_id=universe_id,
            observed_at=observed_at,
            observations=(
                InstrumentObservation(
                    instrument=instrument,
                    listing_age_days=3650.0,
                    quote_volume=1_000_000_000.0,
                    trade_count=1_000_000,
                    spread_bps=1.0,
                    open_interest=1_000_000_000.0,
                    funding_rate=0.0,
                    realised_volatility=0.2,
                    depth_notional=10_000_000.0,
                    data_completeness=1.0,
                    strategy_eligibility=("diagnostic",),
                ),
            ),
            policy=policy,
        )
        manifest_payload = {
            "schema": "platform.feature_manifest/v1",
            "version": "core-bars-v1",
            "required_nodes": ["bar_return"],
            "market_type": "spot" if instrument.market_type is MarketType.SPOT else "futures",
        }
        feature_manifest_id = canonical_hash(manifest_payload)
        cost_payload = {
            "schema": "platform.cost_model/v1",
            "product_id": product_id,
            "execution_costs": dict(product["execution_costs"]),
        }
        cost_model_id = canonical_hash(cost_payload)
        parameter_set_id = canonical_hash(
            {"schema": "platform.parameter_set/v1", "product_id": product_id, "parameters": {}}
        )
        with self.engine.begin() as connection:
            _record(
                connection,
                feature_manifest,
                {
                    "id": feature_manifest_id,
                    "created_at": observed_at,
                    "payload": manifest_payload,
                },
            )
            _record(
                connection,
                feature_set,
                {
                    "id": canonical_hash({"feature_manifest_id": feature_manifest_id}),
                    "created_at": observed_at,
                    "payload": {"feature_manifest_id": feature_manifest_id, **manifest_payload},
                },
            )
            _record(
                connection,
                cost_model_manifest,
                {
                    "id": cost_model_id,
                    "created_at": observed_at,
                    "payload": cost_payload,
                },
            )
        interval_end = (dt.datetime.fromisoformat(observed_at) - dt.timedelta(days=1)).replace(
            microsecond=0
        )
        interval_start = interval_end - dt.timedelta(days=365)
        data = _diagnostic_dataset(
            product_id=product_id,
            instrument=instrument,
            start=interval_start,
        )
        stage_payloads = _diagnostic_stage_payloads(data)
        total_seconds = (interval_end - interval_start).total_seconds()
        stage_intervals: dict[str, dict[str, str]] = {}
        for index, role in enumerate(RESEARCH_BUNDLE_ROLES):
            start = interval_start + dt.timedelta(seconds=total_seconds * index / 5)
            end = interval_start + dt.timedelta(seconds=total_seconds * (index + 1) / 5)
            stage_intervals[role] = {"start": start.isoformat(), "end": end.isoformat()}
        dataset_bundle = CanonicalResearchDatasetBuilder(self.engine).build(
            product_id,
            intervals=stage_intervals,
            payload_by_role=stage_payloads,
            universe_snapshot_id=universe_snapshot_id,
            feature_manifest_id=feature_manifest_id,
            cost_model_id=cost_model_id,
            parameter_set_id=parameter_set_id,
            instrument_scope=(instrument.instrument_id,),
            availability_timestamp={
                **{role: observed_at for role in RESEARCH_BUNDLE_ROLES[:-1]},
                "forward_observation": (
                    dt.datetime.fromisoformat(observed_at) + dt.timedelta(hours=1)
                ).isoformat(),
            },
            created_at=observed_at,
            engine_version="platform-bootstrap/v1",
        )
        snapshot_id = dataset_bundle.stage_snapshot_ids["screening"]
        DatabaseJobQueue(self.engine).enqueue_if_absent(
            job_id=f"initial-universe-refresh:{product_id}",
            name="universe_refresh",
            payload={
                "schedule_name": "universe_refresh",
                "product_id": product_id,
                "universe_id": universe_id,
                "available_at": observed_at,
                "producer_identity": "platform-scheduler",
                "market_type": "spot" if instrument.market_type is MarketType.SPOT else "futures",
                "policy": {},
                "maximum_symbols": 100,
            },
            available_at=observed_at,
            priority=300,
            producer_identity="platform-bootstrap",
        )
        DatabaseJobQueue(self.engine).enqueue_if_absent(
            job_id=f"initial-dataset:{product_id}:{snapshot_id}",
            name="dataset_snapshot_validate",
            payload={
                "dataset_snapshot_id": snapshot_id,
                "product_id": product_id,
                "feature_manifest_id": feature_manifest_id,
                "cost_model_id": cost_model_id,
                "parameter_set_id": parameter_set_id,
                "producer_identity": "platform-bootstrap",
            },
            available_at=observed_at,
            priority=90,
            producer_identity="platform-bootstrap",
        )
        definition = StrategyDefinition(
            identity=f"platform-bootstrap:{product_id}:diagnostic",
            version="diagnostic-v1",
            # Keep the bootstrap assignment inside an existing strategy family.
            # The diagnostic behaviour is identified by its immutable metadata,
            # not by introducing a new strategy family.
            family="time_series",
            product=product_id,
            universe={"snapshot_id": universe_snapshot_id, "symbols": [instrument.exchange_symbol]},
            data_requirements={"bars": "1m", "closed_only": True},
            feature_graph={"version": "core-bars-v1", "required_nodes": ["bar_return"]},
            signal_model={"diagnostic": True, "production_rule": "no_exposure"},
            position_model={"kind": "flat_diagnostic"},
            execution_preferences={"policy": "paper_only"},
            risk_policy={"id": str(product["risk_policy_id"])},
            validation_policy={"promotable": False},
            source_type=StrategySourceType.GENERATED_DSL,
            source_hash=canonical_hash({"bootstrap": "diagnostic-v1", "product_id": product_id}),
            metadata={"diagnostic": True, "promotable": False},
        )
        with self.engine.begin() as connection:
            _record(
                connection,
                strategy_definition,
                {
                    "id": definition.definition_hash,
                    "identity": definition.identity,
                    "product_id": product_id,
                    "source_type": definition.source_type.value,
                    "source_hash": definition.source_hash,
                    "definition": to_primitive(definition),
                },
            )
            _record(
                connection,
                strategy_version,
                {
                    "id": definition.strategy_version_id,
                    "definition_id": definition.definition_hash,
                    "version": definition.version,
                    "created_at": observed_at,
                    "payload": {"definition_hash": definition.definition_hash},
                },
            )
        artefact = StrategyArtefact(
            definition=definition,
            dependency_hash=canonical_hash({"bootstrap": "dependencies-v1"}),
            dataset_snapshot_hashes=(snapshot_id,),
            feature_set_version="core-bars-v1",
            cost_model_version="bootstrap-costs-v1",
            validation_evidence={"diagnostic": True, "promotable": False},
            holdout_claim={"diagnostic": True, "promotable": False},
            promotion_policy={"automatic": False, "promotable": False},
            position_limits={"maximum_position": 0.0},
            risk_limits={"risk_policy_id": str(product["risk_policy_id"])},
            model_hashes=(),
            supported_products=(product_id,),
            supported_instruments=(instrument.instrument_id,),
            created_at=observed_at,
            metadata={"diagnostic": True, "promotable": False},
            authoritative_evidence={"diagnostic": True, "promotable": False},
            product_id=product_id,
            portfolio_id=str(product["portfolio_id"]),
            account_id=str(product["account_id"]),
            promotion_policy_id=str(product["promotion_policy_id"]),
            engine_version="platform-bootstrap/v1",
        )
        SqlStrategyArtefactRepository(self.engine).put(
            artefact.artefact_hash, artefact.to_dict(), created_at=observed_at
        )
        paper_balances = account_config.get("paper_starting_balances")
        paper_positions = account_config.get("paper_starting_positions", {})
        if product.get("execution_mode") == "paper":
            if not isinstance(paper_balances, Mapping) or not paper_balances:
                raise ValueError(
                    f"paper account {account_config['account_id']} needs explicit balances"
                )
            self._save_account_snapshot(
                product_id=product_id,
                account_config=account_config,
                balances={str(key): float(value) for key, value in paper_balances.items()},
                positions={str(key): float(value) for key, value in paper_positions.items()},
                observed_at=observed_at,
                source="paper_config",
            )
        assignment_id = SqlActiveStrategyAssignmentRepository(self.engine).assign(
            product_id=product_id,
            portfolio_id=str(product["portfolio_id"]),
            strategy_version_id=definition.strategy_version_id,
            artefact_hash=artefact.artefact_hash,
            lifecycle_state="development",
            execution_mode="paper",
            capital_limit=float(account_config.get("paper_capital_limit", 0.0) or 0.0),
            risk_budget=0.0,
            assigned_at=observed_at,
            assigned_by="platform-bootstrap",
            assignment_reason="isolated non-promotable diagnostic paper assignment",
            instrument_id=instrument.instrument_id,
            payload={
                "instrument_ids": [instrument.instrument_id],
                "diagnostic": True,
                "promotable": False,
            },
        )
        if product.get("execution_mode") == "paper":
            self._enqueue_diagnostic_round_trip(
                product_id=product_id,
                product=product,
                instrument=instrument,
                assignment_id=assignment_id,
                strategy_version_id=definition.strategy_version_id,
                artefact_hash=artefact.artefact_hash,
                observed_at=observed_at,
            )
        install_product_risk_policies(
            SqlRiskPolicyStore(self.engine),
            risk_configuration=self.configuration["risk"],
            products=self.products,
        )
        return {
            "universe_snapshot_id": universe_snapshot_id,
            "dataset_snapshot_id": snapshot_id,
            "feature_manifest_id": feature_manifest_id,
            "strategy_version_id": definition.strategy_version_id,
            "artefact_hash": artefact.artefact_hash,
            "assignment_id": assignment_id,
        }

    def _enqueue_diagnostic_round_trip(
        self,
        *,
        product_id: str,
        product: Mapping[str, Any],
        instrument: Instrument,
        assignment_id: str,
        strategy_version_id: str,
        artefact_hash: str,
        observed_at: str,
    ) -> None:
        open_order_id = canonical_hash(
            {"schema": "platform.diagnostic_paper/v1", "product_id": product_id, "phase": "open"}
        )
        close_order_id = canonical_hash(
            {
                "schema": "platform.diagnostic_paper/v1",
                "product_id": product_id,
                "phase": "close",
                "open_order_id": open_order_id,
            }
        )
        payload = {
            "schema": "platform.diagnostic_paper/v1",
            "diagnostic": True,
            "product_id": product_id,
            "instrument_id": instrument.instrument_id,
            "assignment_id": assignment_id,
            "strategy_version_id": strategy_version_id,
            "artefact_hash": artefact_hash,
            "event_id": f"platform-bootstrap:diagnostic:{product_id}",
            "open_order_id": open_order_id,
            "close_order_id": close_order_id,
            "quantity": 0.0001,
            "price": 100_000.0,
        }
        DatabaseJobQueue(self.engine).enqueue_if_absent(
            job_id=f"diagnostic-paper:open:{product_id}",
            name="diagnostic_paper_open",
            payload=payload,
            available_at=observed_at,
            priority=100,
            producer_identity="platform-bootstrap:diagnostic-paper",
        )

    def _save_account_snapshot(
        self,
        *,
        product_id: str,
        account_config: Mapping[str, Any],
        balances: Mapping[str, float],
        positions: Mapping[str, float],
        observed_at: str,
        source: str,
    ) -> str:
        account_id = str(account_config["account_id"])
        payload = {
            "account_id": account_id,
            "product_id": product_id,
            "balances": dict(balances),
            "free_balances": dict(balances),
            "positions": dict(positions),
            "regular_orders": [],
            "conditional_orders": [],
            "used_margin": 0.0,
            "maintenance_margin": 0.0,
            "used_margin_fraction": 0.0,
            "liquidation_buffer_fraction": 1.0,
            "account_mode": str(account_config.get("margin_mode", "cash")),
            "unknown_exposure": {},
            "account_state_known": True,
            "account_state_authority": source,
            "account_fingerprint": canonical_hash(
                {
                    "account_id": account_id,
                    "product_id": product_id,
                    "source": source,
                }
            ),
            "observed_at": observed_at,
        }
        content_hash = canonical_hash(payload)
        snapshot_id = canonical_hash(
            {
                "account_id": account_id,
                "observed_at": observed_at,
                "content_hash": content_hash,
            }
        )
        with self.engine.begin() as connection:
            _record(
                connection,
                account_snapshot,
                {
                    "id": snapshot_id,
                    "account_id": account_id,
                    "observed_at": observed_at,
                    "source": source,
                    "content_hash": content_hash,
                    "payload": payload,
                },
            )
            balance_id = canonical_hash(
                {"account_snapshot": snapshot_id, "balances": dict(balances)}
            )
            _record(
                connection,
                balance_snapshot,
                {
                    "id": balance_id,
                    "created_at": observed_at,
                    "payload": {
                        **payload,
                        "source": source,
                        "account_state_authority": source,
                    },
                },
            )
        return snapshot_id
