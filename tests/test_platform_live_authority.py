from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.data.database import PlatformDatabase
from src.data.universe import InstrumentObservation, SqlUniverseStore, UniverseEligibilityPolicy
from src.domain._codec import canonical_hash
from src.domain.instruments import Instrument, MarketType
from src.execution.config import ExchangeConfig
from src.research.canonical import (
    CanonicalEvidenceError,
    SqlApprovalRepository,
    SqlForwardEvidenceRepository,
    SqlPreflightRepository,
)
from src.services.config import load_split_configuration
from src.services.platform_live_authority import (
    PlatformLiveAuthority,
    PlatformLiveAuthorityError,
)
from src.services.platform_smoke import _seed_strategy

ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-08-29T10:00:00+00:00"
PREFLIGHT_AT = "2026-08-29T10:00:01+00:00"
APPROVED_AT = "2026-08-29T10:00:02+00:00"
ASSIGNED_AT = "2026-08-29T10:00:03+00:00"


class _Broker:
    def __init__(self, account_fingerprint: str) -> None:
        self.account_fingerprint = account_fingerprint

    @staticmethod
    def get_price(symbol: str) -> float:
        assert symbol == "BTCUSDT"
        return 100_000.0

    @staticmethod
    def supports_native_protective_stops() -> bool:
        return True

    def account_snapshot(self, *, expected_symbols: tuple[str, ...]) -> dict:
        assert expected_symbols == ("BTCUSDT",)
        return {
            "balances": {"USDT": 1_000.0},
            "free_balances": {"USDT": 1_000.0},
            "positions": {},
            "regular_orders": [],
            "conditional_orders": [],
            "used_margin": 0.0,
            "maintenance_margin": 0.0,
            "used_margin_fraction": 0.0,
            "liquidation_buffer_fraction": 1.0,
            "account_mode": "one_way",
            "unknown_exposure": {},
            "account_state_known": True,
            "account_state_authority": "authenticated_rest",
            "account_fingerprint": self.account_fingerprint,
        }


def _fixture(tmp_path: Path, monkeypatch) -> tuple[PlatformDatabase, PlatformLiveAuthority, dict]:
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'authority.db'}")
    database.migrate()
    configuration = load_split_configuration(ROOT / "config")
    configuration = copy.deepcopy(configuration)
    product = next(
        item
        for item in configuration["products"]["products"]
        if item["product_id"] == "active_income"
    )
    product["execution_mode"] = "live"
    account = next(
        item
        for item in configuration["accounts"]["accounts"]
        if item["account_id"] == product["account_id"]
    )
    account["environment"] = "testnet"
    monkeypatch.setenv("BINANCE_API_KEY", "testnet-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "testnet-secret")
    monkeypatch.setenv("TRADING_LIVE", "1")
    monkeypatch.setenv("EXCHANGE_TESTNET", "1")
    fingerprint = ExchangeConfig(
        exchange="binanceusdm",
        market_type="futures",
        api_key="testnet-key",
        testnet=True,
        max_futures_leverage=int(account["maximum_leverage"]),
        quote_asset="USDT",
        allow_multi_symbol_positions=True,
    ).account_fingerprint
    instrument = Instrument(
        venue="binance",
        market_type=MarketType.FUTURES,
        base_asset="BTC",
        quote_asset="USDT",
        settlement_asset="USDT",
        exchange_symbol="BTCUSDT",
        price_precision=2,
        quantity_precision=6,
        minimum_quantity=0.000001,
        minimum_notional=5.0,
    )
    universe_snapshot_id = SqlUniverseStore(database.engine).record_snapshot(
        universe_id=str(product["universe_id"]),
        observed_at=NOW,
        observations=(
            InstrumentObservation(
                instrument,
                365,
                1_000_000_000,
                1_000_000,
                1.0,
                1_000_000_000,
                0.0001,
                0.2,
                10_000_000,
                1.0,
            ),
        ),
        policy=UniverseEligibilityPolicy(),
    )
    assignment_id = _seed_strategy(
        database,
        product,
        instrument,
        universe_snapshot_id,
        NOW,
        "manual-live-authority",
        execution_mode="paper",
        sleeve_id="directional",
    )
    from src.research.canonical import SqlActiveStrategyAssignmentRepository

    paper_assignment = SqlActiveStrategyAssignmentRepository(database.engine).by_id(assignment_id)
    assert paper_assignment is not None
    forward = SqlForwardEvidenceRepository(database.engine)
    observation_at = "2026-08-29T10:00:01+00:00"
    facts = {
        "schema": "platform.forward_evidence_facts/v1",
        "window_start": NOW,
        "source_event_ids": [str(paper_assignment["artefact_hash"])],
        "metrics": {
            "net_pnl": 1.0,
            "benchmark_pnl": 0.0,
            "drawdown": 0.0,
            "execution_drift": 0.0,
            "model_drift": 0.0,
            "portfolio_capacity": 1.0,
            "risk_budget_available": 1.0,
            "data_gaps": 0,
            "effective_trades": 1,
            "fill_rate": 1.0,
            "slippage": 0.0,
            "data_uptime": 1.0,
            "rejected_orders": 0,
        },
        "forecast_hash": canonical_hash({"source": "authority-test"}),
        "target_hash": None,
    }
    facts["facts_hash"] = canonical_hash(facts)
    observation_id = forward.append(
        strategy_version_id=str(paper_assignment["strategy_version_id"]),
        product_id="active_income",
        instrument_id=instrument.instrument_id,
        observed_at=observation_at,
        artefact_hash=str(paper_assignment["artefact_hash"]),
        observation={"decision_id": "authority-test-decision", "facts": facts},
    )
    summary_id = forward.append_summary(
        strategy_version_id=str(paper_assignment["strategy_version_id"]),
        product_id="active_income",
        observed_at=observation_at,
        artefact_hash=str(paper_assignment["artefact_hash"]),
        evidence={
            "observed_from": observation_at,
            "observed_until": observation_at,
            "elapsed_days": 1.0,
            "independent_decisions": 1,
            "net_pnl": 1.0,
            "benchmark_pnl": 0.0,
            "excess_benchmark_pnl": 1.0,
            "drawdown": 0.0,
            "execution_drift": 0.0,
            "model_drift": 0.0,
            "portfolio_capacity": 1.0,
            "risk_budget_available": 1.0,
            "data_gaps": 0,
            "effective_trades": 1,
            "fill_rate": 1.0,
            "slippage": 0.0,
            "data_uptime": 1.0,
            "rejected_orders": 0,
            "observation_ids": [observation_id],
        },
    )
    forward.append_decision(
        summary_id=summary_id,
        decided_at="2026-08-29T10:00:01+00:00",
        accepted=True,
    )
    SqlActiveStrategyAssignmentRepository(database.engine).assign(
        product_id="active_income",
        portfolio_id="active-income-portfolio",
        strategy_version_id=str(paper_assignment["strategy_version_id"]),
        artefact_hash=str(paper_assignment["artefact_hash"]),
        lifecycle_state="live_ready",
        execution_mode="paper",
        capital_limit=100.0,
        risk_budget=100.0,
        assigned_at=observation_at,
        assigned_by="promotion-test",
        instrument_id=instrument.instrument_id,
        payload={"account_id": "binance-futures-main"},
    )
    authority = PlatformLiveAuthority(
        engine=database.engine,
        configuration=configuration,
        broker_factory=lambda _account, _market: _Broker(fingerprint),
    )
    selection = {
        "product_id": "active_income",
        "artefact_hash": str(paper_assignment["artefact_hash"]),
        "instrument_id": instrument.instrument_id,
        "sleeve_id": "directional",
    }
    return database, authority, selection


def _preflight_and_approve(authority: PlatformLiveAuthority, selection: dict) -> tuple[dict, dict]:
    preflight = authority.preflight(
        **selection,
        capital_cap=0.01,
        checked_at=PREFLIGHT_AT,
    )
    approval = authority.approve(
        **selection,
        expected_preflight_id=preflight["preflight_id"],
        capital_cap=0.01,
        approved_by="henrique",
        approved_at=APPROVED_AT,
        confirm=True,
    )
    return preflight, approval


def test_manual_authority_records_exact_preflight_approval_and_live_assignment(
    tmp_path: Path, monkeypatch
) -> None:
    database, authority, selection = _fixture(tmp_path, monkeypatch)
    try:
        review = authority.inspect(**selection)
        preflight, approval = _preflight_and_approve(authority, selection)
        assignment = authority.assign(
            **selection,
            expected_preflight_id=preflight["preflight_id"],
            expected_approval_id=approval["approval_id"],
            capital_limit=0.01,
            risk_budget=0.01,
            assigned_by="henrique",
            assigned_at=ASSIGNED_AT,
            confirm=True,
        )

        assert review["artefact_hash"] == selection["artefact_hash"]
        assert preflight["environment"] == "testnet"
        assert preflight["account_fingerprint"].startswith("account-v1:")
        assert approval["preflight_id"] == preflight["preflight_id"]
        active = authority.assignments.active(
            "active_income", execution_mode="live", at=ASSIGNED_AT
        )
        assert active is not None
        assert assignment["assignment_id"] == active["id"]
        assert active["artefact_hash"] == selection["artefact_hash"]
        assert active["instrument_id"] == selection["instrument_id"]
        assert active["sleeve_id"] == selection["sleeve_id"]
    finally:
        database.dispose()


def test_live_assignment_replacement_is_ordered_and_single_authority(
    tmp_path: Path, monkeypatch
) -> None:
    database, authority, selection = _fixture(tmp_path, monkeypatch)
    try:
        preflight, approval = _preflight_and_approve(authority, selection)
        authority.assign(
            **selection,
            expected_preflight_id=preflight["preflight_id"],
            expected_approval_id=approval["approval_id"],
            capital_limit=0.01,
            risk_budget=0.01,
            assigned_by="henrique",
            assigned_at=ASSIGNED_AT,
            confirm=True,
        )
        authority.assignments.deactivate("active_income", at=ASSIGNED_AT)
        artefact = authority.artefacts.get(selection["artefact_hash"])
        replacement_id = authority.assignments.assign(
            product_id="active_income",
            portfolio_id=str(artefact["portfolio_id"]),
            sleeve_id=selection["sleeve_id"],
            strategy_version_id=str(artefact["strategy_version_id"]),
            instrument_id=selection["instrument_id"],
            artefact_hash=selection["artefact_hash"],
            lifecycle_state="live",
            execution_mode="live",
            capital_limit=0.01,
            risk_budget=0.01,
            assigned_at=ASSIGNED_AT,
            assigned_by="henrique",
            assignment_reason="promotion transition to live",
            payload={
                "approval_id": approval["approval_id"],
                "preflight_id": preflight["preflight_id"],
            },
        )

        active = authority.assignments.active(
            "active_income", execution_mode="live", at=ASSIGNED_AT
        )
        assert active is not None
        assert active["id"] == replacement_id
        assert active["lifecycle_state"] == "live"
        assert len(authority.assignments.active_assignments("active_income", at=ASSIGNED_AT)) == 1
    finally:
        database.dispose()


def test_manual_authority_requires_confirmation_and_human_actor(
    tmp_path: Path, monkeypatch
) -> None:
    database, authority, selection = _fixture(tmp_path, monkeypatch)
    try:
        preflight = authority.preflight(
            **selection,
            capital_cap=0.01,
            checked_at=PREFLIGHT_AT,
        )
        with pytest.raises(PlatformLiveAuthorityError, match="confirmation"):
            authority.approve(
                **selection,
                expected_preflight_id=preflight["preflight_id"],
                capital_cap=0.01,
                approved_by="henrique",
                approved_at=APPROVED_AT,
                confirm=False,
            )
        with pytest.raises(CanonicalEvidenceError, match="human operator"):
            authority.approve(
                **selection,
                expected_preflight_id=preflight["preflight_id"],
                capital_cap=0.01,
                approved_by="automation",
                approved_at=APPROVED_AT,
                confirm=True,
            )
    finally:
        database.dispose()


def test_assignment_rejects_revoked_or_competing_live_authority(
    tmp_path: Path, monkeypatch
) -> None:
    database, authority, selection = _fixture(tmp_path, monkeypatch)
    try:
        preflight, approval = _preflight_and_approve(authority, selection)
        authority.assign(
            **selection,
            expected_preflight_id=preflight["preflight_id"],
            expected_approval_id=approval["approval_id"],
            capital_limit=0.01,
            risk_budget=0.01,
            assigned_by="henrique",
            assigned_at=ASSIGNED_AT,
            confirm=True,
        )
        with pytest.raises(CanonicalEvidenceError, match="already has an active live assignment"):
            authority.assignments.assign(
                product_id="active_income",
                portfolio_id="active-income-portfolio",
                sleeve_id=selection["sleeve_id"],
                strategy_version_id=str(
                    authority.artefacts.get(selection["artefact_hash"])["strategy_version_id"]
                ),
                instrument_id=selection["instrument_id"],
                artefact_hash=selection["artefact_hash"],
                lifecycle_state="live_canary",
                execution_mode="live",
                capital_limit=0.01,
                risk_budget=0.01,
                assigned_at="2026-08-29T10:00:04+00:00",
                assigned_by="henrique",
                assignment_reason="duplicate live authority",
            )
        with pytest.raises(CanonicalEvidenceError, match="cannot predate"):
            authority.assignments.assign(
                product_id="active_income",
                portfolio_id="active-income-portfolio",
                sleeve_id=selection["sleeve_id"],
                strategy_version_id=str(
                    authority.artefacts.get(selection["artefact_hash"])["strategy_version_id"]
                ),
                instrument_id=selection["instrument_id"],
                artefact_hash=selection["artefact_hash"],
                lifecycle_state="live_canary",
                execution_mode="live",
                capital_limit=0.01,
                risk_budget=0.01,
                assigned_at="2026-08-29T10:00:02.500000+00:00",
                assigned_by="henrique",
                assignment_reason="backdated live authority",
            )

        artefact = authority.artefacts.get(selection["artefact_hash"])
        SqlApprovalRepository(database.engine).append(
            strategy_version_id=str(artefact["strategy_version_id"]),
            product_id="active_income",
            account_id="binance-futures-main",
            artefact_hash=selection["artefact_hash"],
            source_commit_hash=str(artefact["source_commit_hash"]),
            engine_version=str(artefact["engine_version"]),
            capital_cap=0.01,
            actor="henrique",
            approved_at="2026-08-29T10:00:05+00:00",
            status="revoked",
            payload={"reason": "operator revocation"},
        )
        with pytest.raises(PlatformLiveAuthorityError, match="approval is not current"):
            authority.assign(
                **selection,
                expected_preflight_id=preflight["preflight_id"],
                expected_approval_id=approval["approval_id"],
                capital_limit=0.01,
                risk_budget=0.01,
                assigned_by="henrique",
                assigned_at="2026-08-29T10:00:06+00:00",
                confirm=True,
            )
    finally:
        database.dispose()


def test_non_promotable_artefact_is_not_reviewable() -> None:
    with pytest.raises(PlatformLiveAuthorityError, match="non-promotable"):
        PlatformLiveAuthority._assert_reviewable(
            {
                "metadata": {"promotable": False},
                "position_limits": {"maximum_position": 0.1},
            }
        )


def test_generic_records_cannot_create_live_authority(tmp_path: Path, monkeypatch) -> None:
    database, authority, selection = _fixture(tmp_path, monkeypatch)
    try:
        artefact = authority.artefacts.get(selection["artefact_hash"])
        SqlPreflightRepository(database.engine).append(
            {
                "strategy_version_id": artefact["strategy_version_id"],
                "product_id": selection["product_id"],
                "account_id": artefact["account_id"],
                "artefact_hash": selection["artefact_hash"],
                "source_commit_hash": artefact["source_commit_hash"],
                "engine_version": artefact["engine_version"],
                "capital_cap": 0.01,
                "checked_at": PREFLIGHT_AT,
                "accepted": True,
            }
        )
        SqlApprovalRepository(database.engine).append(
            strategy_version_id=artefact["strategy_version_id"],
            product_id=selection["product_id"],
            account_id=artefact["account_id"],
            artefact_hash=selection["artefact_hash"],
            source_commit_hash=artefact["source_commit_hash"],
            engine_version=artefact["engine_version"],
            capital_cap=0.01,
            actor="release-pipeline",
            approved_at=APPROVED_AT,
        )
        with pytest.raises(CanonicalEvidenceError, match="matching approval"):
            authority.assignments.assign(
                product_id=selection["product_id"],
                portfolio_id=artefact["portfolio_id"],
                sleeve_id=selection["sleeve_id"],
                strategy_version_id=artefact["strategy_version_id"],
                instrument_id=selection["instrument_id"],
                artefact_hash=selection["artefact_hash"],
                lifecycle_state="live_canary",
                execution_mode="live",
                capital_limit=0.01,
                risk_budget=0.01,
                assigned_at=ASSIGNED_AT,
                assigned_by="release-pipeline",
            )
    finally:
        database.dispose()
