import datetime as dt
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import insert, select

from src.data.database import (
    PlatformDatabase,
    dataset_snapshot,
    holdout_claim,
    holdout_outcome,
    selection_cohort,
    selection_cohort_open,
    strategy_definition,
    strategy_version,
)
from src.domain._codec import canonical_hash
from src.research.accounting import FUTURES_ACCOUNTING_RISK_CONTRACT
from src.research.canonical import CanonicalEvidenceError, SqlHoldoutRepository
from src.research.cohorts import prepare_cohort
from src.research.economic_series import ECONOMIC_ENGINE_VERSION
from src.research.evaluation import (
    CanonicalResearchEvaluator,
    EvaluationRequest,
    ProtectedHoldoutWorker,
)
from src.research.executors import DataUnavailableError
from src.research.selection import FrozenCohort, selection_adjusted_block_bootstrap
from src.research.store import SqlResearchStore


def _approval_calibration_fixture(configuration):
    from src.research.calibration import (
        CASES,
        GENERATOR_VERSION,
        _case_summary,
        calibration_runtime,
    )
    from src.research.policy_approval import calibration_source_hashes
    from src.research.selection import METHOD

    trials = []
    candidates = configuration["search_budgets"]["maximum_candidates_per_epoch"]
    members = [f"configuration-{index}" for index in range(candidates)]
    times = [
        (dt.datetime(2026, 1, 1, tzinfo=dt.UTC) + dt.timedelta(hours=index)).isoformat()
        for index in range(128)
    ]
    for index, case in enumerate(CASES):
        for repeat in range(40):
            seed = 11 + index * 10000 + repeat
            trials.append(
                {
                    "case": case,
                    "seed": seed,
                    "causal": case != "leakage_trap",
                    "qualified_in_simulation": case == "weak_edge",
                    "inference": {
                        "method": METHOD,
                        "seed": seed,
                        "draws": 2000,
                        "members": members,
                        "timestamps": times,
                        "sample_unit": "calendar_interval",
                        "qualification_authority": False,
                        "status": "inconclusive" if case == "flat" else "measured",
                        "adjusted_pvalues": {}
                        if case == "flat"
                        else {
                            member: 1 / 2001 if case in {"weak_edge", "leakage_trap"} else 1.0
                            for member in members
                        },
                    },
                }
            )
    return {
        "schema": "research.selection_calibration/v1",
        "method": METHOD,
        "generator_version": GENERATOR_VERSION,
        "policy_hash": canonical_hash(configuration),
        "source_hashes": calibration_source_hashes(),
        "runtime": calibration_runtime(),
        "qualification_authority": False,
        "policy_approval_required": True,
        "candidates": candidates,
        "calibration_scope": "selection_inference_kernel",
        "nominal_test_alpha": 0.05,
        "proposed_false_qualification_budget": 0.1,
        "replicates": 40,
        "observations": 128,
        "seed": 11,
        "trials": trials,
        "summary": {
            case: _case_summary([trial for trial in trials if trial["case"] == case])
            for case in CASES
        },
    }


def test_research_policy_approval_binds_reviewed_receipt_configuration_code_and_revocation(
    tmp_path, monkeypatch
):
    import hashlib
    import json
    from pathlib import Path

    from src.research.evaluation import EvidencePolicy
    from src.research.policy_approval import (
        approve_policy,
        current_approval,
        revoke_policy,
        validate_calibration,
    )

    configuration = json.loads(Path("config/research.json").read_text())
    configuration["evidence_policy"]["version"] = "fixture-" + uuid.uuid4().hex
    receipt = _approval_calibration_fixture(configuration)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(receipt))
    reviewed = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'approval.sqlite3'}")
    database.create_schema()
    arguments = {
        "receipt_path": path,
        "expected_hash": reviewed,
        "configuration": configuration,
        "approved_by": "test operator",
        "approved_at": "2026-09-08T01:00:00+00:00",
    }
    with pytest.raises(ValueError, match="reviewed hash"):
        approve_policy(database.engine, **{**arguments, "expected_hash": "sha256:" + "0" * 64})
    identity = approve_policy(database.engine, **arguments)
    assert approve_policy(database.engine, **arguments) == identity
    query = {
        "policy_hash": EvidencePolicy.from_configuration(
            configuration["evidence_policy"]
        ).policy_hash,
        "configuration_hash": canonical_hash(configuration),
        "at": "2026-09-08T02:00:00+00:00",
    }
    from src.research.cohorts import _open_cohort

    cohort_id = canonical_hash({"test_cohort": uuid.uuid4().hex})
    cohort = {
        "cohort_id": cohort_id,
        "policy_hash": query["policy_hash"],
        "engine_version": ECONOMIC_ENGINE_VERSION,
        "members": ["candidate"],
        "frozen_at": "2026-09-08T00:00:00+00:00",
    }
    with database.engine.begin() as connection:
        connection.execute(
            insert(selection_cohort).values(
                id=cohort_id, scope_id=cohort_id, created_at=cohort["frozen_at"], payload=cohort
            )
        )
        connection.execute(
            insert(selection_cohort_open).values(
                id=cohort_id,
                created_at="2026-09-08T00:30:00+00:00",
                payload={"research_policy_approval_id": None},
            )
        )
        _open_cohort(
            connection,
            cohort,
            SimpleNamespace(candidate_id="candidate"),
            query["at"],
            query["policy_hash"],
            query["configuration_hash"],
        )
        opening = connection.execute(
            select(selection_cohort_open.c.payload).where(selection_cohort_open.c.id == cohort_id)
        ).scalar_one()
        assert opening["research_policy_approval_id"] is None
    with database.engine.connect() as connection:
        assert current_approval(connection, **query)["approval_id"] == identity
        assert current_approval(connection, **{**query, "at": "2026-09-08T00:00:00+00:00"}) is None
        assert (
            current_approval(connection, **{**query, "configuration_hash": canonical_hash({})})
            is None
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                "src.research.policy_approval.research_policy_engine_digest",
                lambda: "sha256:" + "0" * 64,
            )
            assert current_approval(connection, **query) is None
    revoke_policy(
        database.engine,
        approval_id=identity,
        revoked_by="test operator",
        revoked_at="2026-09-08T01:30:00+00:00",
    )
    with database.engine.connect() as connection:
        assert current_approval(connection, **query) is None
    with pytest.raises(ValueError, match="binding"):
        validate_calibration(
            {**receipt, "runtime": {"python": "other", "numpy": "other"}}, configuration
        )
    receipt["summary"]["null"]["qualification_rate"] = 0.5
    with pytest.raises(ValueError, match="summary"):
        validate_calibration(receipt, configuration)
    database.dispose()


def test_return_block_stress_preserves_loss_runs_terminal_wealth_and_seed():
    import math

    from src.research.executors import _maximum_drawdown
    from src.research.stress import return_block_permutations

    returns = [-0.1, -0.1, 0.2, 0.2]
    paths = list(return_block_permutations(returns, block_length=2, seed=31, iterations=40))
    assert paths == list(return_block_permutations(returns, block_length=2, seed=31, iterations=40))
    assert len({tuple(path) for path in paths}) == 2
    for path in paths:
        assert _maximum_drawdown(path) == pytest.approx(0.19)
        assert math.prod(1 + value for value in path) == pytest.approx(1.1664)
    for path in return_block_permutations([*returns, -0.03], block_length=2, seed=31):
        assert sorted(path) == sorted([*returns, -0.03])
    with pytest.raises(ValueError, match="multi-observation"):
        list(return_block_permutations(returns, block_length=1, seed=31))


def test_evidence_cannot_exempt_core_checks_or_unrelated_negative_controls():
    from src.research.evaluation import EvidencePolicy, EvidenceStatus
    from src.research.evidence import EvidenceApplicability

    policy = EvidencePolicy()
    exemption = {"status": "not_applicable", "passed": True}
    for name in ("data_integrity", "semantic_parity", "realistic_costs", "cost_stress"):
        statuses = policy.statuses(
            "robustness", {name: exemption}, (), applicability=EvidenceApplicability(1, 0, 0)
        )
        assert statuses[name] is EvidenceStatus.UNAVAILABLE
    evidence = {
        "negative_control_results": {
            "placebo_event_times": exemption,
            "cross_instrument": exemption,
        }
    }
    statuses = policy.statuses(
        "robustness",
        evidence,
        ("placebo_event_times",),
        applicability=EvidenceApplicability(1, 0, 0),
    )
    assert statuses["negative_control_results"] is EvidenceStatus.UNAVAILABLE
    for count, expected in ((1, EvidenceStatus.NOT_APPLICABLE), (2, EvidenceStatus.UNAVAILABLE)):
        statuses = policy.statuses(
            "robustness",
            evidence,
            ("cross_instrument",),
            applicability=EvidenceApplicability(instrument_count=count),
        )
        assert statuses["negative_control_results"] is expected


@pytest.mark.skipif(
    os.environ.get("TRADING_PLATFORM_PRIVILEGE_TEST") != "1",
    reason="isolated PostgreSQL fixture required",
)
def test_overlapping_frozen_cohorts_cannot_open_concurrently():
    database = PlatformDatabase(os.environ["TRADING_PLATFORM_DATABASE_URL"])
    run_id = uuid.uuid4().hex
    now = "2026-09-07T00:00:00+00:00"
    candidates = []
    cohort_ids = []
    with database.engine.begin() as connection:
        for index in range(2):
            snapshot = canonical_hash({"run": run_id, "snapshot": index})
            identity = canonical_hash({"run": run_id, "candidate": index})
            cohort_id = canonical_hash({"run": run_id, "cohort": index})
            cohort_ids.append(cohort_id)
            scope = canonical_hash({"product_id": "active_income", "protected_snapshot": snapshot})
            payload = {
                "cohort_id": cohort_id,
                "scope_id": scope,
                "protected_snapshot_id": snapshot,
                "product_id": "active_income",
                "members": [identity],
                "instruments": [f"concurrency-fixture-{run_id}"],
                "frozen_at": now,
                "engine_version": ECONOMIC_ENGINE_VERSION,
                "policy_hash": "fixture-policy",
                "interval": {
                    "start": "2025-01-01T00:00:00+00:00",
                    "end": "2025-02-01T00:00:00+00:00",
                },
            }
            connection.execute(
                insert(selection_cohort).values(
                    id=cohort_id, scope_id=scope, created_at=now, payload=payload
                )
            )
            candidates.append(
                SimpleNamespace(
                    candidate_id=identity,
                    definition=SimpleNamespace(product="active_income"),
                    dataset_plan=SimpleNamespace(protected_holdout_snapshot_id=snapshot),
                )
            )

    def open_candidate(candidate):
        try:
            prepare_cohort(
                database.engine,
                candidate,
                now="2026-09-07T00:00:01+00:00",
                policy_hash="fixture-policy",
            )
            return "opened"
        except DataUnavailableError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(open_candidate, candidates)) == ["blocked", "opened"]
    with database.engine.connect() as connection:
        assert (
            len(
                connection.execute(
                    select(selection_cohort_open).where(selection_cohort_open.c.id.in_(cohort_ids))
                ).all()
            )
            == 1
        )
    database.dispose()


def test_holdout_retry_reuses_claim_after_restart_without_reopening_changed_inputs(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'holdout.sqlite3'}"
    database = PlatformDatabase(url)
    database.create_schema()
    now = "2026-09-07T00:00:00+00:00"
    snapshot = canonical_hash({"fixture": "protected"})
    source = canonical_hash({"fixture": "code"})
    with database.engine.begin() as connection:
        connection.execute(
            insert(strategy_definition).values(
                id=source,
                identity="fixture",
                product_id="active_income",
                source_type="registered_python",
                source_hash=source,
                definition={},
            )
        )
        connection.execute(
            insert(strategy_version).values(
                id="fixture-v1",
                definition_id=source,
                version="1",
                created_at=now,
                payload={},
            )
        )
        connection.execute(insert(dataset_snapshot).values(id=snapshot, created_at=now, payload={}))
    request = {
        "strategy_version_id": "fixture-v1",
        "data_snapshot_id": snapshot,
        "cohort_id": "frozen-cohort",
        "source_hashes": {"code": source},
    }
    first = SqlHoldoutRepository(database.engine).claim(**request, claimed_at=now)
    database.dispose()
    database = PlatformDatabase(url)
    repository = SqlHoldoutRepository(database.engine)
    assert repository.claim(**request, claimed_at="2026-09-08T00:00:00+00:00") == first
    with pytest.raises(CanonicalEvidenceError, match="only one protected"):
        repository.claim(
            **{**request, "source_hashes": {"code": canonical_hash("changed")}}, claimed_at=now
        )
    with database.engine.connect() as connection:
        rows = connection.execute(select(holdout_claim)).mappings().all()
    assert len(rows) == 1 and rows[0]["payload"]["claimed_at"] == now
    database.dispose()


def test_active_income_context_binds_risk_contract_while_btc_context_does_not(tmp_path):
    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'risk-context.sqlite3'}")
    database.create_schema()
    snapshot = canonical_hash({"fixture": "context"})
    request = EvaluationRequest(
        candidate_id=canonical_hash({"candidate": "fixture"}),
        evaluation_policy_id="policy-v1",
        dataset_snapshot_ids=(snapshot,),
        requested_stage="development",
        evaluated_at="2026-09-07T00:00:00+00:00",
        dataset_roles={snapshot: "development"},
    )
    evaluator = CanonicalResearchEvaluator(SqlResearchStore(database.engine))
    shared = {
        "request": request,
        "family": "trend",
        "horizon": "1h",
        "evidence_type": "swing",
        "stage_snapshot_ids": (snapshot,),
        "thesis_id": canonical_hash({"thesis": "missing"}),
    }
    active_income = evaluator._evaluation_context(
        definition=SimpleNamespace(
            strategy_version_id="active-v1",
            product="active_income",
            source_hash=canonical_hash({"source": "active"}),
        ),
        **shared,
    )
    btc_accumulation = evaluator._evaluation_context(
        definition=SimpleNamespace(
            strategy_version_id="btc-v1",
            product="btc_accumulation",
            source_hash=canonical_hash({"source": "btc"}),
        ),
        **shared,
    )

    assert active_income["financial_risk_contract"] == FUTURES_ACCOUNTING_RISK_CONTRACT
    assert "financial_risk_contract" not in btc_accumulation
    database.dispose()


def test_protected_holdout_reuse_requires_risk_contract_in_outcome_and_receipt(tmp_path):
    contract = FUTURES_ACCOUNTING_RISK_CONTRACT
    variants = (
        {"passed": True, "execution_receipt": {}},
        {"passed": True, "financial_risk_contract": contract, "execution_receipt": {}},
        {
            "passed": True,
            "execution_receipt": {"financial_risk_contract": contract},
        },
        {
            "passed": True,
            "financial_risk_contract": contract,
            "execution_receipt": {"financial_risk_contract": contract},
        },
    )
    now = "2026-09-07T00:00:00+00:00"
    for index, sealed in enumerate(variants):
        database = PlatformDatabase(
            f"sqlite+pysqlite:///{tmp_path / f'protected-risk-{index}.sqlite3'}"
        )
        database.create_schema()
        snapshot = canonical_hash({"fixture": f"protected-{index}"})
        source = canonical_hash({"fixture": f"source-{index}"})
        with database.engine.begin() as connection:
            connection.execute(
                insert(strategy_definition).values(
                    id=source,
                    identity=f"fixture-{index}",
                    product_id="active_income",
                    source_type="registered_python",
                    source_hash=source,
                    definition={},
                )
            )
            connection.execute(
                insert(strategy_version).values(
                    id=f"fixture-v{index}",
                    definition_id=source,
                    version="1",
                    created_at=now,
                    payload={},
                )
            )
            connection.execute(
                insert(dataset_snapshot).values(id=snapshot, created_at=now, payload={})
            )
        request = {
            "strategy_version_id": f"fixture-v{index}",
            "data_snapshot_id": snapshot,
            "cohort_id": "frozen-cohort",
            "source_hashes": {"code": source},
        }
        repository = SqlHoldoutRepository(database.engine)
        claim_id = repository.claim(**request, claimed_at=now)
        repository.record_outcome(
            claim_id=claim_id,
            evaluated_at=now,
            accepted=True,
            outcome=sealed,
        )
        worker = ProtectedHoldoutWorker(
            database.engine,
            lambda _claim: pytest.fail("an immutable protected outcome must not be rerun"),
            product_id="active_income",
        )
        if index < 3:
            with pytest.raises(DataUnavailableError, match="does not bind"):
                worker.claim_and_evaluate(
                    strategy_version_id=request["strategy_version_id"],
                    dataset_snapshot_id=snapshot,
                    cohort_id=request["cohort_id"],
                    source_hashes=request["source_hashes"],
                    evaluated_at=now,
                )
        else:
            result = worker.claim_and_evaluate(
                strategy_version_id=request["strategy_version_id"],
                dataset_snapshot_id=snapshot,
                cohort_id=request["cohort_id"],
                source_hashes=request["source_hashes"],
                evaluated_at=now,
            )
            assert result[2] is True
        database.dispose()

    database = PlatformDatabase(f"sqlite+pysqlite:///{tmp_path / 'protected-risk-new.sqlite3'}")
    database.create_schema()
    snapshot = canonical_hash({"fixture": "protected-new"})
    source = canonical_hash({"fixture": "source-new"})
    with database.engine.begin() as connection:
        connection.execute(
            insert(strategy_definition).values(
                id=source,
                identity="fixture-new",
                product_id="active_income",
                source_type="registered_python",
                source_hash=source,
                definition={},
            )
        )
        connection.execute(
            insert(strategy_version).values(
                id="fixture-new-v1",
                definition_id=source,
                version="1",
                created_at=now,
                payload={},
            )
        )
        connection.execute(insert(dataset_snapshot).values(id=snapshot, created_at=now, payload={}))
    worker = ProtectedHoldoutWorker(
        database.engine,
        lambda _claim: (
            True,
            {"sealed_result": {"passed": True, "financial_risk_contract": contract}},
        ),
        product_id="active_income",
    )
    with pytest.raises(DataUnavailableError, match="does not bind"):
        worker.claim_and_evaluate(
            strategy_version_id="fixture-new-v1",
            dataset_snapshot_id=snapshot,
            cohort_id="frozen-cohort",
            source_hashes={"code": source},
            evaluated_at=now,
        )
    with database.engine.connect() as connection:
        assert connection.execute(select(holdout_outcome)).first() is None
    database.dispose()


def _inference(values, *, development=None):
    count, candidates, instruments = values.shape
    development = (
        np.random.default_rng(4).normal(0, 0.01, values.shape)
        if development is None
        else development
    )
    cohort = FrozenCohort(
        tuple(f"candidate-{i}" for i in range(candidates)),
        tuple(f"instrument-{i}" for i in range(instruments)),
        "2026-09-07T00:00:00+00:00",
        canonical_hash(development.tolist()),
    )
    start = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    return selection_adjusted_block_bootstrap(
        cohort,
        timestamps=[(start + dt.timedelta(hours=i)).isoformat() for i in range(count)],
        net_returns=values,
        benchmark_returns=np.zeros(count),
        development_returns=development,
        opened_at="2026-09-07T00:01:00+00:00",
        seed=42,
    )


def test_bounded_noise_reference_has_exact_minimum_and_maximum_pvalues():
    noise = np.random.default_rng(3).uniform(-0.01, 0.01, (256, 2, 1))
    noise[:, 0, :] += 0.05
    noise[:, 1, :] -= 0.05
    result = _inference(noise)
    assert result["status"] == "measured"
    assert result["adjusted_pvalues"]["candidate-0"] == pytest.approx(1 / 2001)
    assert result["adjusted_pvalues"]["candidate-1"] == 1
    assert result["qualification_authority"] is False


def test_identical_candidates_keep_joint_dependence_instead_of_becoming_independent_trials():
    values = np.random.default_rng(1).normal(0, 0.01, (256, 1, 1))
    development = np.random.default_rng(2).normal(0, 0.01, values.shape)
    single = _inference(values, development=development)
    duplicated = _inference(
        np.repeat(values, 2, axis=1), development=np.repeat(development, 2, axis=1)
    )
    assert (
        duplicated["adjusted_pvalues"]["candidate-0"] == single["adjusted_pvalues"]["candidate-0"]
    )
    assert (
        duplicated["adjusted_pvalues"]["candidate-1"] == single["adjusted_pvalues"]["candidate-0"]
    )


def test_portfolio_aggregation_before_joint_resampling_is_identical():
    values = np.random.default_rng(17).normal(0, 0.01, (256, 2, 3))
    development = np.random.default_rng(18).normal(0, 0.01, values.shape)
    cohort = FrozenCohort(
        ("a", "b"),
        ("BTC", "ETH", "SOL"),
        "2026-09-07T00:00:00+00:00",
        canonical_hash(development.tolist()),
    )
    arguments = {
        "timestamps": [
            (dt.datetime(2026, 1, 1, tzinfo=dt.UTC) + dt.timedelta(hours=index)).isoformat()
            for index in range(256)
        ],
        "benchmark_returns": np.zeros(256),
        "development_returns": development,
        "opened_at": "2026-09-07T00:00:01+00:00",
        "seed": 19,
    }
    instrument_result = selection_adjusted_block_bootstrap(cohort, net_returns=values, **arguments)
    portfolio_result = selection_adjusted_block_bootstrap(
        cohort, net_returns=values.sum(axis=2), **arguments
    )
    assert instrument_result["adjusted_pvalues"] == portfolio_result["adjusted_pvalues"]


def test_flat_and_short_series_are_inconclusive():
    assert _inference(np.zeros((256, 2, 1)))["status"] == "inconclusive"
    values = np.random.default_rng(2).normal(0, 0.01, (20, 2, 1))
    development = np.random.default_rng(3).normal(0, 0.01, (256, 2, 1))
    assert _inference(values, development=development)["status"] == "inconclusive"


def test_calibration_cutoff_is_separate_from_false_qualification_budget():
    from src.research.calibration import calibrate

    arguments = {"seed": 20260908, "replicates": 1, "policy_hash": "test", "maximum_seconds": 30}
    strict = calibrate(**arguments, test_alpha=0.01)
    permissive = calibrate(**arguments, test_alpha=0.99)
    assert strict["proposed_false_qualification_budget"] == 0.1
    assert permissive["proposed_false_qualification_budget"] == 0.1
    assert strict["summary"]["null"]["qualification_rate"] == 0
    assert permissive["summary"]["null"]["qualification_rate"] == 1
    assert not strict["qualification_authority"]
    assert strict["policy_approval_required"]


def test_calendar_or_manifest_mismatch_cannot_produce_a_statistic():
    development = np.ones((256, 1, 1))
    cohort = FrozenCohort(
        ("candidate",), ("BTC",), "2026-09-07T00:00:00+00:00", canonical_hash(development.tolist())
    )
    with pytest.raises(ValueError, match="calendar index"):
        selection_adjusted_block_bootstrap(
            cohort,
            timestamps=["2026-01-01T00:00:00+00:00"],
            net_returns=development,
            benchmark_returns=[0],
            development_returns=development,
            opened_at="2026-09-07T00:01:00+00:00",
            seed=1,
        )


def test_joint_bootstrap_t_matches_bartlett_quadratic_form_reference():
    from src.research.selection import _bootstrap_length, _hac_variance

    count, block, draws, seed = 64, 4, 2000, 51
    values = np.random.default_rng(512).normal(0.0003, 0.01, (count, 2))
    centred = values - values.mean(axis=0)
    distance = np.abs(np.arange(count)[:, None] - np.arange(count)[None, :])
    weights = np.maximum(0, 1 - distance / block)
    observed_variance = np.diag(centred.T @ weights @ centred) / count
    np.testing.assert_allclose(_hac_variance(values, block), observed_variance)
    alternating = np.tile([1.0, -1.0], count // 2)[:, None]
    assert _hac_variance(alternating, block)[0] >= 0
    observed = values.mean(axis=0) / np.sqrt(observed_variance / count)
    starts = np.random.default_rng(seed).integers(0, count, size=(draws, count // block, 1))
    indices = ((starts + np.arange(block)) % count).reshape(draws, count)
    exceedances = np.zeros(2, dtype=int)
    for panel in centred[indices]:
        residuals = panel - panel.mean(axis=0)
        variance = np.diag(residuals.T @ weights @ residuals) / count
        maximum = np.max(panel.mean(axis=0) / np.sqrt(variance / count))
        exceedances += maximum >= observed
    result = _bootstrap_length(values, block, seed, draws)
    assert result is not None
    np.testing.assert_array_equal(result["pvalues"], (exceedances + 1) / (draws + 1))


def test_calibration_resumes_identical_trials_and_rejects_changed_checkpoints(tmp_path):
    from unittest.mock import patch

    from src.research.calibration import _calibration_trial, calibrate

    arguments = {
        "seed": 731,
        "replicates": 1,
        "policy_hash": "checkpoint-contract",
        "maximum_seconds": 60,
        "candidates": 2,
    }
    expected = calibrate(**arguments)
    directory = tmp_path / "calibration"
    with patch(
        "src.research.calibration._calibration_trial",
        side_effect=[expected["trials"][0], TimeoutError("interrupted")],
    ):
        with pytest.raises(TimeoutError, match="interrupted"):
            calibrate(**arguments, checkpoint_dir=directory)
    assert len(list(directory.glob("null-*.json"))) == 1
    with patch("src.research.calibration._calibration_trial", wraps=_calibration_trial) as trial:
        resumed = calibrate(**arguments, checkpoint_dir=directory)
    assert trial.call_count == 5
    assert resumed["summary"] == expected["summary"]
    assert canonical_hash(resumed["trials"]) == canonical_hash(expected["trials"])
    with patch("src.research.calibration._calibration_trial") as trial:
        repeated = calibrate(**arguments, checkpoint_dir=directory)
    trial.assert_not_called()
    assert canonical_hash(repeated["trials"]) == canonical_hash(expected["trials"])
    with pytest.raises(ValueError, match="different design or runtime"):
        calibrate(**{**arguments, "seed": 732}, checkpoint_dir=directory)
    path = directory / "null-731.json"
    checkpoint = json.loads(path.read_text())
    checkpoint["trial"]["qualified_in_simulation"] = not checkpoint["trial"][
        "qualified_in_simulation"
    ]
    path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="content or trial binding"):
        calibrate(**arguments, checkpoint_dir=directory)
