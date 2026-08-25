from __future__ import annotations

import pytest

from src.domain._codec import canonical_hash
from src.services.job_schemas import (
    JobSchemaError,
    ResearchJobRequest,
    RiskAssessmentRequest,
    build_content_hash,
)


def _research_request() -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_id": "sha256:" + "1" * 64,
        "dataset_snapshot_ids": ["sha256:" + "2" * 64],
        "feature_manifest_id": "sha256:" + "3" * 64,
        "cost_model_id": "sha256:" + "4" * 64,
        "parameter_set_id": "sha256:" + "5" * 64,
        "evaluator_version": "evaluator/v2",
        "requested_stage": "development",
        "evaluated_at": "2026-08-23T00:00:00+00:00",
        "producer_identity": "agent:test",
    }
    payload["content_hash"] = build_content_hash(payload)
    return payload


def test_research_request_has_only_immutable_input_identities() -> None:
    request = ResearchJobRequest.from_mapping(_research_request())
    assert request.dataset_snapshot_ids == ("sha256:" + "2" * 64,)
    with pytest.raises(JobSchemaError, match="unknown fields"):
        ResearchJobRequest.from_mapping({**_research_request(), "metrics": {"return": 1}})


def test_production_research_requests_require_explicit_dataset_roles() -> None:
    with pytest.raises(JobSchemaError, match="explicit dataset roles"):
        ResearchJobRequest.from_mapping(_research_request(), require_dataset_roles=True)


def test_risk_request_rejects_values_and_result_fields() -> None:
    payload = {
        "assessment_id": "assessment-1",
        "product_id": "active_income",
        "event_id": "event-1",
        "target_position_snapshot_id": "sha256:" + "1" * 64,
        "account_snapshot_id": "sha256:" + "2" * 64,
        "positions_snapshot_id": "sha256:" + "3" * 64,
        "balances_snapshot_id": "sha256:" + "4" * 64,
        "market_data_snapshot_id": "sha256:" + "5" * 64,
        "risk_policy_ids": ["risk-policy"],
        "evaluated_at": "2026-08-23T00:00:00+00:00",
        "producer_identity": "portfolio-engine",
    }
    payload["content_hash"] = canonical_hash(payload)
    request = RiskAssessmentRequest.from_mapping(payload)
    assert request.product_id == "active_income"
    with pytest.raises(JobSchemaError):
        RiskAssessmentRequest.from_mapping({**payload, "accepted": True})
