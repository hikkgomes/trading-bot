"""Transactional paper eligibility derived from canonical research records."""

from collections.abc import Mapping
from typing import Any

from sqlalchemy import exists, insert, select

from src.data.database import (
    evidence_invalidation,
    experiment,
    experiment_run,
    holdout_claim,
    holdout_outcome,
    paper_eligibility_event,
    validation_stage,
)
from src.research.accounting import FUTURES_ACCOUNTING_RISK_CONTRACT


def _validate_financial_contract(product_id: str, contract: Any, receipt_contract: Any) -> None:
    if product_id == "active_income" and (
        contract != FUTURES_ACCOUNTING_RISK_CONTRACT
        or receipt_contract != FUTURES_ACCOUNTING_RISK_CONTRACT
    ):
        raise ValueError(
            "paper eligibility requires current futures financial feasibility evidence"
        )


def publish_paper_eligibility(connection, artefact: Mapping[str, Any], *, created_at: str) -> None:
    metadata = artefact.get("metadata") or {}
    if metadata.get("promotable") is not True or metadata.get("diagnostic") is True:
        return
    identity = artefact["artefact_hash"]
    if connection.execute(
        select(paper_eligibility_event.c.id).where(paper_eligibility_event.c.id == identity)
    ).first():
        return
    authority = artefact["authoritative_evidence"]
    _validate_stages(connection, artefact, authority)
    _validate_holdout(connection, artefact, authority)
    connection.execute(
        insert(paper_eligibility_event).values(
            id=identity,
            created_at=created_at,
            candidate_id=metadata["candidate_id"],
            engine_version=artefact["engine_version"],
            payload={
                "schema": "research.paper_eligibility/v1",
                "artefact_hash": identity,
                "strategy_version_id": artefact["strategy_version_id"],
                "product_id": artefact["product_id"],
                "sealed_at": created_at,
                "state": "paper_eligible",
                "research_qualified": False,
                "live_authority": False,
            },
        )
    )


def _validate_stages(connection, artefact: Mapping[str, Any], authority: Mapping[str, Any]) -> None:
    ids = authority["validation_stage_ids"]
    rows = (
        connection.execute(
            select(
                validation_stage.c.id,
                validation_stage.c.stage,
                validation_stage.c.accepted,
                validation_stage.c.evaluated_at,
                validation_stage.c.engine_version,
                validation_stage.c.payload["evidence"]["evaluation_context"][
                    "financial_risk_contract"
                ]
                .as_string()
                .label("financial_risk_contract"),
                experiment_run.c.payload["receipt"]["financial_risk_contract"]
                .as_string()
                .label("receipt_contract"),
            )
            .join(
                experiment,
                validation_stage.c.experiment_id == experiment.c.id,
            )
            .join(experiment_run, validation_stage.c.source_run_id == experiment_run.c.id)
            .where(
                validation_stage.c.id.in_(ids),
                experiment.c.strategy_version_id == artefact["strategy_version_id"],
                ~exists(
                    select(evidence_invalidation.c.id).where(
                        evidence_invalidation.c.id == "stage:" + validation_stage.c.id
                    )
                ),
            )
        )
        .mappings()
        .all()
    )
    if len(rows) != len(ids) or {row["stage"] for row in rows} != {
        "screening",
        "development",
        "robustness",
        "protected",
    }:
        raise ValueError("paper eligibility needs every prior canonical research stage")
    if not all(row["accepted"] and row["evaluated_at"] <= artefact["created_at"] for row in rows):
        raise ValueError(
            "paper eligibility research evidence is rejected or newer than the artefact"
        )

    if any(row["engine_version"] != artefact["engine_version"] for row in rows):
        raise ValueError("paper eligibility research engine differs from the artefact")
    for row in rows:
        _validate_financial_contract(
            artefact["product_id"], row["financial_risk_contract"], row["receipt_contract"]
        )


def _validate_holdout(
    connection, artefact: Mapping[str, Any], authority: Mapping[str, Any]
) -> None:
    claim = connection.execute(
        select(holdout_claim.c.payload).where(
            holdout_claim.c.id == authority["holdout_claim_id"],
        )
    ).scalar_one()
    if claim["strategy_version_id"] != artefact["strategy_version_id"]:
        raise ValueError("paper eligibility holdout belongs to another strategy")
    if claim["data_snapshot_id"] not in artefact["dataset_snapshot_hashes"]:
        raise ValueError("paper eligibility holdout dataset differs from the artefact")
    outcomes = (
        connection.execute(
            select(
                holdout_outcome.c.evaluated_at,
                holdout_outcome.c.payload["financial_risk_contract"]
                .as_string()
                .label("financial_risk_contract"),
                holdout_outcome.c.payload["execution_receipt"]["financial_risk_contract"]
                .as_string()
                .label("receipt_contract"),
            ).where(
                holdout_outcome.c.holdout_claim_id == authority["holdout_claim_id"],
                holdout_outcome.c.accepted.is_(True),
            )
        )
        .mappings()
        .all()
    )
    if len(outcomes) != 1 or outcomes[0]["evaluated_at"] > artefact["created_at"]:
        raise ValueError("paper eligibility requires an accepted sealed holdout")
    _validate_financial_contract(
        artefact["product_id"],
        outcomes[0]["financial_risk_contract"],
        outcomes[0]["receipt_contract"],
    )
