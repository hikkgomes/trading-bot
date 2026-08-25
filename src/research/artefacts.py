"""Immutable, reproducible deployable strategy artefacts."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.domain._codec import canonical_hash, json_value, timestamp, to_primitive
from src.domain.strategies import StrategyDefinition


def _hashes(values: tuple[str, ...], *, field_name: str, allow_empty: bool = False) -> None:
    if not allow_empty and not values:
        raise ValueError(f"{field_name} cannot be empty")
    if any(
        not item.startswith("sha256:")
        or len(item) != 71
        or any(character not in "0123456789abcdef" for character in item[7:])
        for item in values
    ):
        raise ValueError(f"{field_name} must contain SHA-256 hashes")


@dataclass(frozen=True)
class StrategyArtefact:
    definition: StrategyDefinition
    dependency_hash: str
    dataset_snapshot_hashes: tuple[str, ...]
    feature_set_version: str
    cost_model_version: str
    validation_evidence: Mapping[str, Any]
    holdout_claim: Mapping[str, Any]
    forward_evidence: Mapping[str, Any]
    promotion_policy: Mapping[str, Any]
    position_limits: Mapping[str, Any]
    risk_limits: Mapping[str, Any]
    model_hashes: tuple[str, ...]
    supported_products: tuple[str, ...]
    supported_instruments: tuple[str, ...]
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    authoritative_evidence: Mapping[str, Any] = field(default_factory=dict)
    source_commit_hash: str | None = None
    dependency_lock_hash: str | None = None
    feature_set_hash: str | None = None
    cost_model_hash: str | None = None
    strategy_version_id: str | None = None
    product_id: str | None = None
    portfolio_id: str | None = None
    account_id: str | None = None
    promotion_policy_id: str | None = None
    engine_version: str | None = None

    def __post_init__(self) -> None:
        _hashes((self.dependency_hash,), field_name="dependency_hash")
        _hashes(self.dataset_snapshot_hashes, field_name="dataset_snapshot_hashes")
        _hashes(self.model_hashes, field_name="model_hashes", allow_empty=True)
        if not self.supported_products or not self.supported_instruments:
            raise ValueError("artefacts need supported products and instruments")
        object.__setattr__(self, "created_at", timestamp(self.created_at, field="created_at"))
        for field_name in (
            "validation_evidence",
            "holdout_claim",
            "forward_evidence",
            "promotion_policy",
            "position_limits",
            "risk_limits",
            "metadata",
            "authoritative_evidence",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{field_name} must be an object")
            object.__setattr__(self, field_name, json_value(dict(value), field=field_name))
        for field_name, fallback in (
            ("source_commit_hash", self.definition.source_hash),
            ("dependency_lock_hash", self.dependency_hash),
            ("feature_set_hash", canonical_hash({"feature_set_version": self.feature_set_version})),
            ("cost_model_hash", canonical_hash({"cost_model_version": self.cost_model_version})),
        ):
            value = getattr(self, field_name) or fallback
            _hashes((value,), field_name=field_name)
            object.__setattr__(self, field_name, value)
        strategy_version_id = self.strategy_version_id or self.definition.strategy_version_id
        if strategy_version_id != self.definition.strategy_version_id:
            raise ValueError("strategy_version_id does not match the artefact definition")
        object.__setattr__(self, "strategy_version_id", strategy_version_id)
        product_id = self.product_id or (
            self.supported_products[0] if len(self.supported_products) == 1 else None
        )
        if product_id is not None:
            object.__setattr__(self, "product_id", str(product_id))
        for field_name in (
            "portfolio_id",
            "account_id",
            "promotion_policy_id",
            "engine_version",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, str(value))

    @property
    def artefact_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": "platform.strategy_artefact/v2",
            "definition": to_primitive(self.definition),
            "definition_hash": self.definition.definition_hash,
            "dependency_hash": self.dependency_hash,
            "dataset_snapshot_hashes": list(self.dataset_snapshot_hashes),
            "feature_set_version": self.feature_set_version,
            "cost_model_version": self.cost_model_version,
            "validation_evidence": dict(self.validation_evidence),
            "holdout_claim": dict(self.holdout_claim),
            "forward_evidence": dict(self.forward_evidence),
            "promotion_policy": dict(self.promotion_policy),
            "position_limits": dict(self.position_limits),
            "risk_limits": dict(self.risk_limits),
            "model_hashes": list(self.model_hashes),
            "supported_products": list(self.supported_products),
            "supported_instruments": list(self.supported_instruments),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "authoritative_evidence": dict(self.authoritative_evidence),
            "source_commit_hash": self.source_commit_hash,
            "dependency_lock_hash": self.dependency_lock_hash,
            "feature_set_hash": self.feature_set_hash,
            "cost_model_hash": self.cost_model_hash,
            "strategy_version_id": self.strategy_version_id,
            "product_id": self.product_id,
            "portfolio_id": self.portfolio_id,
            "account_id": self.account_id,
            "promotion_policy_id": self.promotion_policy_id,
            "engine_version": self.engine_version,
        }
        if include_hash:
            payload["artefact_hash"] = self.artefact_hash
        return payload

    @classmethod
    def from_authoritative_evidence(
        cls,
        *,
        definition: StrategyDefinition,
        dependency_hash: str,
        dependency_lock_hash: str,
        source_commit_hash: str,
        dataset_snapshot_hashes: tuple[str, ...],
        feature_set_version: str,
        feature_set_hash: str,
        cost_model_version: str,
        cost_model_hash: str,
        validation_stage_ids: tuple[str, ...],
        holdout_claim_id: str,
        forward_evidence_id: str,
        promotion_policy: Mapping[str, Any],
        position_limits: Mapping[str, Any],
        risk_limits: Mapping[str, Any],
        model_hashes: tuple[str, ...],
        supported_products: tuple[str, ...],
        supported_instruments: tuple[str, ...],
        created_at: str,
        validation_evidence: Mapping[str, Any] | None = None,
        holdout_claim: Mapping[str, Any] | None = None,
        forward_evidence: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        product_id: str | None = None,
        portfolio_id: str | None = None,
        account_id: str | None = None,
        promotion_policy_id: str | None = None,
        engine_version: str | None = None,
    ) -> StrategyArtefact:
        """Build v2 only from immutable evidence identities and hashes.

        Acceptance booleans are intentionally not accepted as the source of
        truth. The promotion service resolves these record identities before it
        calls this constructor.
        """

        if not validation_stage_ids or not holdout_claim_id or not forward_evidence_id:
            raise ValueError(
                "v2 artefacts need validation, holdout, and forward evidence identities"
            )
        for field_name, value in (
            ("product_id", product_id or definition.product),
            ("portfolio_id", portfolio_id),
            ("account_id", account_id),
            ("promotion_policy_id", promotion_policy_id),
            ("engine_version", engine_version),
        ):
            if value is None or not str(value).strip():
                raise ValueError(f"v2 artefacts need {field_name}")
        authoritative = {
            "strategy_version_id": definition.strategy_version_id,
            "product_id": product_id or definition.product,
            "portfolio_id": portfolio_id,
            "account_id": account_id,
            "promotion_policy_id": promotion_policy_id,
            "engine_version": engine_version,
            "validation_stage_ids": list(validation_stage_ids),
            "holdout_claim_id": holdout_claim_id,
            "forward_evidence_id": forward_evidence_id,
            "dataset_snapshot_hashes": list(dataset_snapshot_hashes),
            "source_commit_hash": source_commit_hash,
            "dependency_lock_hash": dependency_lock_hash,
            "feature_set_hash": feature_set_hash,
            "cost_model_hash": cost_model_hash,
        }
        return cls(
            definition=definition,
            dependency_hash=dependency_hash,
            dataset_snapshot_hashes=dataset_snapshot_hashes,
            feature_set_version=feature_set_version,
            cost_model_version=cost_model_version,
            validation_evidence=validation_evidence or {},
            holdout_claim=holdout_claim or {},
            forward_evidence=forward_evidence or {},
            promotion_policy=promotion_policy,
            position_limits=position_limits,
            risk_limits=risk_limits,
            model_hashes=model_hashes,
            supported_products=supported_products,
            supported_instruments=supported_instruments,
            created_at=created_at,
            metadata=metadata or {},
            authoritative_evidence=authoritative,
            source_commit_hash=source_commit_hash,
            dependency_lock_hash=dependency_lock_hash,
            feature_set_hash=feature_set_hash,
            cost_model_hash=cost_model_hash,
            strategy_version_id=definition.strategy_version_id,
            product_id=product_id or definition.product,
            portfolio_id=portfolio_id,
            account_id=account_id,
            promotion_policy_id=promotion_policy_id,
            engine_version=engine_version,
        )


class StrategyArtefactStore:
    def __init__(self, root: Path):
        self.root = root

    def put(self, artefact: StrategyArtefact) -> Path:
        digest = artefact.artefact_hash.removeprefix("sha256:")
        destination = self.root / digest[:2] / f"{digest}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(artefact.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        if destination.exists():
            if destination.is_symlink() or destination.read_text(encoding="utf-8") != encoded:
                raise RuntimeError("immutable strategy artefact hash collision")
            return destination
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.read_text(encoding="utf-8") != encoded:
                    raise RuntimeError("immutable strategy artefact hash collision") from None
        finally:
            temporary.unlink(missing_ok=True)
        return destination
