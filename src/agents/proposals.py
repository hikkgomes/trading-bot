"""Canonical OpenClaw proposal contracts and safety policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from src.domain._codec import canonical_hash, json_value, non_empty, timestamp
from src.domain.strategies import ResearchThesis
from src.services.job_schemas import JobSchemaError, ResearchJobRequest


class AgentRole(StrEnum):
    RESEARCHER = "researcher"
    CRITIC = "critic"
    IMPLEMENTER = "implementer"
    REVIEWER = "reviewer"


class AgentAction(StrEnum):
    CREATE_THESIS = "create_thesis"
    REVISE_STRATEGY = "revise_strategy"
    CREATE_DSL = "create_dsl"
    CREATE_PYTHON_STRATEGY = "create_python_strategy"
    CREATE_FEATURE = "create_feature"
    CREATE_DATA_ADAPTER = "create_data_adapter"
    CREATE_PARAMETER_SPACE = "create_parameter_space"
    CREATE_ML_EXPERIMENT = "create_ml_experiment"
    CREATE_ENSEMBLE = "create_ensemble"
    REQUEST_RESEARCH_TEST = "request_research_test"
    RETIRE_LINEAGE = "retire_lineage"
    CREATE_TESTS = "create_tests"
    RUN_BOUNDED_RESEARCH = "run_bounded_research"
    PRODUCE_BRANCH = "produce_branch"
    PRODUCE_MERGE_REQUEST = "produce_merge_request"
    PRODUCE_EVIDENCE_REPORT = "produce_evidence_report"


ALLOWED_FILE_ROOTS = (
    PurePosixPath("src/strategies/library"),
    PurePosixPath("src/features"),
    PurePosixPath("src/data/adapters"),
    PurePosixPath("src/research"),
    PurePosixPath("tests"),
)
FORBIDDEN_CODE_MARKERS = (
    "src.execution",
    "src.run_bot",
    "ccxt",
    "CcxtBroker",
    "BrokerExecutionVenue",
    ".place_order(",
    "EXCHANGE_API_KEY",
    "EXCHANGE_API_SECRET",
    "TRADING_LIVE",
    "runtime/approvals.json",
)
FORBIDDEN_PROPOSAL_MARKERS = (
    "secret",
    "credential",
    "password",
    "api_key",
    "api_secret",
    "approval",
    "protected",
    "holdout",
    "live",
    "order",
    "risk_decision",
    "promotion",
)


def _assert_safe_provenance(value: Any, *, path: str = "provenance") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in FORBIDDEN_PROPOSAL_MARKERS):
                raise ValueError(f"agent proposal contains forbidden key: {path}.{key}")
            _assert_safe_provenance(item, path=f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _assert_safe_provenance(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class AgentProposal:
    proposal_id: str
    role: AgentRole
    action: AgentAction
    product_id: str
    created_at: str
    thesis: str
    economic_thesis: ResearchThesis | None = None
    files: Mapping[str, str] = field(default_factory=dict)
    research_jobs: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposal_id", non_empty(self.proposal_id, field="proposal_id"))
        object.__setattr__(self, "product_id", non_empty(self.product_id, field="product_id"))
        if self.product_id not in {"btc_accumulation", "active_income"}:
            raise ValueError("agent proposal product is unsupported")
        object.__setattr__(self, "created_at", timestamp(self.created_at, field="created_at"))
        object.__setattr__(self, "thesis", non_empty(self.thesis, field="thesis"))
        if len(self.thesis.encode()) > 32_768:
            raise ValueError("agent proposal thesis is too large")
        if self.economic_thesis is not None and not isinstance(
            self.economic_thesis, ResearchThesis
        ):
            raise ValueError("agent economic thesis must be typed")
        if not isinstance(self.files, Mapping):
            raise ValueError("agent proposal files must be an object")
        normalised_files: dict[str, str] = {}
        total_bytes = 0
        for raw_path, content in self.files.items():
            path = PurePosixPath(non_empty(raw_path, field="proposal file path"))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("agent proposal file paths must stay in the worktree")
            if not any(path == root or root in path.parents for root in ALLOWED_FILE_ROOTS):
                raise ValueError(f"agent proposal file path is outside allowed roots: {path}")
            if not isinstance(content, str):
                raise ValueError("agent proposal file content must be text")
            total_bytes += len(content.encode())
            if any(marker in content for marker in FORBIDDEN_CODE_MARKERS):
                raise ValueError(f"agent proposal contains forbidden execution marker: {path}")
            normalised_files[str(path)] = content
        if len(normalised_files) > 40 or total_bytes > 524_288:
            raise ValueError("agent proposal exceeds the code resource budget")
        object.__setattr__(self, "files", normalised_files)
        if len(self.research_jobs) > 10:
            raise ValueError("agent proposal requests too many research jobs")
        normalised_jobs: list[dict[str, Any]] = []
        forbidden_result_fields = {
            "accepted",
            "evidence",
            "holdout_result",
            "forward_result",
            "limits",
            "metrics",
            "risk_decision",
            "returns",
            "targets",
            "validation",
        }
        for item in self.research_jobs:
            if not isinstance(item, Mapping):
                raise ValueError("agent research requests must be objects")
            request_payload = item.get("request")
            if isinstance(request_payload, Mapping):
                if forbidden_result_fields & (set(item) | set(request_payload)):
                    raise ValueError("agent research requests cannot contain results")
                try:
                    request = ResearchJobRequest.from_mapping(request_payload)
                except JobSchemaError as exc:
                    raise ValueError(f"invalid agent research request: {exc}") from exc
                name = non_empty(str(item.get("name") or "evaluate_candidate"), field="job name")
                if name != "evaluate_candidate":
                    raise ValueError("agent research requests must use the typed evaluator command")
                normalised_jobs.append(
                    {
                        "name": name,
                        "maximum_seconds": item.get("maximum_seconds", 60),
                        "request": request.to_payload(),
                    }
                )
                continue
            if forbidden_result_fields & set(item):
                raise ValueError("agent research requests cannot contain results")
            # An empty request is retained for proposal review compatibility,
            # but the worker will not enqueue it as executable work.
            normalised_jobs.append(json_value(dict(item), field="research job"))
        object.__setattr__(
            self,
            "research_jobs",
            tuple(normalised_jobs),
        )
        if not isinstance(self.provenance, Mapping):
            raise ValueError("agent proposal provenance must be an object")
        _assert_safe_provenance(self.provenance)
        object.__setattr__(
            self, "provenance", json_value(dict(self.provenance), field="provenance")
        )

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)
