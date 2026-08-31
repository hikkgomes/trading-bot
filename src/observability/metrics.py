"""Small dependency-free Prometheus metrics registry."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


@dataclass
class MetricsRegistry:
    gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(default_factory=dict)
    counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(default_factory=dict)

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        self._validate(name, value)
        self.gauges[(name, tuple(sorted(labels.items())))] = float(value)

    def increment(self, name: str, amount: float = 1.0, **labels: str) -> None:
        self._validate(name, amount)
        if amount < 0:
            raise ValueError("counter increments cannot be negative")
        key = (name, tuple(sorted(labels.items())))
        self.counters[key] = self.counters.get(key, 0.0) + amount

    def render(self) -> str:
        lines: list[str] = []
        for metric_type, values in (("counter", self.counters), ("gauge", self.gauges)):
            names = sorted({name for name, _labels in values})
            for name in names:
                lines.append(f"# TYPE {name} {metric_type}")
                for (metric_name, labels), value in sorted(values.items()):
                    if metric_name != name:
                        continue
                    suffix = ""
                    if labels:
                        suffix = (
                            "{"
                            + ",".join(f'{key}="{_label_value(value)}"' for key, value in labels)
                            + "}"
                        )
                    lines.append(f"{name}{suffix} {value:g}")
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _validate(name: str, value: float) -> None:
        if not name or not name.replace("_", "a").isalnum():
            raise ValueError("metric names must contain letters, digits, and underscores")
        if not math.isfinite(float(value)):
            raise ValueError("metric values must be finite")


class DatabaseMetricsProvider:
    """Render safe operational gauges from the canonical platform report."""

    def __init__(self, report_builder: Callable[[], Mapping[str, object]]) -> None:
        self.report_builder = report_builder

    def render(self) -> str:
        registry = MetricsRegistry()
        try:
            report = self.report_builder()
            self._record_report(registry, report)
        except Exception:
            registry.set_gauge("platform_report_available", 0.0)
        else:
            registry.set_gauge("platform_report_available", 1.0)
        return registry.render()

    @staticmethod
    def _record_report(registry: MetricsRegistry, report: Mapping[str, object]) -> None:
        funnel = report.get("research")
        funnel = funnel.get("funnel", {}) if isinstance(funnel, Mapping) else {}
        for name in (
            "theses_generated",
            "candidates_generated",
            "candidates_evaluated",
            "forward_paper_count",
            "active_forward_count",
            "strategy_promotions",
            "jobs_waiting",
            "jobs_running",
            "jobs_completed",
            "jobs_dead_letter",
            "missing_stage_dataset_count",
        ):
            registry.set_gauge(
                f"platform_{name}",
                _numeric(funnel.get(name)),
            )
        rejection_reasons = funnel.get("top_rejection_reasons")
        if isinstance(rejection_reasons, Mapping):
            for reason, count in rejection_reasons.items():
                registry.set_gauge(
                    "platform_rejections",
                    _numeric(count),
                    reason=str(reason),
                )
        operations = report.get("operations")
        operations = operations if isinstance(operations, Mapping) else {}
        slis = operations.get("slis")
        slis = slis if isinstance(slis, Mapping) else {}
        for name in (
            "unresolved_recovery_count",
        ):
            registry.set_gauge(f"platform_{name}", _numeric(slis.get(name)))
        for group, metric in (
            ("stale_account_authority", "platform_stale_account_authority"),
            ("stale_market_data", "platform_stale_market_data"),
            ("missing_risk_data", "platform_missing_risk_data"),
        ):
            value = slis.get(group)
            value = value.get("count", 0) if isinstance(value, Mapping) else 0
            registry.set_gauge(metric, _numeric(value))


def build_metrics_server(
    *,
    bind: tuple[str, int],
    provider: Callable[[], str],
) -> ThreadingHTTPServer:
    """Build a local Prometheus HTTP server for the platform report."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/metrics":
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()
                return
            body = provider().encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer(bind, Handler)


def _numeric(value: object) -> float:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0
