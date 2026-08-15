"""Small dependency-free Prometheus metrics registry."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


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
