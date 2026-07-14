"""Append-only experiment log: what was tested, and the verdict.

The point of the research loop is that we never re-test the same idea blindly and
never lose a result. Each evaluation appends one JSON line recording the
hypothesis, the exact test config (so results are reproducible), the headline
metrics, and a keep/reject/inconclusive verdict.

A small content hash over (hypothesis + config) lets ``already_tested`` skip work
that's already been done.

Run:  python -m research_exploration.experiment_log --summary
      python -m research_exploration.experiment_log --tail 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_LOG = Path("outputs/research_exploration/experiment_log.jsonl")

VERDICTS = ("keep", "reject", "inconclusive", "needs_data", "error")


def _git_sha() -> str | None:
    try:
        import subprocess

        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def fingerprint(hypothesis_dict: dict, config: dict) -> str:
    """Stable short hash over the idea + how it was tested."""
    blob = json.dumps({"h": hypothesis_dict, "c": config}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class ExperimentRecord:
    hypothesis_id: str
    family: str
    direction: str
    fingerprint: str
    verdict: str  # one of VERDICTS
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, object] = field(default_factory=dict)
    data_window: dict[str, str] = field(default_factory=dict)  # train/val/holdout ranges
    notes: str = ""
    hypothesis: dict | None = None  # full hypothesis dict for provenance
    timestamp: str = ""
    git_sha: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {self.verdict!r}")
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()
        if self.git_sha is None:
            self.git_sha = _git_sha()


def log_result(record: ExperimentRecord, log_path: Path = DEFAULT_LOG) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record), default=str) + "\n")


def load_log(log_path: Path = DEFAULT_LOG) -> list[dict]:
    if not Path(log_path).exists():
        return []
    out = []
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def already_tested(fp: str, log_path: Path = DEFAULT_LOG) -> bool:
    return any(r.get("fingerprint") == fp for r in load_log(log_path))


def summarize(log_path: Path = DEFAULT_LOG) -> str:
    rows = load_log(log_path)
    if not rows:
        return "No experiments logged yet."
    from collections import Counter

    by_verdict = Counter(r["verdict"] for r in rows)
    by_family = Counter(r["family"] for r in rows)
    keepers = [r for r in rows if r["verdict"] == "keep"]
    lines = [f"{len(rows)} experiments logged.", "", "By verdict:"]
    lines += [f"  {v:14} {n}" for v, n in by_verdict.most_common()]
    lines += ["", "By family:"]
    lines += [f"  {f:24} {n}" for f, n in by_family.most_common()]
    if keepers:
        lines += ["", "Kept (candidate edges):"]
        for r in keepers:
            m = r.get("metrics", {})
            lines.append(
                f"  {r['hypothesis_id']:40} trades={m.get('trades', '?')} "
                f"sharpe={m.get('sharpe', '?')} dsr={m.get('psr', m.get('dsr', '?'))}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the experiment log.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--tail", type=int, default=0)
    args = parser.parse_args()

    if args.tail:
        for r in load_log(args.log)[-args.tail :]:
            m = r.get("metrics", {})
            print(f"[{r['timestamp'][:19]}] {r['verdict']:12} {r['hypothesis_id']}  {m}")
    if args.summary or not args.tail:
        print(summarize(args.log))


if __name__ == "__main__":
    main()
