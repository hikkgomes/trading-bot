"""Pin research scope and advance epochs only onto an untouched holdout interval."""

import datetime as dt

from sqlalchemy import select

from src.data.database import (
    dataset_snapshot,
    experiment,
    research_epoch,
    selection_cohort_open,
    universe_snapshot,
)
from src.data.universe import SqlUniverseStore
from src.domain._codec import canonical_hash, timeframe_seconds
from src.research.data_plan import product_data_plan
from src.research.datasets import SqlDatasetBundleRepository
from src.research.outcomes import terminal_trial_disposition


def research_inputs(engine, product, research, now):
    product_id = product["product_id"]
    bundle = SqlDatasetBundleRepository(engine).latest_ready(product_id, at=now)
    previous = None
    if bundle is not None and _matches_plan(bundle, product, research):
        successor_at = _successor_at(engine, bundle, product)
        if successor_at is None or now < successor_at:
            return _bundle_inputs(engine, bundle, product, research, successor_at)
        previous = bundle.bundle_id
    specification = canonical_hash(
        {"product": product, "research": research, "previous_bundle_id": previous}
    )
    scope = _pinned_scope(engine, product, research, now, specification)
    if scope is None:
        return {"instrument_universe": [], "dataset_snapshot_hashes": []}
    symbols = sorted(
        item["instrument"]["exchange_symbol"] for item in scope["payload"]["observations"]
    )
    plan = product_data_plan(product, research, symbols)
    return {
        "instrument_universe": symbols,
        "data_plan": plan.to_payload(),
        "dataset_snapshot_hashes": [],
        "universe_id": scope["universe_id"],
        "universe_snapshot_id": scope["id"],
        "research_as_of": scope["observed_at"],
        "catalogue_submitted_at": scope["observed_at"],
        "previous_bundle_id": previous,
    }


def _matches_plan(bundle, product, research):
    symbols = [item.split(":")[2] if ":" in item else item for item in bundle.instrument_scope]
    return bundle.data_plan_id == product_data_plan(product, research, symbols).plan_id


def _successor_at(engine, bundle, product):
    with engine.connect() as connection:
        state = connection.execute(
            select(research_epoch.c.state).where(
                research_epoch.c.dataset_bundle_id == bundle.bundle_id
            )
        ).scalar_one_or_none()
        if state not in {"exhausted", "closed_for_selection"}:
            return None
        trials = (
            connection.execute(
                select(experiment.c.state, experiment.c.metadata).where(
                    experiment.c.dataset_bundle_id == bundle.bundle_id
                )
            )
            .mappings()
            .all()
        )
        if any(not _trial_finished(row) for row in trials):
            return None
        opened = connection.execute(
            select(selection_cohort_open.c.id)
            .where(
                selection_cohort_open.c.payload["protected_snapshot_id"].as_string()
                == bundle.stage_snapshot_ids["protected_holdout"]
            )
            .limit(1)
        ).first()
        if opened is None:
            return (dt.datetime.fromisoformat(bundle.created_at) + dt.timedelta(days=1)).isoformat()
        interval = connection.execute(
            select(dataset_snapshot.c.payload["interval"]).where(
                dataset_snapshot.c.id == bundle.stage_snapshot_ids["protected_holdout"]
            )
        ).scalar_one()
    end, start = (dt.datetime.fromisoformat(interval[key]) for key in ("end", "start"))
    # Current canonical splitter uses four equal chronological partitions.
    # Two bars cover endpoint rounding so the successor cannot reuse holdout bars.
    margin = dt.timedelta(seconds=2 * timeframe_seconds(product["dataset_timeframe"]))
    return (end + (end - start) + margin).isoformat()


def _trial_finished(row):
    state = str(row["state"])
    return (
        state.endswith("_rejected")
        or state in {"protected", "forward", "forward_paper", "completed", "retired"}
        or terminal_trial_disposition(state, row["metadata"]) is not None
    )


def _bundle_inputs(engine, bundle, product, research, successor_at):
    with engine.connect() as connection:
        scope = (
            connection.execute(
                select(universe_snapshot.c.universe_id, universe_snapshot.c.observed_at).where(
                    universe_snapshot.c.id == bundle.universe_snapshot_id
                )
            )
            .mappings()
            .one()
        )
    symbols = sorted(
        item.split(":")[2] if ":" in item else item for item in bundle.instrument_scope
    )
    return {
        "instrument_universe": symbols,
        "data_plan": product_data_plan(product, research, symbols).to_payload(),
        "dataset_snapshot_hashes": list(bundle.stage_snapshot_ids.values()),
        "dataset_bundle_id": bundle.bundle_id,
        "universe_id": scope["universe_id"],
        "universe_snapshot_id": bundle.universe_snapshot_id,
        "research_as_of": scope["observed_at"],
        "catalogue_submitted_at": bundle.created_at,
        "successor_eligible_at": successor_at,
    }


def _pinned_scope(engine, product, research, now, specification):
    from src.services.universe_service import DatabaseUniverseService

    statement = select(universe_snapshot).where(
        universe_snapshot.c.payload["specification_hash"].as_string() == specification,
        universe_snapshot.c.observed_at <= now,
    )
    with engine.connect() as connection:
        existing = (
            connection.execute(
                statement.order_by(universe_snapshot.c.observed_at, universe_snapshot.c.id).limit(1)
            )
            .mappings()
            .first()
        )
    if existing is not None:
        return existing
    scope = DatabaseUniverseService(store=SqlUniverseStore(engine)).resolve_research_scope(
        product_id=product["product_id"],
        source_universe_id=product["universe_id"],
        selected_at=now,
        maximum_symbols=int(product.get("research_max_symbols", 1)),
        minimum_history_days=int(
            research.get("data_requirements", {}).get("minimum_history_days", 365)
        ),
        specification_hash=specification,
    )
    if scope.snapshot_id is None:
        return None
    with engine.connect() as connection:
        return (
            connection.execute(
                select(universe_snapshot).where(universe_snapshot.c.id == scope.snapshot_id)
            )
            .mappings()
            .one()
        )
