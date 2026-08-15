"""Canonical market-data catalogue, snapshots, and immutable file storage."""

from src.data.binance_market import normalise_public_event
from src.data.binance_user_stream import normalise_user_event
from src.data.catalogue import InstrumentCatalogue
from src.data.feature_store import DeterministicFeatureCalculator, FeatureValue, SqlFeatureStore
from src.data.historical_query import DuckDBHistoricalQuery
from src.data.parquet_store import ContentAddressedStore, PartitionedMarketEventStore
from src.data.snapshots import DatasetSnapshot
from src.data.universe import (
    InstrumentObservation,
    SqlUniverseStore,
    UniverseEligibilityPolicy,
    UniverseMembership,
)

__all__ = [
    "ContentAddressedStore",
    "DatasetSnapshot",
    "DeterministicFeatureCalculator",
    "DuckDBHistoricalQuery",
    "FeatureValue",
    "InstrumentCatalogue",
    "InstrumentObservation",
    "PartitionedMarketEventStore",
    "SqlUniverseStore",
    "SqlFeatureStore",
    "UniverseEligibilityPolicy",
    "UniverseMembership",
    "normalise_public_event",
    "normalise_user_event",
]
