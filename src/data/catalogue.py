"""Point-in-time instrument catalogue used by research and execution."""

from __future__ import annotations

from collections.abc import Iterable

from src.domain.instruments import Instrument


class InstrumentCatalogue:
    def __init__(self) -> None:
        self._by_id: dict[str, list[tuple[str, Instrument]]] = {}

    def record(self, instrument: Instrument, *, observed_at: str) -> None:
        history = self._by_id.setdefault(instrument.instrument_id, [])
        if history and observed_at <= history[-1][0]:
            raise ValueError("instrument snapshots must be recorded in chronological order")
        history.append((observed_at, instrument))

    def at(self, instrument_id: str, *, observed_at: str) -> Instrument:
        history = self._by_id.get(instrument_id, [])
        matching = [instrument for timestamp, instrument in history if timestamp <= observed_at]
        if not matching:
            raise KeyError(f"no instrument snapshot available at {observed_at}: {instrument_id}")
        return matching[-1]

    def active_at(self, *, observed_at: str) -> tuple[Instrument, ...]:
        return tuple(
            instrument
            for instrument_id in sorted(self._by_id)
            if (instrument := self.at(instrument_id, observed_at=observed_at)).is_tradable
        )

    def history(self, instrument_id: str) -> tuple[tuple[str, Instrument], ...]:
        return tuple(self._by_id.get(instrument_id, ()))

    def record_many(self, instruments: Iterable[Instrument], *, observed_at: str) -> None:
        for instrument in instruments:
            self.record(instrument, observed_at=observed_at)
