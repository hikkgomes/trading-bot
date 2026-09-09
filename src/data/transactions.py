"""Repository access within a caller-owned financial transaction."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.engine import Connection, Engine


@contextmanager
def connection_scope(
    engine: Engine, connection: Connection | None, *, write: bool
) -> Iterator[Connection]:
    if connection is not None:
        yield connection
    else:
        with engine.begin() if write else engine.connect() as opened:
            yield opened
