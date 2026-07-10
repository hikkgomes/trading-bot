"""Small, dependency-free helpers for reading dotenv-style files."""

from __future__ import annotations

from collections.abc import Iterable


def parse_env_value(raw: str) -> str:
    """Parse one dotenv value, preserving hashes inside quotes.

    An unquoted ``#`` starts a comment only at the beginning of the value or
    when preceded by whitespace. This keeps tokens such as ``abc#123`` intact
    while accepting the common ``KEY=value  # explanation`` form.
    """

    quote: str | None = None
    escaped = False
    comment_at: int | None = None
    for index, char in enumerate(raw):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None and (index == 0 or raw[index - 1].isspace()):
            comment_at = index
            break

    value = raw[:comment_at].strip() if comment_at is not None else raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_env_lines(lines: Iterable[str]) -> dict[str, str]:
    """Return dotenv assignments from *lines* without mutating ``os.environ``."""

    values: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        values[key] = parse_env_value(value)
    return values
