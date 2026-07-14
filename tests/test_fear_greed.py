import json

import pytest

from src import fear_greed


class FakeResponse:
    def __init__(self, payload: bytes, *, url: str = fear_greed._API_URL):
        self._payload = payload
        self._url = url
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def geturl(self):
        return self._url

    def read(self, limit: int):
        return self._payload[:limit]


def _payload() -> bytes:
    return json.dumps(
        {
            "data": [
                {
                    "timestamp": "1704067200",
                    "value": "42",
                    "value_classification": "Fear",
                }
            ]
        }
    ).encode()


def test_fetch_fear_greed_revalidates_remote_and_writes_cache(tmp_path, monkeypatch):
    cache = tmp_path / "fear_greed.parquet"
    monkeypatch.setattr(fear_greed, "_CACHE", cache)
    monkeypatch.setattr(
        fear_greed.urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(_payload()),
    )

    frame = fear_greed.fetch_fear_greed(use_cache=False)

    assert frame["fear_greed"].tolist() == [42]
    assert frame["fear_greed_label"].tolist() == ["Fear"]
    assert cache.exists()


def test_fetch_fear_greed_rejects_cross_host_redirect(tmp_path, monkeypatch):
    monkeypatch.setattr(fear_greed, "_CACHE", tmp_path / "fear_greed.parquet")
    monkeypatch.setattr(
        fear_greed.urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(
            _payload(),
            url="https://attacker.invalid/fng",
        ),
    )

    with pytest.raises(RuntimeError, match="approved HTTPS host"):
        fear_greed.fetch_fear_greed(use_cache=False)


def test_fetch_fear_greed_rejects_oversized_response(tmp_path, monkeypatch):
    monkeypatch.setattr(fear_greed, "_CACHE", tmp_path / "fear_greed.parquet")
    response = FakeResponse(b"{}")
    response.headers["Content-Length"] = str(fear_greed._MAX_RESPONSE_BYTES + 1)
    monkeypatch.setattr(
        fear_greed.urllib.request,
        "urlopen",
        lambda request, timeout: response,
    )

    with pytest.raises(RuntimeError, match="size limit"):
        fear_greed.fetch_fear_greed(use_cache=False)
