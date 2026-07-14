from pathlib import Path

from src import config


def test_candle_data_dir_ignores_inaccessible_legacy_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    legacy = tmp_path / "data" / "candles" / "BTCUSDT"
    expected = tmp_path / "data" / "candles" / "futures" / "BTCUSDT"
    original_exists = Path.exists

    def restricted_exists(path):
        if path == legacy:
            raise PermissionError("market data is intentionally inaccessible")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", restricted_exists)

    assert config.candle_data_dir("BTCUSDT", "futures") == expected
