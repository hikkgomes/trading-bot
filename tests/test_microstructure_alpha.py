from src.alpha.microstructure import MicrostructureAlphaPolicy, forecast_from_microstructure


def _features(**overrides):
    payload = {
        "ok": True,
        "symbol": "BTCUSDT",
        "spread_bps": 2.0,
        "bid_depth": 10.0,
        "ask_depth": 2.0,
        "weighted_depth_imbalance": 0.8,
        "aggressor_imbalance": 0.7,
        "microprice_dislocation_bps": 4.0,
        "cancel_add_pressure": 0.5,
        "liquidity_vacuum_ratio": 1.0,
        "liquidation_imbalance": 0.4,
    }
    payload.update(overrides)
    return payload


def test_microstructure_alpha_combines_book_flow_and_cancel_pressure():
    forecast, detail = forecast_from_microstructure(
        _features(), generated_at="2026-08-10T00:00:00+00:00"
    )

    assert forecast is not None
    assert forecast.direction == "long"
    assert forecast.horizon_seconds == 30
    assert detail["eligible"] is True
    assert detail["components"]["liquidation"] == 0.4


def test_microstructure_alpha_blocks_wide_or_weak_books():
    forecast, detail = forecast_from_microstructure(
        _features(spread_bps=20),
        policy=MicrostructureAlphaPolicy(maximum_spread_bps=5),
        generated_at="2026-08-10T00:00:00+00:00",
    )

    assert forecast is None
    assert detail["reason"] == "microstructure_liquidity_gate"
