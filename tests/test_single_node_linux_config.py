from __future__ import annotations

from src.services.config import load_platform_config


def test_platform_v2_is_single_linux_authority_with_required_services() -> None:
    config = load_platform_config()
    assert config.schema == "platform/v2"
    assert len(config.nodes) == 1
    assert config.nodes[0].operating_system == "linux"
    assert {"strategy-evaluator", "universe-service"}.issubset(config.nodes[0].services)
