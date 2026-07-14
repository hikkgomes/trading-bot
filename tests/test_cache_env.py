import os
from pathlib import Path

import src.cache_env as cache_env


def test_configure_private_process_cache_uses_private_root(monkeypatch):
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(cache_env, "_PROCESS_CACHE", None)

    cache_env.configure_private_process_cache()

    matplotlib_root = Path(os.environ["MPLCONFIGDIR"])
    xdg_root = Path(os.environ["XDG_CACHE_HOME"])
    assert matplotlib_root.parent == xdg_root.parent
    assert matplotlib_root.parent.name.startswith("trading-bot-cache-")
    assert matplotlib_root.parent.stat().st_mode & 0o077 == 0


def test_configure_private_process_cache_preserves_operator_paths(monkeypatch, tmp_path):
    matplotlib_root = tmp_path / "mpl"
    xdg_root = tmp_path / "xdg"
    monkeypatch.setenv("MPLCONFIGDIR", str(matplotlib_root))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_root))
    monkeypatch.setattr(cache_env, "_PROCESS_CACHE", None)

    cache_env.configure_private_process_cache()

    assert os.environ["MPLCONFIGDIR"] == str(matplotlib_root)
    assert os.environ["XDG_CACHE_HOME"] == str(xdg_root)
    assert cache_env._PROCESS_CACHE is None
