"""Unit tests for the Forge server (no HTTP) — security + config + patch surface."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps" / "server"))

from forge_server import config as config_module  # noqa: E402 (path setup above)
from forge_server.config import Config  # noqa: E402
from forge_server.security import (  # noqa: E402
    ForgeSecurityError,
    assert_local_request,
    check_permission,
    resolve_workspace_path,
)


@pytest.fixture(autouse=True)
def _hermetic_config(tmp_path, monkeypatch):
    """Isolate every test from any real %LOCALAPPDATA% config."""
    monkeypatch.setattr(config_module, "default_config_path", lambda: tmp_path / "config.json")
    monkeypatch.setattr(config_module, "default_data_dir", lambda: tmp_path)
    yield


def test_config_loads_defaults():
    cfg = Config.load()
    assert cfg.host == "127.0.0.1"
    assert cfg.max_chat_context == 4096
    assert cfg.max_completion_context == 1024
    assert cfg.completion_temperature == 0.1


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("FORGECODER_PORT", "9797")
    monkeypatch.setenv("FORGECODER_MAX_CHAT_CONTEXT", "2048")
    cfg = Config.load()
    assert cfg.port == 9797
    assert cfg.max_chat_context == 2048


def test_config_host_cannot_be_non_loopback(monkeypatch):
    monkeypatch.setenv("FORGECODER_HOST", "0.0.0.0")
    cfg = Config.load()
    assert cfg.host == "127.0.0.1"  # hard architecture rule


def test_security_resolve_workspace_path_blocks_escape(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    within = resolve_workspace_path(str(root), "src/a.py")
    assert "a.py" in within.name
    with pytest.raises(ForgeSecurityError):
        resolve_workspace_path(str(root), "../outside.txt")


def test_security_loopback_guard():
    for host in ("127.0.0.1", "::1", "localhost", "testclient"):
        assert_local_request(host)
    for foreign in ("10.0.0.5", "0.0.0.0"):
        with pytest.raises(ForgeSecurityError):
            assert_local_request(foreign)


def test_security_permissions():
    check_permission("SEARCH")  # auto-allowed tier
    for denied in ("WRITE_FILE", "BUILD", "TEST", "SHELL"):
        with pytest.raises(ForgeSecurityError):
            check_permission(denied)  # requires confirmation in v0.1
    with pytest.raises(ForgeSecurityError):
        check_permission("NOT_A_PERMISSION")


def test_config_write_self_roundtrip(tmp_path):
    cfg = Config.load()
    cfg.port = 8123
    cfg.write_self()
    loaded = Config.load()
    assert loaded.port == 8123
    assert (tmp_path / "config.json").exists()
