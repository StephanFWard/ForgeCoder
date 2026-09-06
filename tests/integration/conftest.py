"""Integration test configuration: real app, mocked inference, temp workspace."""
import sys
from pathlib import Path

import pytest

# Make both source trees importable without a pip install.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps" / "server"))


class FakeInference:
    """Minimal inference stand-in: no llama.cpp needed for API tests."""

    def __init__(self):
        self.chat_calls = 0

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None

    async def chat(self, messages, **kwargs) -> str:
        self.chat_calls += 1
        return "Fake explanation for the supplied code."

    async def chat_stream(self, messages, **kwargs):
        for piece in ("Fake ", "streaming ", "reply."):
            yield piece

    async def complete(self, prompt, **kwargs) -> str:
        return "return repository.findById(id);"

    def server_info(self) -> dict:
        return {}


@pytest.fixture()
def fake_inference():
    return FakeInference()


@pytest.fixture()
def workspace(tmp_path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "service.py").write_text(
        "class Service:\n    def get(self, key):\n        return self.store.get(key)\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def client(tmp_path, fake_inference, monkeypatch):
    """Build the FastAPI app with the fake inference and a temp DB."""
    from forge_server.config import Config
    from forge_server.main import AppState, create_app

    from core.inference import client as inference_module

    monkeypatch.setattr(inference_module.InferenceClient, "ping", fake_inference.ping)

    cfg = Config.load()
    cfg.data_dir = str(tmp_path)
    cfg.db_path = str(tmp_path / "forge.db")

    state = AppState(cfg)
    # Replace the real inference client with the fake.
    state.inference = fake_inference  # type: ignore[assignment]

    app = create_app(cfg)
    app.state.forge = state

    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client

    state.index.close()


@pytest.fixture()
def indexed_client(client, workspace):
    client.post("/v1/index", json={"workspace": str(workspace)})
    return client
