"""Integration tests for the Forge API with a mocked inference backend."""


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llama_server"]["ok"] is True
    assert "database" in body


def test_loopback_guard_blocks_foreign_clients():
    """Security model: non-loopback clients are refused outright."""
    import pytest
    from forge_server.security import LOOPBACK_HOSTS, ForgeSecurityError, assert_local_request

    for host in LOOPBACK_HOSTS:
        assert_local_request(host)  # all loopback forms pass

    for foreign in ("10.0.0.5", "192.168.1.20", "evil.example.com"):
        with pytest.raises(ForgeSecurityError):
            assert_local_request(foreign)


def test_loopback_guard_allows_test_transport():
    """Starlette's TestClient transport uses the synthetic 'testclient' host."""
    from forge_server.security import assert_local_request

    assert_local_request("testclient")


def test_chat_streams(client, workspace):
    payload = {
        "message": "Why does get return null?",
        "workspace": str(workspace),
        "file": "src/service.py",
        "selection": {"start": 1, "end": 4},
    }
    with client.stream("POST", "/v1/chat", json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = "".join(resp.iter_text())
    assert "data: " in lines
    assert '"type": "delta"' in lines
    assert '"type": "done"' in lines


def test_chat_without_stream_fields(client):
    # history accepted and clipped
    payload = {"message": "hi", "history": [{"role": "user", "content": "prev"}, {"role": "assistant", "content": "old"}]}
    with client.stream("POST", "/v1/chat", json=payload) as resp:
        assert resp.status_code == 200


def test_completion(client):
    resp = client.post(
        "/v1/completion",
        json={"language": "java", "file": "UserService.java",
              "prefix": "public User findUser(Long id) {", "suffix": "\n}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["completion"] == "return repository.findById(id);"


def test_index_and_search(client, workspace):
    resp = client.post("/v1/index", json={"workspace": str(workspace)})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    search = client.post("/v1/search", json={"query": "Service", "workspace": str(workspace)})
    body = search.json()
    assert body["ok"] is True
    assert body["results"]
    assert body["results"][0]["path"] == "src/service.py"


def test_patch_preview_no_write(client, workspace):
    file = workspace / "src" / "service.py"
    original = file.read_text(encoding="utf-8")
    patch = {
        "workspace": str(workspace),
        "patch": {
            "path": "src/service.py",
            "operations": [{"type": "replace", "start_line": 1, "end_line": 2, "content": "class Service:"}],
        },
    }
    resp = client.post("/v1/patch/preview", json=patch)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["diff"].startswith("--- a/src/service.py")
    # File on disk untouched by preview
    assert file.read_text(encoding="utf-8") == original


def test_patch_apply_requires_confirmation(client, workspace):
    patch = {
        "workspace": str(workspace),
        "patch": {
            "path": "src/service.py",
            "operations": [{"type": "replace", "start_line": 1, "end_line": 1, "content": "# edited"}],
        },
    }
    resp = client.post("/v1/patch/apply", json=patch)
    assert resp.status_code == 200
    assert resp.json()["ok"] is False  # no X-Forge-Confirm header

    resp2 = client.post(
        "/v1/patch/apply", json=patch, headers={"X-Forge-Confirm": "true"},
    )
    assert resp2.json()["ok"] is True
    assert (workspace / "src" / "service.py").read_text(encoding="utf-8").startswith("# edited")


def test_patch_rejects_path_escape(client, workspace):
    patch = {
        "workspace": str(workspace),
        "patch": {
            "path": "../escape.txt",
            "operations": [{"type": "replace", "start_line": 1, "end_line": 1, "content": "x"}],
        },
    }
    resp = client.post("/v1/patch/preview", json=patch)
    assert resp.json()["ok"] is False
    assert "escape" in resp.json()["error"]


def test_explain_action(client, workspace):
    resp = client.post(
        "/v1/explain",
        json={"code": "def f():\n    pass", "workspace": str(workspace), "file": "src/service.py"},
    )
    body = resp.json()
    assert body["ok"] is True
    assert "Fake explanation" in body["explanation"]


def test_fix_returns_fallback_when_not_json(client, workspace):
    """Without a JSON-constrained backend, /v1/fix must degrade gracefully."""
    resp = client.post(
        "/v1/fix",
        json={"code": "x = None\ny = x.name", "error": "AttributeError: 'NoneType'",
              "workspace": str(workspace)},
    )
    body = resp.json()
    assert body["ok"] is False
    assert body["reason"] == "no_structured_patch"
    assert "proposal" in body or "diagnosis" in body


def test_code_actions_use_behavior_prompt_and_context(client, workspace, fake_inference, monkeypatch):
    """Regression: /v1/tests receives the test prompt and current-file context."""
    captured: dict = {}

    async def fake_chat(messages, **kwargs):
        captured["messages"] = messages
        return "ok"

    monkeypatch.setattr(fake_inference, "chat", fake_chat)
    resp = client.post(
        "/v1/tests",
        json={"code": "def get(self, key):\n    return self.store.get(key)",
              "workspace": str(workspace), "file": "src/service.py"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    roles = [m["role"] for m in captured["messages"]]
    assert roles[0] == "system"
    system = captured["messages"][0]["content"]
    user = captured["messages"][-1]["content"]
    # 1. Behavior-specific system prompt (test.txt), not the generic chat one.
    assert "unit tests" in system
    assert "software engineering assistant" not in system
    # 2. Retrieved repository context is injected with real paths + line numbers.
    assert "src/service.py" in user
    assert "class Service" in user


def test_edit_instruction_preview_and_local_apply(client, workspace, fake_inference, monkeypatch):
    """Generated edits preserve the request and write only after confirmation."""
    import json

    from core.retrieval.context import load_prompt

    file = workspace / "src" / "service.py"
    original = file.read_text(encoding="utf-8")
    replacement = "        return self.store.get(key, None)"
    patch = {"path": "src/service.py", "operations": [
        {"type": "replace", "start_line": 3, "end_line": 3, "content": replacement},
    ]}

    async def fake_chat(messages, **kwargs):
        assert messages[0]["content"] == load_prompt("edit")
        user = messages[-1]["content"]
        for text in ("Update lookup", "Use an explicit None default", "Selected code:",
                     "src/service.py", "class Service:"):
            assert text in user
        return json.dumps({"summary": "Use explicit default", "files": [patch]})

    monkeypatch.setattr(fake_inference, "chat", fake_chat)
    generated = client.post("/v1/edit", json={
        "message": "Update lookup", "instruction": "Use an explicit None default",
        "code": "return self.store.get(key)", "workspace": str(workspace),
        "file": "src/service.py",
    }).json()
    assert generated["ok"] is True
    assert generated["patches"] == [patch]
    assert file.read_text(encoding="utf-8") == original

    body = {"workspace": str(workspace), "patch": generated["patches"][0]}
    preview = client.post("/v1/patch/preview", json=body).json()
    assert preview["ok"] is True
    assert replacement in preview["proposed"]
    assert file.read_text(encoding="utf-8") == original
    assert client.post("/v1/patch/apply", json=body).json()["ok"] is False
    assert file.read_text(encoding="utf-8") == original

    applied = client.post("/v1/patch/apply", json=body,
                          headers={"X-Forge-Confirm": "true"}).json()
    assert applied["ok"] is True
    assert applied["applied"] is True
    assert file.read_text(encoding="utf-8") == preview["proposed"]
    compile(file.read_text(encoding="utf-8"), str(file), "exec")


def test_plan_and_explain_step(indexed_client, workspace, fake_inference, monkeypatch):
    import json

    calls = []

    async def fake_chat(messages, **kwargs):
        calls.append(messages)
        if kwargs.get("schema"):
            return json.dumps({"summary": "Explain Service", "steps": [
                {"title": "Explain lookup", "action": "explain", "detail": "Service"},
            ]})
        return "Service returns a stored value."

    monkeypatch.setattr(fake_inference, "chat", fake_chat)
    plan = indexed_client.post("/v1/plan", json={
        "message": "Service", "workspace": str(workspace),
    }).json()
    assert plan["ok"] is True
    assert not plan.get("fallback")
    result = indexed_client.post("/v1/act", json={
        "plan_id": plan["plan_id"], "index": 0, "workspace": str(workspace),
    }).json()
    assert result["ok"] is True
    assert result["output"] == "Service returns a stored value."
    assert "verification" in result
    assert len(calls) == 2
    assert "src/service.py" in calls[-1][-1]["content"]


def test_apply_rejects_changed_original(client, workspace):
    file = workspace / "src" / "service.py"
    original = file.read_text(encoding="utf-8")
    body = {"workspace": str(workspace), "expected_original": original, "patch": {
        "path": "src/service.py", "operations": [
            {"type": "replace", "start_line": 3, "end_line": 3, "content": "        return None"},
        ],
    }}
    changed = original + "# concurrent edit\n"
    file.write_text(changed, encoding="utf-8")
    result = client.post("/v1/patch/apply", json=body, headers={"X-Forge-Confirm": "true"}).json()
    assert result["ok"] is False
    assert "changed" in result["error"]
    assert file.read_text(encoding="utf-8") == changed
    body["expected_original"] = changed
    result = client.post("/v1/patch/apply", json=body, headers={"X-Forge-Confirm": "true"}).json()
    assert result["ok"] is True
    assert "return None" in file.read_text(encoding="utf-8")
