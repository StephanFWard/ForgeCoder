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
        # The behavior prompt rides first; the inline rule block follows it.
        assert messages[0]["content"].startswith(load_prompt("edit"))
        assert "OPERATING RULES" in messages[0]["content"]
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


def test_edit_blocks_out_of_scope_patch(client, workspace, fake_inference, monkeypatch):
    """A patch outside the frame's allowed paths must be refused, not previewed."""
    import json

    patch = {"path": "src/other.py", "operations": [
        {"type": "replace", "start_line": 1, "end_line": 1, "content": "x = 1"},
    ]}

    async def fake_chat(messages, **kwargs):
        # The frame in the user turn names the allowed path and the receipt.
        user = messages[-1]["content"]
        assert "src/service.py" in user
        assert "ALLOWED" in user or "allowed" in user
        return json.dumps({"summary": "Drift", "files": [patch]})

    monkeypatch.setattr(fake_inference, "chat", fake_chat)
    body = client.post("/v1/edit", json={
        "message": "Change the constant", "workspace": str(workspace),
        "file": "src/service.py", "code": "return self.store.get(key)",
    }).json()
    assert body["ok"] is False
    assert body["reason"] == "rule_violation"
    assert any(f["slug"] == "allowed-paths-only" for f in body["findings"])


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


def _git_workspace(workspace):
    """Init a repo in the test workspace and commit the fixture file."""
    import subprocess

    for args in (["init", "-q"], ["config", "user.email", "forge@test"],
                 ["config", "user.name", "Forge"]):
        subprocess.run(["git", *args], cwd=str(workspace), check=True)
    subprocess.run(["git", "add", "-A"], cwd=str(workspace), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=str(workspace), check=True)
    return workspace / "src" / "service.py"


def _finding(source: str, line: int) -> str:
    import json

    return json.dumps({"summary": "Guarded the lookup.", "findings": [{
        "source": source, "path": "src/service.py", "line": line,
        "severity": "warning", "message": "Defaulting to None can mask a bad key.",
    }]})


def test_edit_and_fix_are_schema_constrained(client, workspace, fake_inference, monkeypatch):
    """Structured actions constrain the reply instead of trusting the prose contract."""
    import json

    seen = []

    async def fake_chat(messages, **kwargs):
        seen.append(kwargs.get("schema"))
        return json.dumps({
            "summary": "Added validation.",
            "diagnosis": "The key was not checked.",
            "files": [{"path": "src/service.py", "operations": [
                {"type": "replace", "start_line": 2, "end_line": 2, "content": "        self.validate(key)"},
            ]}],
        })

    monkeypatch.setattr(fake_inference, "chat", fake_chat)
    body = {"workspace": str(workspace), "file": "src/service.py",
            "code": "def get(self, key):", "instruction": "add validation"}

    edit = client.post("/v1/edit", json=body).json()
    fix = client.post("/v1/fix", json={**body, "error": "KeyError: 'k'"}).json()

    assert edit["ok"] is True
    assert edit["patches"][0]["path"] == "src/service.py"
    assert fix["ok"] is True
    assert fix["diagnosis"] == "The key was not checked."
    assert seen[0]["$defs"]["FilePatchPayload"]["properties"]["path"]["minLength"] == 1
    assert "diagnosis" in seen[1]["properties"] and "diagnosis" not in seen[0]["properties"]
    # Explain and tests stay free-form: no schema is imposed on prose answers.
    seen.clear()
    client.post("/v1/explain", json=body)
    assert seen == [None]


def test_git_changes_review_is_line_anchored(client, workspace, fake_inference, monkeypatch):
    """The review is schema-constrained, then validated against the supplied diff."""
    file = _git_workspace(workspace)
    file.write_text(
        "class Service:\n    def get(self, key):\n        return self.store.get(key, None)\n",
        encoding="utf-8",
    )
    seen = {}

    async def fake_chat(messages, **kwargs):
        seen["schema"] = kwargs.get("schema")
        seen["user"] = messages[-1]["content"]
        return _finding("unstaged", 3)

    monkeypatch.setattr(fake_inference, "chat", fake_chat)
    body = client.post("/v1/git/changes", json={"workspace": str(workspace)}).json()

    assert body["ok"] is True
    assert body["clean"] is False
    assert body["review_status"] == "validated"
    assert body["findings"] == [{"source": "unstaged", "path": "src/service.py", "line": 3,
                                 "severity": "warning",
                                 "message": "Defaulting to None can mask a bad key."}]
    assert "- [warning] unstaged src/service.py:3:" in body["review"]
    assert "src/service.py" in body["diff"]
    # The call was grammar-constrained and fed the labeled source + path anchors.
    schema = seen["schema"]
    assert schema["properties"]["findings"]["items"] == {"$ref": "#/$defs/Finding"}
    assert {"source", "path", "line", "severity"} <= set(schema["$defs"]["Finding"]["required"])
    assert schema["$defs"]["Finding"]["additionalProperties"] is False
    assert "SOURCE: unstaged" in seen["user"]
    assert "+++ b/src/service.py" in seen["user"]
    # Review must never write to the tree it is reviewing.
    assert file.read_text(encoding="utf-8").endswith("return self.store.get(key, None)\n")


def test_git_changes_rejects_unanchored_finding(client, workspace, fake_inference, monkeypatch):
    """A hallucinated line is rejected, but the raw diff still reaches the user."""
    file = _git_workspace(workspace)
    file.write_text("class Service:\n    def get(self, key):\n        return None\n",
                    encoding="utf-8")

    async def fake_chat(messages, **kwargs):
        return _finding("unstaged", 99)

    monkeypatch.setattr(fake_inference, "chat", fake_chat)
    body = client.post("/v1/git/changes", json={"workspace": str(workspace)}).json()

    assert body["ok"] is True
    assert body["review_status"] == "invalid_output"
    assert body["findings"] == []
    assert "rejected" in body["review"]
    assert "return None" in body["diff"]  # the diff is always returned


def test_git_changes_rejects_wrong_source_line_numbers(client, workspace, fake_inference, monkeypatch):
    """Staged line numbers must not be reported as working-tree line numbers."""
    file = _git_workspace(workspace)
    file.write_text("class Service:\n    def get(self, key):\n        return None\n",
                    encoding="utf-8")
    import subprocess

    subprocess.run(["git", "add", "-A"], cwd=str(workspace), check=True)
    seen = {}

    async def fake_chat(messages, **kwargs):
        seen["user"] = messages[-1]["content"]
        return _finding("unstaged", 3)  # the change is staged, not unstaged

    monkeypatch.setattr(fake_inference, "chat", fake_chat)
    body = client.post("/v1/git/changes", json={"workspace": str(workspace)}).json()

    assert "SOURCE: staged" in seen["user"]
    assert body["review_status"] == "invalid_output"
    assert body["findings"] == []


def test_git_changes_untracked_only_is_insufficient_context(client, workspace, fake_inference, monkeypatch):
    """Untracked files have no diff, so the model is not asked to review them."""
    import subprocess

    for args in (["init", "-q"], ["config", "user.email", "forge@test"],
                 ["config", "user.name", "Forge"]):
        subprocess.run(["git", *args], cwd=str(workspace), check=True)
    subprocess.run(["git", "add", "-A"], cwd=str(workspace), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=str(workspace), check=True)
    (workspace / "src" / "brand_new.py").write_text("NEW = True\n", encoding="utf-8")

    async def fake_chat(messages, **kwargs):
        raise AssertionError("no diff anchors: the model must not be called")

    monkeypatch.setattr(fake_inference, "chat", fake_chat)
    body = client.post("/v1/git/changes", json={"workspace": str(workspace)}).json()

    assert body["ok"] is True
    assert body["review_status"] == "insufficient_context"
    assert body["findings"] == []
    assert "Untracked file contents are not included in Git diffs." in body["review"]
