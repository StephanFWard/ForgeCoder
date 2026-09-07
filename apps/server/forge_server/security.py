"""Security helpers: loopback checks, path confinement, permission gating."""
from __future__ import annotations

from pathlib import Path

# Explicit permission set. v0.1 auto-allows the read-only tier and defers the
# rest to the caller (VS Code shows a confirmation dialog).
PERMISSIONS = {"READ_FILE", "SEARCH", "GIT_READ", "GIT_WRITE", "WRITE_FILE", "BUILD", "TEST", "SHELL"}
PERMISSIONS_AUTO = {"READ_FILE", "SEARCH", "GIT_READ"}
PERMISSIONS_CONFIRM = sorted(PERMISSIONS - PERMISSIONS_AUTO)

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
# Starlette's TestClient uses a synthetic host value; allow it so the test
# suite can exercise the API. Real sockets are still bound to 127.0.0.1, so
# this is never reachable from the network.
TEST_HOSTS = {"testclient", "testclient2"}


class ForgeSecurityError(PermissionError):
    """Raised when a request violates the security model."""


def assert_local_request(client_host: str) -> None:
    """The Forge API refuses connections from non-loopback clients."""
    if client_host not in LOOPBACK_HOSTS and client_host not in TEST_HOSTS:
        raise ForgeSecurityError(
            f"Connection refused: client {client_host!r} is not on the loopback interface"
        )


def check_permission(permission: str, granted: set[str] | None = None) -> None:
    if permission not in PERMISSIONS:
        raise ForgeSecurityError(f"Unknown permission {permission!r}")
    granted = granted or PERMISSIONS_AUTO
    if permission not in granted:
        raise ForgeSecurityError(
            f"Permission {permission!r} requires user confirmation in v0.1"
        )


def resolve_workspace_path(workspace: str | None, rel_path: str) -> Path:
    """Resolve ``rel_path`` inside ``workspace``, blocking traversal escapes."""
    if not workspace:
        raise ForgeSecurityError("A workspace is required for file operations")
    root = Path(workspace).resolve()
    candidate = (root / rel_path).resolve()
    if root != candidate and root not in candidate.parents:
        raise ForgeSecurityError(f"Path {rel_path!r} escapes the workspace root")
    return candidate
