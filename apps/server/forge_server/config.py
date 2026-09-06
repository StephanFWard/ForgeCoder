"""Server configuration.

Resolution order:
  1. Environment variables  FORGECODER_*   (highest)
  2. %LOCALAPPDATA%/ForgeCoder/config.json
  3. Built-in defaults

The config file is also where the hardware profile from first launch lives.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_LLAMA_URL = "http://127.0.0.1:8080"

# The network default is loopback-only. Never flip this to 0.0.0.0.
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def default_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "ForgeCoder"
    return Path.home() / ".forgecoder"


def default_config_path() -> Path:
    return default_data_dir() / "config.json"


@dataclass
class Config:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    llama_url: str = DEFAULT_LLAMA_URL
    workspace: str | None = None
    db_path: str | None = None
    data_dir: str | None = None
    max_chat_context: int = 4096
    max_completion_context: int = 1024
    max_retrieved_chunks: int = 6
    retrieval_page_size: int = 20
    completion_max_tokens: int = 64
    completion_temperature: float = 0.1
    stream: bool = True

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        cfg_path = path or default_config_path()
        raw: dict[str, Any] = {}
        if cfg_path.exists():
            try:
                loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    raw = loaded
            except (json.JSONDecodeError, OSError):
                raw = {}
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in raw.items() if k in known and v is not None}
        cfg = cls(**clean)
        cfg._apply_env()
        cfg._validate()
        return cfg

    def _apply_env(self) -> None:
        for f in fields(self):
            env_name = f"FORGECODER_{f.name.upper()}"
            value = os.environ.get(env_name)
            if value is None or value == "":
                continue
            cast = self._cast(f.type, value)
            setattr(self, f.name, cast)

    @staticmethod
    def _cast(annotation: type | str, value: str) -> object:
        annotation_str = getattr(annotation, "__name__", str(annotation))
        if annotation_str == "int":
            return int(value)
        if annotation_str == "float":
            return float(value)
        if annotation_str == "bool":
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if annotation_str == "NoneType":
            return None
        return value

    def _validate(self) -> None:
        if self.host not in _LOOPBACK_HOSTS:
            # Hard architecture rule: the Forge API is localhost-only by default.
            self.host = DEFAULT_HOST
        self.port = max(1, min(int(self.port), 65535))
        if self.data_dir is None:
            self.data_dir = str(default_data_dir())

    # ------------------------------------------------------------- resolved paths
    @property
    def data_dir_path(self) -> Path:
        return Path(self.data_dir or default_data_dir())

    @property
    def db_path_path(self) -> Path:
        return Path(self.db_path) if self.db_path else self.data_dir_path / "forge.db"

    @property
    def log_dir_path(self) -> Path:
        return self.data_dir_path / "logs"

    # ------------------------------------------------------------- persistence
    def write_self(self) -> None:
        path = default_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in {"workspace", "db_path", "data_dir"}
        }
        # merge: keep user's captured settings, refresh hardware-derived ones
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update({k: v for k, v in payload.items() if v is not None})
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def write_hardware_profile(profile: dict) -> Path:
    """Persist the detected hardware profile next to/over the config file."""
    path = default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing.update({k: v for k, v in profile.items() if v is not None})
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return path
