"""Configuration layer: YAML file + environment overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def _load_yaml_config() -> dict[str, Any]:
    yaml_path = CONFIG_DIR / "borg.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml = _load_yaml_config()


def _yaml_get(path: str, default: Any) -> Any:
    keys = path.split(".")
    node: Any = _yaml
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


class Settings(BaseSettings):
    """Runtime settings. Env vars override YAML defaults."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core
    mission: str = Field(default=_yaml_get("mission", ""))
    database_url: str = Field(
        default=os.environ.get(
            "DATABASE_URL",
            "postgresql+psycopg://borg:borg@localhost:5432/borg",
        )
    )
    log_level: str = Field(default="INFO")

    # Loop
    loop_interval_seconds: int = Field(default=_yaml_get("loop.interval_seconds", 120))
    loop_min_interval_seconds: int = Field(default=_yaml_get("loop.min_interval_seconds", 30))
    loop_max_interval_seconds: int = Field(default=_yaml_get("loop.max_interval_seconds", 300))
    brain_interval_seconds: int = Field(default=_yaml_get("loop.brain_interval_seconds", _yaml_get("loop.interval_seconds", 120)))
    brain_concurrent_symbols: int = Field(default=_yaml_get("loop.brain_concurrent_symbols", 5))
    self_improve_interval_seconds: int = Field(default=_yaml_get("loop.self_improve_interval_seconds", 86400))
    consciousness_interval_seconds: int = Field(default=_yaml_get("loop.consciousness_interval_seconds", 120))
    consciousness_timeout_seconds: float = Field(default=_yaml_get("loop.consciousness_timeout_seconds", 90.0))

    # Markets
    symbol_list: list[str] = Field(default_factory=lambda: _yaml_get("symbols", ["EURUSD", "GBPUSD", "USDJPY"]))
    confidence_threshold: float = Field(default=_yaml_get("trading.confidence_threshold", 65.0))
    forecast_horizon_seconds: int = Field(default=_yaml_get("trading.forecast_horizon_seconds", 300))

    # LLM
    llm_provider: str = Field(default=_yaml_get("llm.provider", "ollama"))
    llm_model: str = Field(default=_yaml_get("llm.model", "tinyllama:latest"))
    llm_embed_model: str = Field(default=_yaml_get("llm.embed_model", "nomic-embed-text:latest"))
    llm_base_url: str = Field(default=_yaml_get("llm.base_url", "http://localhost:11434"))
    llm_max_tokens: int = Field(default=_yaml_get("llm.max_tokens", 512))
    llm_temperature: float = Field(default=_yaml_get("llm.temperature", 0.2))
    llm_timeout_seconds: float = Field(default=_yaml_get("llm.timeout_seconds", 45.0))
    llm_force_fallback: bool = Field(default=_yaml_get("llm.force_fallback", False))
    llm_fallback_provider: str = Field(default=_yaml_get("llm.fallback.provider", ""))
    llm_fallback_base_url: str = Field(default=_yaml_get("llm.fallback.base_url", ""))
    llm_fallback_api_key_env: str = Field(default=_yaml_get("llm.fallback.api_key_env", "OPENAI_API_KEY"))
    llm_fallback_model: str = Field(default=_yaml_get("llm.fallback.model", "gpt-4o-mini"))

    # Image generation (Pollinations)
    pollinations_enabled: bool = Field(default=_yaml_get("pollinations.enabled", True))
    pollinations_api_key_env: str = Field(default=_yaml_get("pollinations.api_key_env", "POLLINATIONS_API_KEY"))
    pollinations_api_key: Optional[str] = Field(
        default=None,
        validation_alias=_yaml_get("pollinations.api_key_env", "POLLINATIONS_API_KEY"),
    )
    pollinations_base_url: str = Field(default=_yaml_get("pollinations.base_url", "https://gen.pollinations.ai"))
    pollinations_timeout_seconds: float = Field(default=_yaml_get("pollinations.timeout_seconds", 60.0))
    pollinations_default_model: str = Field(default=_yaml_get("pollinations.default_model", "flux"))
    pollinations_image_dir: Path = Field(default=Path(_yaml_get("pollinations.image_dir", "./output/images")))

    # RTSP cameras (list-based; credentials via env vars)
    cameras: list[dict[str, Any]] = Field(
        default_factory=lambda: _yaml_get(
            "cameras",
            [
                {
                    "name": "default",
                    "enabled": _yaml_get("camera.enabled", False),
                    "host": _yaml_get("camera.host", "10.0.0.91"),
                    "rtsp_port": _yaml_get("camera.rtsp_port", 554),
                    "rtsp_path": _yaml_get("camera.rtsp_path", "/stream1"),
                    "stream_quality": _yaml_get("camera.stream_quality", 75),
                    "reconnect_seconds": _yaml_get("camera.reconnect_seconds", 5.0),
                }
            ],
        )
    )
    camera_stream_port: int = Field(default=_yaml_get("camera.stream_port", 8080))
    camera_stream_quality: int = Field(default=_yaml_get("camera.stream_quality", 75), ge=1, le=100)
    camera_reconnect_seconds: float = Field(default=_yaml_get("camera.reconnect_seconds", 5.0))

    # Resources
    ram_soft_limit_pct: float = Field(default=_yaml_get("resources.ram_soft_limit_pct", 80.0))
    cpu_soft_limit_pct: float = Field(default=_yaml_get("resources.cpu_soft_limit_pct", 85.0))
    memory_limit_mb: int = Field(default=_yaml_get("resources.memory_limit_mb", 4096))

    # Databricks external dashboard export
    databricks_enabled: bool = Field(default=_yaml_get("databricks.enabled", False))
    databricks_host: Optional[str] = Field(default=None, validation_alias=_yaml_get("databricks.host_env", "DATABRICKS_HOST"))
    databricks_token: Optional[str] = Field(default=None, validation_alias=_yaml_get("databricks.token_env", "DATABRICKS_TOKEN"))
    databricks_warehouse_id: Optional[str] = Field(default=None, validation_alias=_yaml_get("databricks.warehouse_id_env", "DATABRICKS_WAREHOUSE_ID"))
    databricks_catalog: str = Field(default=_yaml_get("databricks.catalog", "borg"))
    databricks_schema: str = Field(default=_yaml_get("databricks.schema", "public"))
    databricks_tables: dict[str, str] = Field(
        default_factory=lambda: _yaml_get(
            "databricks.tables",
            {"reports": "borg_reports", "forecasts": "borg_forecasts", "hip4_predictions": "borg_hip4_predictions", "paper_trades": "borg_paper_trades", "candles": "borg_candles"},
        )
    )

    # HyperLong local dashboard integration
    hyperlong_base_url: Optional[str] = Field(default=_yaml_get("hyperlong.base_url", "http://localhost:8080"))
    hyperlong_timeout_seconds: float = Field(default=_yaml_get("hyperlong.timeout_seconds", 15.0))

    # SMB inventory
    smb_host: str = Field(default=_yaml_get("smb.host", ""), validation_alias="BORG_SMB_HOST")
    smb_share: str = Field(default=_yaml_get("smb.share", "projects"), validation_alias="BORG_SMB_SHARE")
    smb_username: str = Field(default=_yaml_get("smb.username", "theone"), validation_alias="BORG_SMB_USERNAME")
    smb_password: str = Field(default=_yaml_get("smb.password", ""), validation_alias="BORG_SMB_PASSWORD")
    smb_domain: str = Field(default=_yaml_get("smb.domain", ""), validation_alias="BORG_SMB_DOMAIN")
    smb_code_extensions: list[str] = Field(
        default_factory=lambda: _yaml_get(
            "smb.code_extensions",
            [".py", ".js", ".ts", ".sql", ".yaml", ".yml", ".json", ".md", ".txt", ".sh"],
        ),
        validation_alias="BORG_SMB_CODE_EXTENSIONS",
    )
    smb_max_file_size_bytes: int = Field(
        default=_yaml_get("smb.max_file_size_bytes", 1_000_000),
        validation_alias="BORG_SMB_MAX_FILE_SIZE_BYTES",
    )
    smb_skip_patterns: list[str] = Field(
        default_factory=lambda: _yaml_get(
            "smb.skip_patterns",
            ["node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build"],
        ),
        validation_alias="BORG_SMB_SKIP_PATTERNS",
    )

    # Paths
    input_path: Path = Field(default=Path(_yaml_get("paths.input", "./input")))
    output_path: Path = Field(default=Path(_yaml_get("paths.output", "./output")))

    # Daily diary
    diary_enabled: bool = Field(default=_yaml_get("diary.enabled", True))
    diary_output_path: Path = Field(default=Path(_yaml_get("diary.output_path", "./output/diary")))
    diary_hour: int = Field(default=_yaml_get("diary.hour", 23), ge=0, le=23)
    diary_minute: int = Field(default=_yaml_get("diary.minute", 59), ge=0, le=59)

    # Web
    web_host: str = Field(default=_yaml_get("web.host", "0.0.0.0"))
    web_port: int = Field(default=_yaml_get("web.port", 8000))
    borg_password: Optional[str] = Field(default=os.environ.get("BORG_PASSWORD"))

    # Safety
    require_confirmation_for: list[str] = Field(
        default_factory=lambda: _yaml_get("safety.require_confirmation_for", ["self_modify", "resource_heavy", "clone", "delete", "assimilate"])
    )

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")

    @property
    def sqlite_path(self) -> Optional[Path]:
        if not self.is_postgres:
            prefix = "sqlite:///"
            if self.database_url.startswith(prefix):
                return Path(self.database_url[len(prefix) :])
        return None


settings = Settings()
