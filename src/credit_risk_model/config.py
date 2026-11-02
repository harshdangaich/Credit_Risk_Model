"""Centralised configuration loaded from environment variables (and an optional .env file).

Relative paths are resolved against ``PROJECT_ROOT`` when that variable is set, otherwise
against the current working directory at the time settings are first read. Nothing is
created on disk when settings are loaded; call :meth:`Settings.ensure_dirs` explicitly
from a CLI, pipeline or application start-up.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PATH_FIELDS = ("governance_dir", "inventory_path", "audit_trail_dir", "pma_register_path",
               "artifacts_dir", "reports_dir", "model_path", "experiments_dir")


def project_root() -> Path:
    """Directory that relative paths are resolved against."""
    return Path(os.getenv("PROJECT_ROOT") or Path.cwd()).resolve()


def resolve_path(path: str | Path) -> Path:
    """Resolve a user-supplied path against the project root unless it is absolute."""
    path = Path(path)
    return path if path.is_absolute() else project_root() / path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        extra="ignore", case_sensitive=False,
    )

    governance_dir: Path = Path("governance")
    inventory_path: Path = Path("governance/model_inventory.yaml")
    audit_trail_dir: Path = Path("governance/audit_trail")
    pma_register_path: Path = Path("governance/pma_log.csv")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    experiments_dir: Path = Path("artifacts/experiments")
    # Champion artefact. Empty means "use the inventory artifact_path".
    model_path: Path | None = None
    # Optional SHA-256 of the champion artefact; loading is refused on mismatch.
    model_sha256: str | None = None

    champion_name: str = "credit_risk_champion"
    allow_unreviewed_model: bool = False
    environment: str = "local"
    log_level: str = "INFO"

    # Monitoring thresholds: this is the single authoritative source.
    psi_investigate: float = 0.10
    psi_escalate: float = 0.25
    csi_investigate: float = 0.10
    csi_escalate: float = 0.25
    auroc_drop_investigate: float = 0.03
    auroc_drop_escalate: float = 0.05

    # Audit trail.
    audit_retention_days: int = Field(default=90, ge=1)
    audit_max_page_size: int = Field(default=500, ge=1)
    audit_fail_closed: bool = True
    audit_api_token: str | None = None
    predict_api_token: str | None = None
    max_request_bytes: int = Field(default=16_384, ge=256)

    mlflow_tracking_uri: str | None = None

    @model_validator(mode="after")
    def _resolve_paths(self):
        root = project_root()
        for name in PATH_FIELDS:
            value = getattr(self, name)
            if value is not None and str(value) and not value.is_absolute():
                setattr(self, name, root / value)
        return self

    @property
    def changes_dir(self) -> Path:
        return self.audit_trail_dir / "changes"

    def ensure_dirs(self) -> None:
        for path in (
            self.governance_dir,
            self.audit_trail_dir / "decisions",
            self.audit_trail_dir / "changes",
            self.audit_trail_dir / "monitoring",
            self.artifacts_dir,
            self.reports_dir,
            self.experiments_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
