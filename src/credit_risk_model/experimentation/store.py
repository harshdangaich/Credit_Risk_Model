"""Lightweight, isolated persistence for experiment workflows and artifacts."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import joblib

from credit_risk_model.config import get_settings
from credit_risk_model.experimentation.schemas import WorkflowResult
from credit_risk_model.governance.inventory import file_sha256

_LOCK = threading.Lock()
logger = logging.getLogger("experiment_store")


class ExperimentStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else get_settings().experiments_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def _workflow_dir(self, workflow_id: str) -> Path:
        p = self.root / workflow_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _experiment_dir(self, workflow_id: str, experiment_id: str) -> Path:
        p = self._workflow_dir(workflow_id) / experiment_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_plan(self, workflow_id: str, markdown_content: str) -> Path:
        with _LOCK:
            wf_dir = self._workflow_dir(workflow_id)
            plan_file = wf_dir / "experiment_plan.md"
            plan_file.write_text(markdown_content, encoding="utf-8")
            return plan_file

    def get_plan_path(self, workflow_id: str) -> Path:
        return self._workflow_dir(workflow_id) / "experiment_plan.md"

    def save_workflow(self, workflow: WorkflowResult) -> Path:
        with _LOCK:
            wf_dir = self._workflow_dir(workflow.workflow_id)
            wf_file = wf_dir / "workflow.json"
            content = workflow.model_dump_json(indent=2)
            wf_file.write_text(content, encoding="utf-8")
            return wf_file

    def get_workflow(self, workflow_id: str) -> WorkflowResult | None:
        with _LOCK:
            wf_file = self.root / workflow_id / "workflow.json"
            if not wf_file.exists():
                return None
            data = json.loads(wf_file.read_text(encoding="utf-8"))
            return WorkflowResult.model_validate(data)

    def list_workflows(self, limit: int = 50) -> list[WorkflowResult]:
        with _LOCK:
            results = []
            if not self.root.exists():
                return []
            for path in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if path.is_dir():
                    wf_file = path / "workflow.json"
                    if wf_file.exists():
                        try:
                            data = json.loads(wf_file.read_text(encoding="utf-8"))
                            results.append(WorkflowResult.model_validate(data))
                        except (json.JSONDecodeError, ValueError, OSError) as exc:
                            logger.debug("Failed to read workflow %s: %s", path, exc)
                            continue
                if len(results) >= limit:
                    break
            return results

    def save_artifact(self, workflow_id: str, experiment_id: str, model: Any) -> tuple[str, str]:
        """Save a model artifact in an isolated experiment directory. Returns (path, sha256)."""
        with _LOCK:
            exp_dir = self._experiment_dir(workflow_id, experiment_id)
            artifact_file = exp_dir / "model.joblib"
            joblib.dump(model, artifact_file)
            digest = file_sha256(artifact_file)
            return str(artifact_file), digest

    def load_artifact(self, artifact_path: str | Path) -> Any:
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        return joblib.load(path)
