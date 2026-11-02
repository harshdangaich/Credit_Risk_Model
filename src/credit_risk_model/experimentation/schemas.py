"""Schemas for the experimentation platform."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class ExperimentObjective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "credit_risk_experiment"
    target_metric: str = "auroc"
    min_target_metric: float | None = 0.70
    max_brier_score: float | None = 0.10
    max_psi: float | None = 0.25
    dataset_ref: str | None = None
    dataset_rows: int = Field(default=20000, ge=100)
    dataset_seed: int = 42
    candidate_algorithms: list[str] = Field(default_factory=lambda: ["histgb", "random_forest"])
    max_experiments: int = Field(default=3, ge=1, le=10)
    time_budget_seconds: int = Field(default=300, ge=10)


class DataValidationReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rows: int
    columns: list[str]
    target_column: str
    default_rate: float
    missing_rate_by_feature: dict[str, float]
    constant_features: list[str]
    is_valid: bool
    issues: list[str] = Field(default_factory=list)


class PlannedIteration(BaseModel):
    model_config = ConfigDict(extra="ignore")

    iteration: int
    algorithm: str
    target_hyperparameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    workflow_id: str
    objective_name: str
    target_metric: str
    hypothesis: str
    execution_steps: list[str] = Field(default_factory=list)
    agent_responsibilities: dict[str, str] = Field(default_factory=dict)
    planned_iterations: list[PlannedIteration] = Field(default_factory=list)
    acceptance_criteria: dict[str, Any] = Field(default_factory=dict)
    experiment_limits: dict[str, Any] = Field(default_factory=dict)
    markdown_plan: str
    markdown_path: str | None = None


class TuningProposal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    experiment_id: str
    algorithm: str
    hyperparameters: dict[str, Any]
    rationale: str
    iteration: int


class ExperimentRun(BaseModel):
    model_config = ConfigDict(extra="ignore")

    experiment_id: str
    workflow_id: str
    iteration: int
    model_type: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    status: ExperimentStatus
    start_time_utc: str
    end_time_utc: str | None = None
    duration_seconds: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    error_message: str | None = None
    passed_criteria: bool = False
    criteria_feedback: dict[str, Any] = Field(default_factory=dict)


class WorkflowResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    workflow_id: str
    objective: ExperimentObjective
    status: WorkflowStatus
    created_at_utc: str
    completed_at_utc: str | None = None
    plan: ExperimentPlan | None = None
    plan_path: str | None = None
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    experiments: list[ExperimentRun] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    candidate_reasoning: str | None = None
    promotion_status: str = "not_promoted"
    promotion_details: dict[str, Any] | None = None
    error_message: str | None = None


class CandidatePromotionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved_by: str = Field(min_length=1)
    approval_notes: str | None = None
    target_status: str = "under_review"  # "active" or "under_review"
    promote_as_champion: bool = False  # If True, updates active champion in model inventory


class CandidatePromotionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    workflow_id: str
    experiment_id: str
    promoted_model_name: str
    promoted_model_version: str
    inventory_status: str
    approval_date: str
    approved_by: str
