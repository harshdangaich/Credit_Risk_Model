"""Planner Agent: LLM-driven experimentation planning and workflow architecture generation."""
from __future__ import annotations

import json
import logging
import os
import urllib.request
import uuid
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
    _env_path = Path(__file__).resolve().parents[3] / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass

from credit_risk_model.experimentation.schemas import (
    ExperimentObjective,
    ExperimentPlan,
    PlannedIteration,
)
from credit_risk_model.experimentation.store import ExperimentStore

logger = logging.getLogger("planner_agent")


class PlannerAgent:
    """Analyzes objectives, formulates hypotheses, generates experiment plans and Mermaid diagrams."""

    def __init__(self, store: ExperimentStore | None = None):
        self.store = store or ExperimentStore()

    def generate_plan(
        self,
        objective: ExperimentObjective,
        workflow_id: str | None = None,
    ) -> ExperimentPlan:
        """Generate a validated structured experiment plan and persist experiment_plan.md."""
        wf_id = workflow_id or f"wf-{uuid.uuid4().hex[:8]}"

        # Attempt LLM synthesis if API key is present; otherwise use expert domain synthesis.
        raw_plan = self._synthesize_with_llm(objective, wf_id) or self._synthesize_deterministic(objective, wf_id)

        # Generate Mermaid diagram & complete Markdown
        markdown_content = self._render_markdown_plan(raw_plan, objective)

        # Persist to artifacts/experiments/{workflow_id}/experiment_plan.md
        plan_path = self.store.save_plan(wf_id, markdown_content)

        # Build and validate structured plan
        plan = ExperimentPlan(
            workflow_id=wf_id,
            objective_name=objective.name,
            target_metric=objective.target_metric,
            hypothesis=raw_plan["hypothesis"],
            execution_steps=raw_plan["execution_steps"],
            agent_responsibilities=raw_plan["agent_responsibilities"],
            planned_iterations=[PlannedIteration(**it) for it in raw_plan["planned_iterations"]],
            acceptance_criteria=raw_plan["acceptance_criteria"],
            experiment_limits=raw_plan["experiment_limits"],
            markdown_plan=markdown_content,
            markdown_path=str(plan_path),
        )
        return plan

    def _synthesize_with_llm(self, objective: ExperimentObjective, workflow_id: str) -> dict[str, Any] | None:
        """Call LLM API (if configured) with structured output validation."""
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")

        if not gemini_key and not openai_key and not azure_key:
            return None

        prompt = (
            f"You are a Senior Lead Credit Risk Modeler and Experimentation Planner.\n"
            f"Formulate an experimentation plan for the following objective:\n"
            f"- Objective Name: {objective.name}\n"
            f"- Target Metric: {objective.target_metric} (min threshold: {objective.min_target_metric})\n"
            f"- Max Brier Score: {objective.max_brier_score}\n"
            f"- Max PSI: {objective.max_psi}\n"
            f"- Candidate Algorithms: {objective.candidate_algorithms}\n"
            f"- Max Experiments: {objective.max_experiments}\n"
            f"- Time Budget: {objective.time_budget_seconds}s\n\n"
            f"Respond ONLY with a JSON object containing keys:\n"
            f"- 'hypothesis' (str): credit risk hypothesis\n"
            f"- 'execution_steps' (list of str): execution steps\n"
            f"- 'agent_responsibilities' (dict of str: str): include keys 'Planner Agent', 'Orchestrator', 'Data Agent', 'Tuning Agent', 'Experiment Agent', 'Evaluation Agent'\n"
            f"- 'planned_iterations' (list of exactly {objective.max_experiments} dicts with iteration, algorithm, target_hyperparameters, rationale)."
        )

        try:
            if gemini_key:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
                data = json.dumps({
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseMimeType": "application/json"}
                }).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=12) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    text = res_json["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = json.loads(self._clean_json_text(text))
                    if self._validate_llm_payload(parsed, objective):
                        return self._complete_plan_dict(parsed, objective, workflow_id)

            elif openai_key:
                url = "https://api.openai.com/v1/chat/completions"
                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a Senior Lead Credit Risk Modeler and Experimentation Planner."},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {openai_key}",
                    },
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    text = res_json["choices"][0]["message"]["content"]
                    parsed = json.loads(self._clean_json_text(text))
                    if self._validate_llm_payload(parsed, objective):
                        return self._complete_plan_dict(parsed, objective, workflow_id)

            elif azure_key:
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_API_BASE")
                if not azure_endpoint:
                    logger.warning(
                        "AZURE_OPENAI_API_KEY found in .env, but AZURE_OPENAI_ENDPOINT is not configured. "
                        "Please add AZURE_OPENAI_ENDPOINT (and optionally AZURE_OPENAI_MODEL) to .env."
                    )
                    return None
                deployment = (
                    os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
                    or os.getenv("AZURE_OPENAI_MODEL")
                    or "gpt-4o"
                )
                api_version = (
                    os.getenv("AZURE_OPENAI_API_VERSION")
                    or os.getenv("AZURE_API_VERSION")
                    or "2024-02-15-preview"
                )
                url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
                payload = {
                    "messages": [
                        {"role": "system", "content": "You are a Senior Lead Credit Risk Modeler and Experimentation Planner."},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "api-key": azure_key,
                    },
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    text = res_json["choices"][0]["message"]["content"]
                    parsed = json.loads(self._clean_json_text(text))
                    if self._validate_llm_payload(parsed, objective):
                        return self._complete_plan_dict(parsed, objective, workflow_id)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            logger.warning("LLM generation unavailable or failed: %s. Using expert deterministic engine.", exc)

        return None

    def _clean_json_text(self, text: str) -> str:
        """Strip markdown fences (e.g. ```json ... ```) if present."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        return cleaned

    def _validate_llm_payload(self, data: dict, objective: ExperimentObjective) -> bool:
        """Validate that LLM payload does not hallucinate invalid algorithms or arbitrary code."""
        if not isinstance(data, dict):
            return False
        if not data.get("hypothesis") or not isinstance(data.get("planned_iterations"), list):
            return False
        for it in data["planned_iterations"]:
            if it.get("algorithm") not in objective.candidate_algorithms:
                return False
        return True

    def _complete_plan_dict(self, parsed: dict, objective: ExperimentObjective, workflow_id: str) -> dict[str, Any]:
        default_resp = {
            "Planner Agent": "Translates business objective into structured plan, hypothesis, and Mermaid flow.",
            "Orchestrator": "Coordinates agent execution, manages workflow state, and enforces bounded stopping rules.",
            "Data Agent": "Verifies feature contracts, missingness tolerances, and data split independence.",
            "Tuning Agent": "Proposes bounded, feedback-driven hyperparameter configurations without overengineering.",
            "Experiment Agent": "Executes training pipelines in isolated artifact directories with SHA-256 fingerprinting.",
            "Evaluation Agent": "Calculates discrimination, calibration, and stability metrics against baseline.",
            "Human Supervisor": "Reviews candidate metrics and grants explicit approval for governance registry promotion.",
        }
        resp = dict(default_resp)
        if isinstance(parsed.get("agent_responsibilities"), dict):
            resp.update(parsed["agent_responsibilities"])
            if "Planner Agent" not in resp:
                resp["Planner Agent"] = default_resp["Planner Agent"]
        parsed["agent_responsibilities"] = resp

        # Ensure planned_iterations has exact max_experiments count
        planned = parsed.get("planned_iterations", [])
        for it in planned:
            if not isinstance(it.get("target_hyperparameters"), dict):
                it["target_hyperparameters"] = {}
            if not it.get("rationale"):
                it["rationale"] = f"Evaluate {it.get('algorithm')} under bounded regularization."

        if len(planned) < objective.max_experiments:
            algorithms = objective.candidate_algorithms or ["histgb"]
            for i in range(len(planned) + 1, objective.max_experiments + 1):
                algo = algorithms[(i - 1) % len(algorithms)]
                planned.append({
                    "iteration": i,
                    "algorithm": algo,
                    "target_hyperparameters": (
                        {"learning_rate": 0.05, "max_depth": 4}
                        if algo == "histgb"
                        else {"n_estimators": 100, "max_depth": 5}
                    ),
                    "rationale": f"Iteration {i}: evaluate {algo} under bounded regularization.",
                })
        parsed["planned_iterations"] = planned[: objective.max_experiments]

        parsed["acceptance_criteria"] = {
            "target_metric": objective.target_metric,
            "min_target_metric": objective.min_target_metric,
            "max_brier_score": objective.max_brier_score,
            "max_psi": objective.max_psi,
        }
        parsed["experiment_limits"] = {
            "max_experiments": objective.max_experiments,
            "time_budget_seconds": objective.time_budget_seconds,
            "dataset_rows": objective.dataset_rows,
        }
        return parsed

    def _synthesize_deterministic(self, objective: ExperimentObjective, workflow_id: str) -> dict[str, Any]:
        """Deterministic domain-expert planning engine ensuring 100% offline reliability."""
        algorithms = [a for a in objective.candidate_algorithms if a in ("histgb", "random_forest")]
        if not algorithms:
            algorithms = ["histgb"]

        hypothesis = (
            f"The linear Logistic Regression baseline champion provides transparent, calibrated PDs, "
            f"but may underfit complex nonlinear feature interactions (e.g. debt-to-income vs. interest rate). "
            f"Iteratively evaluating tree-based ensembles ({', '.join(algorithms)}) under strict "
            f"regularization (bounded depth, min leaf samples) will test whether discrimination "
            f"({objective.target_metric}) can improve beyond {objective.min_target_metric} while preserving "
            f"calibration (Brier <= {objective.max_brier_score}) and population stability (PSI <= {objective.max_psi})."
        )

        steps = [
            "1. Plan Formulation: Planner Agent formalizes hypothesis, execution steps, and Mermaid architecture.",
            "2. Data Preparation: Data Agent inspects schema, missingness, and constructs stratified + OOT splits.",
            "3. Baseline Benchmark: Experiment Agent fits ChampionModel and Evaluation Agent computes baseline metrics.",
            "4. Iterative Tuning: Tuning Agent proposes bounded configurations for each candidate algorithm.",
            "5. Sandboxed Execution: Experiment Agent trains candidates into isolated run directories.",
            "6. Multi-Metric Evaluation: Evaluation Agent calculates AUROC, KS, Brier, PSI, and checks acceptance criteria.",
            "7. Bounded Feedback: Orchestrator applies evaluation feedback to refine subsequent iterations within budget.",
            "8. Candidate Selection: Orchestrator identifies best performing eligible candidate.",
            "9. Governance Gate: Human supervisor reviews evidence pack and authorizes inventory promotion."
        ]

        responsibilities = {
            "Planner Agent": "Translates business objective into structured plan, hypothesis, and Mermaid flow.",
            "Orchestrator": "Coordinates agent execution, manages workflow state, and enforces bounded stopping rules.",
            "Data Agent": "Verifies feature contracts, missingness tolerances, and data split independence.",
            "Tuning Agent": "Proposes bounded, feedback-driven hyperparameter configurations without overengineering.",
            "Experiment Agent": "Executes training pipelines in isolated artifact directories with SHA-256 fingerprinting.",
            "Evaluation Agent": "Calculates discrimination, calibration, and stability metrics against baseline.",
            "Human Supervisor": "Reviews candidate metrics and grants explicit approval for governance registry promotion."
        }

        iterations: list[dict[str, Any]] = []
        for i in range(1, objective.max_experiments + 1):
            algo = algorithms[(i - 1) % len(algorithms)]
            if algo == "histgb":
                params = {"learning_rate": 0.05, "max_depth": 3, "max_iter": 150, "min_samples_leaf": 50}
                rationale = "Evaluate gradient boosting with shallow depth (3) to prevent credit risk overfitting."
            else:
                params = {"n_estimators": 150, "min_samples_leaf": 20, "max_depth": 6}
                rationale = "Evaluate bagged random forest ensemble to capture nonlinear interactions with controlled variance."

            iterations.append({
                "iteration": i,
                "algorithm": algo,
                "target_hyperparameters": params,
                "rationale": rationale
            })

        return {
            "hypothesis": hypothesis,
            "execution_steps": steps,
            "agent_responsibilities": responsibilities,
            "planned_iterations": iterations,
            "acceptance_criteria": {
                "target_metric": objective.target_metric,
                "min_target_metric": objective.min_target_metric,
                "max_brier_score": objective.max_brier_score,
                "max_psi": objective.max_psi,
            },
            "experiment_limits": {
                "max_experiments": objective.max_experiments,
                "time_budget_seconds": objective.time_budget_seconds,
                "dataset_rows": objective.dataset_rows,
            }
        }

    def _render_markdown_plan(self, plan: dict[str, Any], objective: ExperimentObjective) -> str:
        """Render complete experiment_plan.md containing Mermaid diagram and structured sections."""
        lines = [
            f"# 📋 Experiment Plan: {objective.name}",
            "",
            f"- **Target Metric:** `{objective.target_metric}` (Min Acceptable: `{objective.min_target_metric}`)",
            f"- **Max Brier Score:** `{objective.max_brier_score}`",
            f"- **Max Population PSI:** `{objective.max_psi}`",
            f"- **Max Experiments:** `{objective.max_experiments}`",
            f"- **Time Budget:** `{objective.time_budget_seconds}s`",
            "",
            "## 1. Objective and Hypothesis",
            "",
            plan["hypothesis"],
            "",
            "## 2. Mermaid Workflow Diagram",
            "",
            "```mermaid",
            "flowchart TD",
            "    User([User Objective]) --> Planner[Planner Agent: LLM]",
            "    Planner --> PlanDoc[experiment_plan.md]",
            "    PlanDoc --> Orch[Existing Orchestrator]",
            "    Orch --> Data[Data Agent: Validate & Split]",
            "    Data --> Base[Baseline Training: Champion]",
            "    Base --> Loop{Bounded Feedback Loop}",
            "    Loop --> Tune[Tuning Agent: Propose Hyperparameters]",
            "    Tune --> Exp[Experiment Agent: Train Candidate]",
            "    Exp --> Eval[Evaluation Agent: Validate & Compare]",
            "    Eval --> Check{More Experiments Permitted & Needed?}",
            "    Check -- Yes --> Tune",
            "    Check -- No --> Select[Candidate Selection]",
            "    Select --> Super[Supervisor Approval: Governance Gate]",
            "```",
            "",
            "## 3. Proposed Execution Steps",
            "",
        ]

        for step in plan["execution_steps"]:
            lines.append(f"- {step}")

        lines.extend([
            "",
            "## 4. Agent Responsibilities",
            "",
            "| Agent | Core Responsibility |",
            "|---|---|",
        ])
        for agent, resp in plan["agent_responsibilities"].items():
            lines.append(f"| **{agent}** | {resp} |")

        lines.extend([
            "",
            "## 5. Planned Candidate Iterations",
            "",
            "| Iteration | Algorithm | Initial Parameters | Strategy Rationale |",
            "|---|---|---|---|",
        ])
        for it in plan["planned_iterations"]:
            params_str = ", ".join(f"`{k}={v}`" for k, v in it["target_hyperparameters"].items())
            lines.append(f"| {it['iteration']} | `{it['algorithm']}` | {params_str} | {it['rationale']} |")

        lines.extend([
            "",
            "## 6. Evaluation Metrics & Acceptance Criteria",
            "",
            f"- **Primary Metric:** `{plan['acceptance_criteria']['target_metric']}` >= `{plan['acceptance_criteria']['min_target_metric']}`",
            f"- **Calibration Bound:** Brier Score <= `{plan['acceptance_criteria']['max_brier_score']}`",
            f"- **Stability Bound:** Population Stability Index (PSI) <= `{plan['acceptance_criteria']['max_psi']}`",
            "- **Benchmark Hurdle:** Must outperform held-out baseline champion metric without severe calibration distortion.",
            "",
            "## 7. Experiment Limits & Governance Requirements",
            "",
            (
                f"- **Stopping Criteria:** Workflow stops when all {objective.max_experiments} iterations complete, "
                f"the {objective.time_budget_seconds}s execution budget expires, or data validation fails."
            ),
            "- **Artifact Sandboxing:** All runs are isolated under `artifacts/experiments/{workflow_id}/`.",
            (
                "- **Governance Gate:** Automated promotion is prohibited. Candidate promotion to `model_inventory.yaml` "
                "strictly requires explicit authorization from the human supervisor."
            ),
            ""
        ])

        return "\n".join(lines)
