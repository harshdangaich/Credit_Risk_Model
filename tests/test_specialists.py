"""Unit tests verifying the 14 specialist agents in isolation."""
from __future__ import annotations

from credit_risk_model.agents.specialists import (
    ApiEngineerAgent,
    ChallengerAgent,
    DataQualityAgent,
    DevOpsAgent,
    ExperimentTrackingAgent,
    ExplainabilityAgent,
    FeatureEngineeringAgent,
    GovernanceAgent,
    ModelingAgent,
    ModelRiskReviewerAgent,
    MonitoringAgent,
    PipelineEngineer,
    QaTestAgent,
    ReportHandoffAgent,
    TargetLeakageAgent,
    ValidationAgent,
)


def test_stage1_agents(tmp_path):
    dq = DataQualityAgent()
    df, res_dq = dq.run("synthetic", seed=42)
    assert res_dq.status == "ok"
    assert len(df) == 5000
    assert "target" in df.columns

    tl = TargetLeakageAgent()
    res_tl = tl.run(df, "target", tmp_path)
    assert res_tl.status == "ok"
    assert "quality_summary.json" in res_tl.artifacts
    assert (tmp_path / "quality_summary.json").exists()


def test_stage2_agents(tmp_path):
    dq = DataQualityAgent()
    df, _ = dq.run("synthetic", seed=42)

    pe = PipelineEngineer()
    splits, res_pe = pe.run(df, "target", "application_date", 20.0)
    assert res_pe.status == "ok"
    assert len(splits.train) > 0
    assert len(splits.test) > 0

    fe = FeatureEngineeringAgent()
    feature_cols = [c for c in splits.train.columns if c not in ("target", "application_date")]
    specs, res_fe = fe.run(splits.train, feature_cols)
    assert res_fe.status == "ok"
    assert len(specs) > 0


def test_stage3_and_stage4_agents(tmp_path):
    dq = DataQualityAgent()
    df, _ = dq.run("synthetic", seed=42)
    pe = PipelineEngineer()
    splits, _ = pe.run(df, "target", "application_date", 20.0)

    # Stage 3
    mod = ModelingAgent()
    _model, res_mod = mod.run(splits.train, "target", 42, ("application_date",), tmp_path)
    assert res_mod.status == "ok"
    assert (tmp_path / "champion.joblib").exists()

    track = ExperimentTrackingAgent()
    res_track = track.run("credit_risk_champion", {"auroc": 0.77})
    assert res_track.status == "ok"

    # Stage 4
    chal = ChallengerAgent()
    _, res_chal = chal.run(splits.train, splits.test, "target", "histgb", 42, ("application_date",), tmp_path)
    assert res_chal.status == "ok"
    assert (tmp_path / "challenger.joblib").exists()


def test_stage5_and_stage6_agents(tmp_path):
    dq = DataQualityAgent()
    df, _ = dq.run("synthetic", seed=42)
    pe = PipelineEngineer()
    splits, _ = pe.run(df, "target", "application_date", 20.0)
    mod = ModelingAgent()
    model, _ = mod.run(splits.train, "target", 42, ("application_date",), tmp_path)

    # Stage 5
    val = ValidationAgent()
    y_test = splits.test["target"]
    p_test = model.predict_proba(splits.test.drop(columns=["target", "application_date"]))
    metrics, res_val = val.run(y_test, p_test)
    assert res_val.status == "ok"
    assert "auroc" in metrics

    mrr = ModelRiskReviewerAgent()
    p_train = model.predict_proba(splits.train.drop(columns=["target", "application_date"]))
    res_mrr = mrr.run(p_train, p_test, "run-test", "synthetic", metrics, tmp_path)
    assert res_mrr.status in ("ok", "warn")
    assert (tmp_path / "validation_report.md").exists()

    # Stage 6
    exp = ExplainabilityAgent()
    res_exp = exp.run(model)
    assert res_exp.status == "ok"

    gov = GovernanceAgent()
    res_gov = gov.run(model, "synthetic", metrics, tmp_path)
    assert res_gov.status == "ok"
    assert (tmp_path / "model_card.json").exists()


def test_stage7_and_stage8_agents(tmp_path):
    api = ApiEngineerAgent()
    res_api = api.run()
    assert res_api.status == "ok"

    qa = QaTestAgent()
    res_qa = qa.run(tmp_path)
    assert res_qa.status == "ok"
    assert (tmp_path / "test_report.json").exists()

    mon = MonitoringAgent()
    res_mon = mon.run(tmp_path)
    assert res_mon.status == "ok"
    assert (tmp_path / "drift_summary.json").exists()

    devops = DevOpsAgent()
    res_devops = devops.run()
    assert res_devops.status == "ok"

    handoff = ReportHandoffAgent()
    res_handoff = handoff.run("run-test", {"card": "model_card.json"})
    assert res_handoff.status == "ok"
