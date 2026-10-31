"""
tests/test_predictions.py
MaintenIQ — integration tests for the Flask API and prediction pipeline.

Run with:
    python -m pytest tests/ -v
"""

import json
import sys
import pathlib

import pytest

# Ensure backend is importable regardless of cwd
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "backend"))

import numpy as np
from app import app as flask_app, _pipeline, _metadata, _extract_failure_probability, _risk_category  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


# ── Minimal valid payload ─────────────────────────────────────────────────────

VALID_LOW_RISK = {
    "Type": "L",
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 0,
}

VALID_HIGH_RISK = {
    "Type": "H",
    "Air temperature [K]": 304.5,
    "Process temperature [K]": 316.2,
    "Rotational speed [rpm]": 1200,
    "Torque [Nm]": 70.0,
    "Tool wear [min]": 220,
}


# ── Stage 1: Model loads correctly ────────────────────────────────────────────

class TestModelLoad:
    def test_pipeline_loaded(self):
        assert _pipeline is not None, "Pipeline should be loaded at startup"

    def test_metadata_loaded(self):
        assert _metadata is not None, "Metadata should be loaded at startup"

    def test_expected_model_name(self):
        assert _metadata["model_name"] == "Decision Tree"

    def test_threshold_value(self):
        assert _metadata["classification_threshold"] == 0.05

    def test_input_features_order(self):
        expected = [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
        assert _metadata["input_features"] == expected

    def test_pipeline_has_two_steps(self):
        step_names = [name for name, _ in _pipeline.steps]
        assert "preprocessing" in step_names
        assert "classifier" in step_names

    def test_pipeline_output_classes(self):
        classifier = _pipeline.named_steps["classifier"]
        assert list(classifier.classes_) == [0, 1]


# ── Stage 2: Direct pipeline predictions ─────────────────────────────────────

class TestPipelinePredictions:
    def _predict(self, payload):
        import pandas as pd
        df = pd.DataFrame([payload])
        return _extract_failure_probability(_pipeline, df)

    def test_low_risk_returns_probability(self):
        prob = self._predict(VALID_LOW_RISK)
        assert 0.0 <= prob <= 1.0

    def test_high_risk_higher_than_low(self):
        low_prob  = self._predict(VALID_LOW_RISK)
        high_prob = self._predict(VALID_HIGH_RISK)
        assert high_prob >= low_prob, (
            f"High-risk machine ({high_prob:.4f}) should have probability >= "
            f"low-risk machine ({low_prob:.4f})"
        )

    def test_all_machine_types(self):
        for t in ("L", "M", "H"):
            payload = {**VALID_LOW_RISK, "Type": t}
            prob = self._predict(payload)
            assert 0.0 <= prob <= 1.0, f"Unexpected probability for Type={t}: {prob}"

    def test_reproducibility(self):
        p1 = self._predict(VALID_LOW_RISK)
        p2 = self._predict(VALID_LOW_RISK)
        assert p1 == p2, "Same input must produce same output"

    def test_extract_failure_probability_uses_classes_attribute(self):
        import pandas as pd
        df = pd.DataFrame([VALID_LOW_RISK])
        # Verify the helper extracts from class 1 index according to model classes_
        classes = list(getattr(_pipeline, "classes_", _pipeline.named_steps["classifier"].classes_))
        expected_idx = classes.index(1)
        expected_prob = float(_pipeline.predict_proba(df)[0][expected_idx])
        actual_prob = _extract_failure_probability(_pipeline, df)
        assert actual_prob == expected_prob

    def test_extract_failure_probability_handles_inverted_classes(self):
        import pandas as pd
        import numpy as np

        class MockPipeline:
            classes_ = np.array([1, 0])  # Inverted order
            def predict_proba(self, X):
                return np.array([[0.35, 0.65]])

        mock_pipe = MockPipeline()
        dummy_df = pd.DataFrame([VALID_LOW_RISK])
        # Should pick index 0 because class 1 is at index 0
        prob = _extract_failure_probability(mock_pipe, dummy_df)
        assert prob == 0.35

    def test_risk_category_intermediate_probability_handling(self):
        # Verify that intermediate probability (e.g. 0.284) is preserved and threshold is separated
        cat = _risk_category(0.284, threshold=0.05)
        assert cat["failure_probability"] == 0.284
        assert cat["failure_probability_pct"] == 28.4
        assert cat["predicted_failure"] == 1  # 0.284 >= 0.05
        assert cat["level"] == "medium"
        assert cat["label"] == "Moderate Risk"

    def test_decision_tree_leaf_purity_explains_binary_probabilities(self):
        # Inspect trained decision tree leaves to confirm why uncalibrated DT returns 0.0 or 1.0
        clf = _pipeline.named_steps["classifier"]
        dt = clf.calibrated_classifiers_[0].estimator if hasattr(clf, "calibrated_classifiers_") else clf
        tree = dt.tree_
        is_leaf = tree.children_left == -1
        leaf_values = tree.value[is_leaf, 0, :]
        leaf_probs = leaf_values[:, 1] / leaf_values.sum(axis=1)
        # All leaves in the unconstrained tree are 100% pure (prob is 0.0 or 1.0)
        assert set(np.unique(leaf_probs)).issubset({0.0, 1.0})
        pure_0_leaves = (leaf_probs == 0.0).sum()
        pure_1_leaves = (leaf_probs == 1.0).sum()
        assert pure_0_leaves > 0 and pure_1_leaves > 0
        assert (pure_0_leaves + pure_1_leaves) == len(leaf_probs)

    def test_calibrated_pipeline_produces_intermediate_probabilities(self):
        # Verify that the calibrated pipeline produces intermediate probabilities for machines
        import pandas as pd
        from app import EXAMPLE_MACHINES
        probs = []
        for m in EXAMPLE_MACHINES:
            row = {k: v for k, v in m.items() if k != "id"}
            df = pd.DataFrame([row])
            p = _extract_failure_probability(_pipeline, df)
            probs.append(p)
        # Verify we have meaningful intermediate probabilities (not strictly 0.0 or 1.0)
        intermediate = [p for p in probs if 0.0 < p < 1.0]
        assert len(intermediate) > 0, f"Expected intermediate probabilities, got: {probs}"


# ── Stage 3: Health endpoint ──────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_status_ok(self, client):
        data = resp = client.get("/api/health").get_json()
        assert data["status"] == "ok"

    def test_health_includes_model_name(self, client):
        data = client.get("/api/health").get_json()
        assert data["model"] == "Decision Tree"

    def test_health_includes_threshold(self, client):
        data = client.get("/api/health").get_json()
        assert data["threshold"] == 0.05


# ── Stage 4: Prediction API endpoint ─────────────────────────────────────────

class TestPredictEndpoint:
    def _post(self, client, payload):
        return client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_valid_low_risk_returns_200(self, client):
        resp = self._post(client, VALID_LOW_RISK)
        assert resp.status_code == 200

    def test_valid_high_risk_returns_200(self, client):
        resp = self._post(client, VALID_HIGH_RISK)
        assert resp.status_code == 200

    def test_response_contains_probability(self, client):
        data = self._post(client, VALID_LOW_RISK).get_json()
        assert "failure_probability" in data
        assert 0.0 <= data["failure_probability"] <= 1.0

    def test_response_contains_predicted_failure(self, client):
        data = self._post(client, VALID_LOW_RISK).get_json()
        assert "predicted_failure" in data
        assert data["predicted_failure"] in (0, 1)

    def test_response_contains_risk_level(self, client):
        data = self._post(client, VALID_LOW_RISK).get_json()
        assert data["level"] in ("low", "medium", "high", "critical")

    def test_response_contains_explanation(self, client):
        data = self._post(client, VALID_LOW_RISK).get_json()
        assert "explanation" in data
        assert len(data["explanation"]) > 10

    def test_response_contains_disclaimer(self, client):
        data = self._post(client, VALID_LOW_RISK).get_json()
        assert "disclaimer" in data

    def test_threshold_key_present(self, client):
        data = self._post(client, VALID_LOW_RISK).get_json()
        assert "threshold_used" in data
        assert data["threshold_used"] == 0.05

    def test_high_risk_flagged(self, client):
        data = self._post(client, VALID_HIGH_RISK).get_json()
        # High-risk example should be predicted as failure
        assert data["predicted_failure"] == 1, (
            f"High-risk example should predict failure=1, "
            f"got failure_probability={data['failure_probability']}"
        )


# ── Stage 5: Input validation ─────────────────────────────────────────────────

class TestInputValidation:
    def _post(self, client, payload):
        return client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_missing_type_returns_422(self, client):
        payload = {**VALID_LOW_RISK}
        del payload["Type"]
        resp = self._post(client, payload)
        assert resp.status_code == 422

    def test_invalid_type_returns_422(self, client):
        payload = {**VALID_LOW_RISK, "Type": "X"}
        resp = self._post(client, payload)
        assert resp.status_code == 422

    def test_missing_feature_returns_422(self, client):
        payload = {**VALID_LOW_RISK}
        del payload["Torque [Nm]"]
        resp = self._post(client, payload)
        assert resp.status_code == 422

    def test_non_numeric_feature_returns_422(self, client):
        payload = {**VALID_LOW_RISK, "Air temperature [K]": "hot"}
        resp = self._post(client, payload)
        assert resp.status_code == 422

    def test_wrong_content_type_returns_415(self, client):
        resp = client.post("/api/predict", data="not-json", content_type="text/plain")
        assert resp.status_code == 415

    def test_out_of_range_returns_warning(self, client):
        payload = {**VALID_LOW_RISK, "Tool wear [min]": 350}  # > 300 warning threshold
        data = self._post(client, payload).get_json()
        assert data.get("warnings") and len(data["warnings"]) > 0

    def test_type_case_insensitive(self, client):
        payload = {**VALID_LOW_RISK, "Type": "m"}  # lowercase
        resp = self._post(client, payload)
        assert resp.status_code == 200


# ── Stage 6: Model-info endpoint ──────────────────────────────────────────────

class TestModelInfoEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/api/model-info")
        assert resp.status_code == 200

    def test_contains_metadata(self, client):
        data = client.get("/api/model-info").get_json()
        assert "metadata" in data

    def test_contains_model_comparison(self, client):
        data = client.get("/api/model-info").get_json()
        assert "model_comparison" in data
        assert data["model_comparison"] is not None, (
            "model_comparison should not be None — check reports/model_comparison.json exists"
        )

    def test_contains_feature_importance(self, client):
        data = client.get("/api/model-info").get_json()
        assert "feature_importance" in data
        assert data["feature_importance"] is not None, (
            "feature_importance should not be None — check reports/feature_importance.json exists"
        )

    def test_contains_example_machines(self, client):
        data = client.get("/api/model-info").get_json()
        assert "example_machines" in data
        assert len(data["example_machines"]) == 8


# ── Stage 7: Machines endpoint ────────────────────────────────────────────────

class TestMachinesEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/api/machines")
        assert resp.status_code == 200

    def test_returns_list(self, client):
        data = client.get("/api/machines").get_json()
        assert isinstance(data, list)
        assert len(data) == 8

    def test_sorted_by_probability_descending(self, client):
        machines = client.get("/api/machines").get_json()
        probs = [m["failure_probability"] for m in machines]
        assert probs == sorted(probs, reverse=True), (
            "Machines must be sorted by failure probability descending"
        )

    def test_all_machines_have_risk_level(self, client):
        machines = client.get("/api/machines").get_json()
        for m in machines:
            assert m["level"] in ("low", "medium", "high", "critical")


# ── Stage 8: Dashboard and predict page render ────────────────────────────────

class TestPageRender:
    def test_dashboard_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_dashboard_contains_title(self, client):
        resp = client.get("/")
        assert b"MaintenIQ" in resp.data

    def test_predict_page_returns_200(self, client):
        resp = client.get("/predict")
        assert resp.status_code == 200

    def test_predict_page_contains_form(self, client):
        resp = client.get("/predict")
        assert b"predict-form" in resp.data

    def test_live_page_returns_200(self, client):
        resp = client.get("/live")
        assert resp.status_code == 200
        assert b"SIMULATED DATA" in resp.data
        assert b"btn-stream-toggle" in resp.data

    def test_insights_page_returns_200(self, client):
        resp = client.get("/insights")
        assert resp.status_code == 200

    def test_404_returns_json(self, client):
        resp = client.get("/nonexistent-endpoint")
        assert resp.status_code == 404


# ── Stage 9: Diagnostics Model and Endpoints ─────────────────────────────────

class TestDiagnostics:
    def test_diagnostics_pipeline_loaded(self):
        from app import _diag_pipeline, _diag_metadata
        assert _diag_pipeline is not None
        assert _diag_metadata is not None

    def test_target_modes(self):
        from app import _diag_metadata
        expected_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
        assert _diag_metadata["target_modes"] == expected_modes

    def test_predict_endpoint_contains_diagnostics(self, client):
        resp = client.post("/api/predict", json=VALID_LOW_RISK)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "diagnostics" in data
        diag = data["diagnostics"]
        assert "predicted_modes" in diag
        assert "summary" in diag
        assert "disclaimer" in diag
        # Low risk input should identify no failure mode
        assert diag["summary"] == "No failure mode identified by the diagnostic model"

    def test_diagnose_endpoint(self, client):
        resp = client.post("/api/diagnose", json=VALID_HIGH_RISK)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "diagnostics" in data
        assert data["diagnostics"]["available"] is True

    def test_high_stress_detects_modes(self, client):
        stress_payload = {
            "Type": "H",
            "Air temperature [K]": 304.5,
            "Process temperature [K]": 310.2,
            "Rotational speed [rpm]": 1200,
            "Torque [Nm]": 70.0,
            "Tool wear [min]": 220,
        }
        resp = client.post("/api/predict", json=stress_payload)
        assert resp.status_code == 200
        diag = resp.get_json()["diagnostics"]
        assert len(diag["predicted_modes"]) > 0


# ── Stage 10: Explainability & Physics Metrics ────────────────────────────────

class TestExplainabilityAndPhysics:
    def test_local_explanation_structure(self, client):
        resp = client.post("/api/predict", json=VALID_LOW_RISK)
        data = resp.get_json()
        assert "local_explanation" in data
        xai = data["local_explanation"]
        assert xai["available"] is True
        assert "feature_contributions" in xai
        assert "summary" in xai
        assert "method" in xai

    def test_physical_metrics_structure(self, client):
        resp = client.post("/api/predict", json=VALID_LOW_RISK)
        data = resp.get_json()
        assert "physical_metrics" in data
        phys = data["physical_metrics"]
        assert "delta_t" in phys
        assert "power" in phys
        assert "overstrain" in phys
        assert "tool_wear_zone" in phys


# ── Stage 11: Simulator Telemetry Endpoint ───────────────────────────────────

class TestSimulatorAPI:
    def test_simulate_next_default(self, client):
        resp = client.get("/api/simulate/next")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["simulated"] is True
        assert "Air temperature [K]" in data
        assert "Torque [Nm]" in data

    def test_simulate_scenarios(self, client):
        for scenario in ["thermal_stress", "mechanical_overstrain", "power_surge", "tool_degradation", "normal"]:
            resp = client.post("/api/simulate/next", json={"scenario": scenario})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["scenario"] == scenario

    def test_simulate_deterministic_seed(self, client):
        r1 = client.post("/api/simulate/next", json={"scenario": "normal", "seed": 42}).get_json()
        r2 = client.post("/api/simulate/next", json={"scenario": "normal", "seed": 42}).get_json()
        assert r1["Air temperature [K]"] == r2["Air temperature [K]"]
        assert r1["Torque [Nm]"] == r2["Torque [Nm]"]

