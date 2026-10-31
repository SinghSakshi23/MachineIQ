"""
MaintenIQ - Predictive Maintenance and Machine Failure Intelligence
Flask application entry point.

Loads:
1. Trained binary Decision Tree pipeline (mainteniq_pipeline.joblib).
2. Trained multi-label diagnostic Random Forest pipeline (mainteniq_diagnostics_pipeline.joblib).
3. Permutation feature importance and evaluation metrics reports.

All preprocessing is encapsulated in the scikit-learn pipelines.
"""

import json
import math
import os
import pathlib
import time
import random

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Paths - resolved relative to this file so the app works from any cwd
# ---------------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent   # project root
MODEL_PATH = BASE_DIR / "model" / "mainteniq_pipeline.joblib"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.json"
DIAG_MODEL_PATH = BASE_DIR / "model" / "mainteniq_diagnostics_pipeline.joblib"
DIAG_METADATA_PATH = BASE_DIR / "model" / "diagnostics_metadata.json"
REPORTS_DIR = BASE_DIR / "reports"

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "frontend" / "templates"),
    static_folder=str(BASE_DIR / "frontend" / "static"),
)

# ---------------------------------------------------------------------------
# Load model artefacts once at startup
# ---------------------------------------------------------------------------
_pipeline = None
_metadata = None
_diag_pipeline = None
_diag_metadata = None
_model_comparison = None
_feature_importance = None
_diagnostics_evaluation = None
_permutation_importance = None
_startup_error = None
_load_time = None


def _load_artefacts():
    global _pipeline, _metadata, _diag_pipeline, _diag_metadata
    global _model_comparison, _feature_importance, _diagnostics_evaluation
    global _permutation_importance, _startup_error, _load_time

    t0 = time.time()
    try:
        # 1. Primary binary pipeline & metadata
        _pipeline = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, encoding="utf-8") as fh:
            _metadata = json.load(fh)

        # 2. Multi-label diagnostics pipeline & metadata
        if DIAG_MODEL_PATH.exists():
            _diag_pipeline = joblib.load(DIAG_MODEL_PATH)
        if DIAG_METADATA_PATH.exists():
            with open(DIAG_METADATA_PATH, encoding="utf-8") as fh:
                _diag_metadata = json.load(fh)

        # 3. Model comparison report
        comparison_path = REPORTS_DIR / "model_comparison.json"
        if comparison_path.exists():
            with open(comparison_path, encoding="utf-8") as fh:
                _model_comparison = json.load(fh)

        # 4. Feature importance report
        importance_path = REPORTS_DIR / "feature_importance.json"
        if importance_path.exists():
            with open(importance_path, encoding="utf-8") as fh:
                _feature_importance = json.load(fh)

        # 5. Diagnostics evaluation report
        diag_eval_path = REPORTS_DIR / "diagnostics_evaluation.json"
        if diag_eval_path.exists():
            with open(diag_eval_path, encoding="utf-8") as fh:
                _diagnostics_evaluation = json.load(fh)

        # 6. Permutation importance report
        perm_path = REPORTS_DIR / "permutation_importance.json"
        if perm_path.exists():
            with open(perm_path, encoding="utf-8") as fh:
                _permutation_importance = json.load(fh)

        _load_time = round(time.time() - t0, 3)
    except Exception as exc:
        _startup_error = str(exc)
        raise


_load_artefacts()

# ---------------------------------------------------------------------------
# Input validation helpers and reference limits
# ---------------------------------------------------------------------------

VALID_TYPES = {"L", "M", "H"}

WARNING_RANGES = {
    "Air temperature [K]":      (290.0, 310.0),
    "Process temperature [K]":  (300.0, 325.0),
    "Rotational speed [rpm]":   (900.0, 3000.0),
    "Torque [Nm]":               (0.0, 100.0),
    "Tool wear [min]":           (0.0, 300.0),
}

# Documented benchmark limits from the AI4I 2020 dataset specification (Matanovic et al.)
REFERENCE_LIMITS = {
    "Air temperature [K]": {
        "dataset_min": 295.3,
        "dataset_max": 304.5,
        "typical_low": 297.0,
        "typical_high": 303.0,
        "unit": "K",
        "description": "Ambient factory air temperature surrounding the machine.",
        "source": "AI4I 2020 Dataset Documentation (Matanovic et al.)",
    },
    "Process temperature [K]": {
        "dataset_min": 305.7,
        "dataset_max": 313.8,
        "typical_low": 307.0,
        "typical_high": 312.5,
        "unit": "K",
        "description": "Process chamber operating temperature.",
        "source": "AI4I 2020 Dataset Documentation (Matanovic et al.)",
    },
    "Rotational speed [rpm]": {
        "dataset_min": 1168.0,
        "dataset_max": 2886.0,
        "typical_low": 1300.0,
        "typical_high": 1800.0,
        "unit": "rpm",
        "description": "Spindle rotational velocity.",
        "source": "AI4I 2020 Dataset Documentation (Matanovic et al.)",
    },
    "Torque [Nm]": {
        "dataset_min": 3.8,
        "dataset_max": 76.6,
        "typical_low": 25.0,
        "typical_high": 55.0,
        "unit": "Nm",
        "description": "Resistance torque delivered during cutting/milling.",
        "source": "AI4I 2020 Dataset Documentation (Matanovic et al.)",
    },
    "Tool wear [min]": {
        "dataset_min": 0.0,
        "dataset_max": 253.0,
        "typical_low": 0.0,
        "typical_high": 180.0,
        "unit": "min",
        "description": "Accumulated cutting time for the current milling insert.",
        "source": "AI4I 2020 Dataset Documentation (Matanovic et al.)",
    },
}


def _validate_and_parse(data: dict):
    """
    Validate incoming JSON payload.
    Returns (parsed_df, warnings_list, error_str).
    If error_str is not None the request is invalid.
    """
    errors = []
    warnings = []

    # --- Type ---
    raw_type = data.get("Type", "")
    machine_type = None
    if not raw_type:
        errors.append("'Type' is required (L = Low, M = Medium, H = High).")
    else:
        machine_type = str(raw_type).strip().upper()
        if machine_type not in VALID_TYPES:
            errors.append(
                f"'Type' must be one of: L, M, H. Received: {raw_type!r}"
            )

    # --- Numeric features ---
    numeric_values = {}
    numeric_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    for feat in numeric_features:
        raw = data.get(feat)
        if raw is None or str(raw).strip() == "":
            errors.append(f"'{feat}' is required.")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"'{feat}' must be a number. Received: {raw!r}")
            continue

        lo, hi = WARNING_RANGES[feat]
        if not (lo <= value <= hi):
            warnings.append(
                f"'{feat}' value {value} is outside the expected range "
                f"[{lo}, {hi}]. The model was trained on data within this range."
            )

        numeric_values[feat] = value

    if errors:
        return None, warnings, "; ".join(errors)

    row = {"Type": machine_type, **numeric_values}
    df = pd.DataFrame([row])
    return df, warnings, None


# ---------------------------------------------------------------------------
# Risk categorisation & Probability Extraction
# ---------------------------------------------------------------------------

def _extract_failure_probability(pipeline, input_df: pd.DataFrame) -> float:
    """
    Retrieve probability corresponding to class 1 (machine failure),
    using the model's actual class ordering rather than assuming column index 1.
    """
    if hasattr(pipeline, "classes_"):
        classes = list(pipeline.classes_)
    elif hasattr(pipeline, "named_steps") and "classifier" in pipeline.named_steps:
        classes = list(pipeline.named_steps["classifier"].classes_)
    else:
        classes = [0, 1]
    class_1_idx = classes.index(1) if 1 in classes else 1
    proba_matrix = pipeline.predict_proba(input_df)
    return float(proba_matrix[0][class_1_idx])


def _risk_category(probability: float, threshold: float) -> dict:
    """
    Return risk level, label and a plain-language explanation.
    Explanation is derived from the probability value only.
    """
    pct = round(probability * 100, 1)
    predicted_failure = int(probability >= threshold)

    if probability < 0.10:
        level = "low"
        label = "Low Risk"
        colour = "#16a34a"
        explanation = (
            f"The model assigns a {pct}% estimated failure probability. "
            "Operating conditions fall within ranges associated with low failure rates "
            "in the dataset."
        )
    elif probability < 0.35:
        level = "medium"
        label = "Moderate Risk"
        colour = "#d97706"
        explanation = (
            f"The model assigns a {pct}% estimated failure probability. "
            "One or more operating conditions are approaching ranges associated with "
            "higher failure rates in the dataset. Consider scheduling a "
            "preventive inspection."
        )
    elif probability < 0.65:
        level = "high"
        label = "High Risk"
        colour = "#dc2626"
        explanation = (
            f"The model assigns a {pct}% estimated failure probability. "
            "Operating conditions closely resemble patterns found in failure cases "
            "within the dataset. Preventive maintenance is advisable."
        )
    else:
        level = "critical"
        label = "Critical Risk"
        colour = "#7c3aed"
        explanation = (
            f"The model assigns a {pct}% estimated failure probability. "
            "Operating conditions are very similar to confirmed failure cases in the "
            "dataset. Immediate inspection is strongly recommended."
        )

    return {
        "level": level,
        "label": label,
        "colour": colour,
        "explanation": explanation,
        "predicted_failure": predicted_failure,
        "failure_probability": round(probability, 4),
        "failure_probability_pct": pct,
        "threshold_used": threshold,
        "disclaimer": (
            "This prediction is an estimate produced by a model trained on historical machine "
            "examples in the dataset. It is intended for educational demonstration only "
            "and is not a guarantee of machine safety."
        ),
    }


# ---------------------------------------------------------------------------
# Explainable AI: Local Tree Decision Path Attribution & Physical Metrics
# ---------------------------------------------------------------------------

def _explain_prediction(input_df: pd.DataFrame, probability: float) -> dict:
    """
    Calculate local feature attribution using the exact decision path of the
    DecisionTreeClassifier on the transformed input features (Tree Path Contribution).
    Supports both single decision trees and calibrated ensemble fold trees.
    """
    if _pipeline is None:
        return {"available": False, "reason": "Pipeline not loaded"}

    try:
        ct = _pipeline.named_steps["preprocessing"]
        clf = _pipeline.named_steps["classifier"]
        feature_names = list(ct.get_feature_names_out())

        X_trans = ct.transform(input_df)

        if hasattr(clf, "calibrated_classifiers_"):
            trees = [cc.estimator for cc in clf.calibrated_classifiers_]
        elif hasattr(clf, "estimator") and hasattr(clf.estimator, "tree_"):
            trees = [clf.estimator]
        else:
            trees = [clf]

        classes = list(getattr(_pipeline, "classes_", getattr(trees[0], "classes_", [0, 1])))
        class_1_idx = classes.index(1) if 1 in classes else 1

        all_contributions = {feat: 0.0 for feat in feature_names}
        base_root_probs = []

        for dt in trees:
            tree = dt.tree_
            base_prob = float(tree.value[0][0, class_1_idx] / tree.value[0][0].sum())
            base_root_probs.append(base_prob)

            node_indicator = dt.decision_path(X_trans)
            node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

            for i in range(len(node_index) - 1):
                curr_node = node_index[i]
                next_node = node_index[i + 1]
                split_feat_idx = tree.feature[curr_node]
                feat_name = feature_names[split_feat_idx]

                curr_prob = float(tree.value[curr_node][0, class_1_idx] / tree.value[curr_node][0].sum())
                next_prob = float(tree.value[next_node][0, class_1_idx] / tree.value[next_node][0].sum())
                all_contributions[feat_name] += (next_prob - curr_prob) / len(trees)

        base_root_prob = float(np.mean(base_root_probs))

        # Aggregate one-hot back to base features
        feature_contributions = {}
        for k, v in all_contributions.items():
            base_k = (
                k.replace("numeric__", "")
                .replace("categorical__Type_H", "Type")
                .replace("categorical__Type_L", "Type")
                .replace("categorical__Type_M", "Type")
            )
            feature_contributions[base_k] = feature_contributions.get(base_k, 0.0) + float(v)

        # Clean round
        feature_contributions = {k: round(v, 4) for k, v in feature_contributions.items()}

        # Identify positive risk drivers and protective factors
        risk_drivers = [
            {"feature": k, "impact": v}
            for k, v in feature_contributions.items() if v > 0.01
        ]
        risk_drivers.sort(key=lambda x: x["impact"], reverse=True)

        stabilizing_factors = [
            {"feature": k, "impact": v}
            for k, v in feature_contributions.items() if v < -0.01
        ]
        stabilizing_factors.sort(key=lambda x: x["impact"])

        # Plain-English local summary
        if risk_drivers:
            top_names = [item["feature"] for item in risk_drivers[:2]]
            local_summary = (
                f"The highest upward influence on risk for this reading comes from "
                f"{', '.join(top_names)}."
            )
        elif probability < 0.10:
            local_summary = "Operating parameters followed decision paths in the model that lead to low failure estimates."
        else:
            local_summary = "Risk influence is evenly distributed across monitored operating parameters."

        return {
            "available": True,
            "method": "Exact Tree Decision Path Attribution",
            "base_root_probability": round(base_root_prob, 4),
            "predicted_probability": round(probability, 4),
            "feature_contributions": feature_contributions,
            "top_risk_drivers": risk_drivers,
            "top_stabilizing_factors": stabilizing_factors,
            "summary": local_summary,
            "distinction_note": (
                "Local explanations show which specific sensor readings shifted the decision "
                "path for this machine. Global feature importance reflects average predictive "
                "utility across the entire test set."
            ),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _compute_physical_checks(row: dict) -> dict:
    """
    Compute physics-based diagnostic relationships defined in the AI4I 2020 benchmark.
    Checks:
    - Temperature difference Delta T = Process Temp - Air Temp (HDF condition: < 8.6 K & Speed < 1380 rpm)
    - Power P = (2 * pi * rpm / 60) * Torque (PWF condition: < 3500 W or > 9000 W)
    - Overstrain = Tool wear * Torque (OSF condition: > 11000/12000/13000 min*Nm)
    - High Tool Wear zone: 200 - 240 min
    """
    try:
        m_type = str(row.get("Type", "L")).upper()
        air_t = float(row.get("Air temperature [K]", 300.0))
        proc_t = float(row.get("Process temperature [K]", 310.0))
        speed = float(row.get("Rotational speed [rpm]", 1500.0))
        torque = float(row.get("Torque [Nm]", 40.0))
        wear = float(row.get("Tool wear [min]", 0.0))

        delta_t = round(proc_t - air_t, 2)
        power_w = round((2.0 * math.pi * speed / 60.0) * torque, 1)
        strain = round(wear * torque, 1)

        # OSF threshold depends on Type
        strain_limit = 11000.0 if m_type == "L" else (12000.0 if m_type == "M" else 13000.0)

        checks = {
            "delta_t": {
                "value": delta_t,
                "unit": "K",
                "condition": "Process Temp - Air Temp",
                "hdf_risk": bool(delta_t < 8.6 and speed < 1380.0),
                "documented_threshold": "< 8.6 K (with speed < 1380 rpm)",
            },
            "power": {
                "value": power_w,
                "unit": "W",
                "condition": "P = (2*pi*rpm/60) * Torque",
                "pwf_risk": bool(power_w < 3500.0 or power_w > 9000.0),
                "documented_threshold": "< 3500 W or > 9000 W",
            },
            "overstrain": {
                "value": strain,
                "unit": "min·Nm",
                "condition": "Tool wear * Torque",
                "osf_risk": bool(strain > strain_limit),
                "documented_threshold": f"> {int(strain_limit)} min·Nm (Type {m_type})",
            },
            "tool_wear_zone": {
                "value": wear,
                "unit": "min",
                "twf_risk": bool(200.0 <= wear <= 240.0),
                "documented_threshold": "200 - 240 min",
            },
            "reference_limits": REFERENCE_LIMITS,
        }
        return checks
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Multi-Label Diagnostics Prediction
# ---------------------------------------------------------------------------

def _diagnose_failure_modes(input_df: pd.DataFrame) -> dict:
    """
    Predict specific failure modes using the separate multi-label diagnostic model.
    Modes: TWF, HDF, PWF, OSF, RNF.
    """
    if _diag_pipeline is None or _diag_metadata is None:
        return {
            "available": False,
            "predicted_modes": [],
            "summary": "Diagnostic model artifact not loaded.",
        }

    try:
        target_modes = _diag_metadata["target_modes"]
        descriptions = _diag_metadata["mode_descriptions"]
        thresholds = _diag_metadata["thresholds"]

        # predict_proba returns a list of 5 (N, 2) arrays
        probs_list = _diag_pipeline.predict_proba(input_df)

        mode_results = []
        predicted_modes = []

        for idx, mode in enumerate(target_modes):
            prob = float(probs_list[idx][0, 1])
            th = float(thresholds.get(mode, 0.5))
            is_predicted = bool(prob >= th)

            mode_info = {
                "code": mode,
                "name": descriptions.get(mode, mode),
                "probability": round(prob, 4),
                "probability_pct": round(prob * 100, 1),
                "threshold": th,
                "predicted": is_predicted,
            }
            mode_results.append(mode_info)

            if is_predicted:
                predicted_modes.append(mode_info)

        if predicted_modes:
            names = [f"{m['name']} ({m['code']})" for m in predicted_modes]
            summary = f"Predicted failure mode(s): {', '.join(names)}"
        else:
            summary = "No failure mode identified by the diagnostic model"

        return {
            "available": True,
            "predicted_modes": predicted_modes,
            "all_modes": mode_results,
            "has_predicted_mode": len(predicted_modes) > 0,
            "summary": summary,
            "disclaimer": (
                "Predicted failure modes are statistical estimates generated by a multi-label "
                "classifier based on historical operating patterns, not a confirmed physical diagnosis."
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "predicted_modes": [],
            "summary": f"Diagnostics error: {str(exc)}",
        }


# ---------------------------------------------------------------------------
# Example machines for the priority table
# ---------------------------------------------------------------------------

EXAMPLE_MACHINES = [
    {"id": "EX-001", "Type": "L", "Air temperature [K]": 298.1,
     "Process temperature [K]": 308.6, "Rotational speed [rpm]": 1551,
     "Torque [Nm]": 42.8, "Tool wear [min]": 0},
    {"id": "EX-002", "Type": "M", "Air temperature [K]": 302.3,
     "Process temperature [K]": 312.1, "Rotational speed [rpm]": 1350,
     "Torque [Nm]": 65.2, "Tool wear [min]": 180},
    {"id": "EX-003", "Type": "H", "Air temperature [K]": 304.5,
     "Process temperature [K]": 316.2, "Rotational speed [rpm]": 1200,
     "Torque [Nm]": 70.0, "Tool wear [min]": 220},
    {"id": "EX-004", "Type": "L", "Air temperature [K]": 296.8,
     "Process temperature [K]": 307.2, "Rotational speed [rpm]": 1620,
     "Torque [Nm]": 38.5, "Tool wear [min]": 45},
    {"id": "EX-005", "Type": "M", "Air temperature [K]": 299.9,
     "Process temperature [K]": 310.0, "Rotational speed [rpm]": 1480,
     "Torque [Nm]": 55.3, "Tool wear [min]": 130},
    {"id": "EX-006", "Type": "H", "Air temperature [K]": 301.0,
     "Process temperature [K]": 314.5, "Rotational speed [rpm]": 1280,
     "Torque [Nm]": 68.7, "Tool wear [min]": 200},
    {"id": "EX-007", "Type": "L", "Air temperature [K]": 297.5,
     "Process temperature [K]": 308.0, "Rotational speed [rpm]": 1700,
     "Torque [Nm]": 32.1, "Tool wear [min]": 10},
    {"id": "EX-008", "Type": "M", "Air temperature [K]": 303.8,
     "Process temperature [K]": 313.4, "Rotational speed [rpm]": 1310,
     "Torque [Nm]": 71.5, "Tool wear [min]": 240},
]


def _score_machines():
    threshold = _metadata["classification_threshold"]
    results = []
    for machine in EXAMPLE_MACHINES:
        machine_id = machine["id"]
        row = {k: v for k, v in machine.items() if k != "id"}
        df = pd.DataFrame([row])
        proba = _extract_failure_probability(_pipeline, df)
        risk = _risk_category(proba, threshold)
        diag = _diagnose_failure_modes(df)
        results.append({
            "id": machine_id,
            **row,
            **risk,
            "diagnostic_summary": diag.get("summary", "N/A"),
            "predicted_modes": [m["code"] for m in diag.get("predicted_modes", [])],
        })
    results.sort(key=lambda x: x["failure_probability"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Dashboard home page."""
    machines = _score_machines()
    high_risk_count = sum(
        1 for m in machines if m["level"] in ("high", "critical")
    )
    return render_template(
        "index.html",
        metadata=_metadata,
        machines=machines,
        total_machines=len(machines),
        high_risk_count=high_risk_count,
        model_comparison=_model_comparison,
        feature_importance=_feature_importance,
        diagnostics_metadata=_diag_metadata,
        diagnostics_evaluation=_diagnostics_evaluation,
        reference_limits=REFERENCE_LIMITS,
    )


@app.route("/predict")
def predict_page():
    """Prediction form page with failure diagnostics and local explainability."""
    return render_template(
        "predict.html",
        metadata=_metadata,
        reference_limits=REFERENCE_LIMITS,
        diagnostics_metadata=_diag_metadata,
    )


@app.route("/live")
def live_page():
    """Live Monitor page with simulated sensor stream and real-time risk charts."""
    return render_template(
        "live.html",
        metadata=_metadata,
        reference_limits=REFERENCE_LIMITS,
        diagnostics_metadata=_diag_metadata,
    )


@app.route("/statistics", endpoint="statistics_page")
@app.route("/insights", endpoint="insights_page")
def statistics_page():
    """Statistics & Model Analytics page — exploratory graphs, model comparison, feature importance from notebook."""
    return render_template(
        "statistics.html",
        metadata=_metadata,
        model_comparison=_model_comparison,
        feature_importance=_feature_importance,
        permutation_importance=_permutation_importance,
        diagnostics_evaluation=_diagnostics_evaluation,
        diagnostics_metadata=_diag_metadata,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Unified prediction endpoint.
    Accepts JSON body with the six input features.
    Returns:
    - Primary binary machine failure probability and risk categorization (preserved contract).
    - Multi-label diagnostic prediction (predicted failure modes).
    - Valid local tree decision path attribution (Explainable AI).
    - Calculated physical metrics and documented limits comparison.
    """
    if not request.is_json:
        return jsonify({"error": "Request Content-Type must be application/json"}), 415

    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    input_df, warnings, error = _validate_and_parse(data)
    if error:
        return jsonify({"error": error, "warnings": warnings}), 422

    # 1. Binary prediction (preserved pipeline)
    threshold = _metadata["classification_threshold"]
    proba = _extract_failure_probability(_pipeline, input_df)
    risk = _risk_category(proba, threshold)

    # 2. Multi-label diagnostics
    diagnostics = _diagnose_failure_modes(input_df)

    # 3. Local Explainability
    local_explanation = _explain_prediction(input_df, proba)

    # 4. Physics checks against documented benchmark limits
    physical_metrics = _compute_physical_checks(data)

    response_payload = {
        "input": data,
        "warnings": warnings,
        **risk,
        "diagnostics": diagnostics,
        "local_explanation": local_explanation,
        "physical_metrics": physical_metrics,
    }

    return jsonify(response_payload), 200


@app.route("/api/diagnose", methods=["POST"])
def api_diagnose():
    """Dedicated endpoint for multi-label failure mode diagnostics."""
    if not request.is_json:
        return jsonify({"error": "Request Content-Type must be application/json"}), 415

    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    input_df, warnings, error = _validate_and_parse(data)
    if error:
        return jsonify({"error": error, "warnings": warnings}), 422

    diagnostics = _diagnose_failure_modes(input_df)
    return jsonify({"input": data, "warnings": warnings, "diagnostics": diagnostics}), 200


@app.route("/api/simulate/next", methods=["GET", "POST"])
def api_simulate_next():
    """
    Generate or step plausible sensor readings for the live telemetry stream.
    Supports continuous realistic random walk, targeted stress scenarios,
    and dynamic multi-phase mixed operation.
    """
    req_data = request.get_json(silent=True) if request.is_json else {}
    if not req_data and request.args:
        req_data = request.args.to_dict()

    scenario = req_data.get("scenario", "normal")
    seed_val = req_data.get("seed")
    step = int(req_data.get("step", 0))
    is_scenario_change = bool(req_data.get("scenario_changed", False))

    if seed_val is not None and str(seed_val).strip() != "":
        try:
            base_seed = int(seed_val)
            tick_seed = (base_seed * 10007 + step) if step > 0 else base_seed
            rng = random.Random(tick_seed & 0xFFFFFFFF)
        except (ValueError, TypeError):
            rng = random.Random()
    else:
        rng = random.Random()

    # Machine Type
    m_type = req_data.get("Type", "L")
    if m_type not in VALID_TYPES:
        m_type = "L"

    # Previous or baseline values
    try:
        air_t = float(req_data.get("Air temperature [K]", 298.1))
        proc_t = float(req_data.get("Process temperature [K]", 308.6))
        speed = float(req_data.get("Rotational speed [rpm]", 1530.0))
        torque = float(req_data.get("Torque [Nm]", 41.5))
        wear = float(req_data.get("Tool wear [min]", 20.0))
    except (ValueError, TypeError):
        air_t, proc_t, speed, torque, wear = 298.1, 308.6, 1530.0, 41.5, 20.0

    delta_t = proc_t - air_t

    if scenario == "thermal_stress":
        # Elevated ambient temperature, low heat dissipation (Delta T < 8.6 K) & speed < 1380 rpm (HDF)
        if is_scenario_change or delta_t > 8.5 or speed > 1380.0 or air_t < 301.5:
            air_t = 303.8 + rng.uniform(-0.4, 0.4)
            proc_t = air_t + rng.uniform(6.0, 7.5)
            speed = 1310.0 + rng.uniform(-30.0, 30.0)
            torque = 50.0 + rng.uniform(-2.5, 2.5)
            wear = float(rng.randint(75, 105))
        else:
            air_t = air_t + 0.35 * (303.8 - air_t) + rng.uniform(-0.25, 0.25)
            proc_t = air_t + rng.uniform(6.0, 7.8)
            speed = speed + 0.35 * (1310.0 - speed) + rng.uniform(-20.0, 20.0)
            torque = torque + 0.35 * (50.0 - torque) + rng.uniform(-1.5, 1.5)
            wear = min(150.0, wear + rng.uniform(0.2, 0.5))

    elif scenario == "mechanical_overstrain":
        # High tool wear combined with high cutting torque (Wear * Torque > 11,000) (OSF)
        if is_scenario_change or wear < 190.0 or torque < 55.0:
            wear = float(rng.randint(205, 222))
            torque = 63.5 + rng.uniform(-2.5, 3.5)
            speed = 1360.0 + rng.uniform(-30.0, 30.0)
            air_t = 298.6 + rng.uniform(-0.3, 0.3)
            proc_t = air_t + 10.3 + rng.uniform(-0.3, 0.3)
        else:
            wear = min(240.0, wear + rng.uniform(0.2, 0.6))
            torque = torque + 0.35 * (63.5 - torque) + rng.uniform(-1.5, 1.5)
            speed = speed + 0.35 * (1360.0 - speed) + rng.uniform(-25.0, 25.0)
            air_t = air_t + 0.35 * (298.6 - air_t) + rng.uniform(-0.2, 0.2)
            proc_t = proc_t + 0.35 * (air_t + 10.3 - proc_t) + rng.uniform(-0.2, 0.2)

    elif scenario == "power_surge":
        # High rotational speed and high torque pushing mechanical power > 9000 W (PWF)
        if is_scenario_change or speed < 2200.0 or torque < 56.0:
            speed = 2450.0 + rng.uniform(-60.0, 80.0)
            torque = 64.0 + rng.uniform(-3.0, 3.0)
            air_t = 299.0 + rng.uniform(-0.4, 0.4)
            proc_t = air_t + 10.2 + rng.uniform(-0.4, 0.4)
            wear = float(rng.randint(50, 75))
        else:
            speed = speed + 0.35 * (2450.0 - speed) + rng.uniform(-40.0, 40.0)
            torque = torque + 0.35 * (64.0 - torque) + rng.uniform(-2.0, 2.0)
            air_t = air_t + 0.35 * (299.0 - air_t) + rng.uniform(-0.2, 0.2)
            proc_t = proc_t + 0.35 * (air_t + 10.2 - proc_t) + rng.uniform(-0.2, 0.2)
            wear = min(120.0, wear + rng.uniform(0.2, 0.5))

    elif scenario == "random_mixed":
        # Multi-phase dynamic operating shift across 24-tick cycle
        cycle = step % 24
        if cycle <= 5:
            # Phase 1: Normal baseline operation (Low Risk ~1.5%)
            target_air, target_dt = 298.2, 10.5
            target_speed, target_torque = 1520.0, 41.0
            wear = min(40.0, max(15.0, wear + rng.uniform(0.1, 0.3)))
        elif cycle <= 10:
            # Phase 2: Moderate load / tool warmup (Moderate Risk ~15.5% - 28.6%)
            target_air, target_dt = 301.2, 9.3
            target_speed, target_torque = 1420.0, 58.0
            wear = min(170.0, max(145.0, wear + rng.uniform(0.3, 0.8)))
        elif cycle <= 16:
            # Phase 3: High stress peak (High/Critical Risk ~55.5% - 68.8%)
            if (step // 24) % 2 == 0:
                # Thermal dissipation deficit
                target_air, target_dt = 303.8, 6.7
                target_speed, target_torque = 1310.0, 50.0
                wear = min(110.0, max(80.0, wear + rng.uniform(0.2, 0.5)))
            else:
                # Power surge peak
                target_air, target_dt = 299.0, 10.2
                target_speed, target_torque = 2450.0, 64.0
                wear = min(90.0, max(55.0, wear + rng.uniform(0.2, 0.5)))
        elif cycle <= 20:
            # Phase 4: Cooldown / operator load throttling (Moderate/Low Risk ~4.2% - 15.5%)
            target_air, target_dt = 299.5, 10.0
            target_speed, target_torque = 1480.0, 45.0
            wear = min(60.0, max(35.0, wear + rng.uniform(0.1, 0.3)))
        else:
            # Phase 5: Restabilized normal operation (Low Risk ~1.5%)
            target_air, target_dt = 298.2, 10.5
            target_speed, target_torque = 1530.0, 41.5
            wear = min(35.0, max(20.0, wear + rng.uniform(0.1, 0.2)))

        air_t = target_air + rng.uniform(-0.3, 0.3)
        proc_t = air_t + target_dt + rng.uniform(-0.25, 0.25)
        speed = target_speed + rng.uniform(-25.0, 25.0)
        torque = target_torque + rng.uniform(-1.5, 1.5)

    elif scenario == "tool_degradation":
        # Progressive rapid tool wear accumulation crossing critical 200-240 min boundary
        if is_scenario_change or wear < 170.0:
            wear = 180.0 + rng.uniform(0.0, 5.0)
            torque = 48.0 + rng.uniform(-1.5, 1.5)
            speed = 1440.0 + rng.uniform(-25.0, 25.0)
            air_t = 298.5 + rng.uniform(-0.3, 0.3)
            proc_t = air_t + 10.3 + rng.uniform(-0.3, 0.3)
        else:
            wear = min(245.0, wear + rng.uniform(1.5, 3.5))
            # Cutting resistance increases with dull cutter
            torque = min(68.0, torque + rng.uniform(0.2, 0.8))
            speed = max(1300.0, speed + rng.uniform(-15.0, 15.0))
            air_t = air_t + 0.35 * (298.5 - air_t) + rng.uniform(-0.15, 0.15)
            proc_t = proc_t + 0.35 * (air_t + 10.3 - proc_t) + rng.uniform(-0.15, 0.15)

    else:
        # Normal bounded random walk within typical safe operational bands
        if is_scenario_change or torque > 52.0 or wear > 120.0 or air_t > 302.0 or speed < 1400.0 or speed > 1700.0:
            air_t = 298.2 + rng.uniform(-0.4, 0.4)
            proc_t = air_t + 10.5 + rng.uniform(-0.3, 0.3)
            speed = 1530.0 + rng.uniform(-30.0, 30.0)
            torque = 41.5 + rng.uniform(-2.0, 2.0)
            wear = float(rng.randint(15, 35))
        else:
            air_t = air_t + 0.35 * (298.2 - air_t) + rng.uniform(-0.2, 0.2)
            proc_t = proc_t + 0.35 * (air_t + 10.5 - proc_t) + rng.uniform(-0.2, 0.2)
            speed = speed + 0.35 * (1530.0 - speed) + rng.uniform(-25.0, 25.0)
            torque = torque + 0.35 * (41.5 - torque) + rng.uniform(-1.5, 1.5)
            wear = min(80.0, wear + rng.uniform(0.1, 0.3))

    # Apply physical safety bounds
    air_t = max(295.0, min(305.0, air_t))
    proc_t = max(air_t + 4.0, min(314.0, proc_t))
    speed = max(1180.0, min(2880.0, speed))
    torque = max(10.0, min(75.0, torque))
    wear = max(0.0, min(250.0, wear))

    next_reading = {
        "Type": m_type,
        "Air temperature [K]": round(air_t, 1),
        "Process temperature [K]": round(proc_t, 1),
        "Rotational speed [rpm]": int(round(speed)),
        "Torque [Nm]": round(torque, 1),
        "Tool wear [min]": int(round(wear)),
        "timestamp": time.strftime("%H:%M:%S"),
        "scenario": scenario,
        "step": step,
        "simulated": True,
    }

    return jsonify(next_reading), 200


@app.route("/api/health")
def api_health():
    """Application and models health check."""
    if _startup_error:
        return jsonify({"status": "error", "error": _startup_error}), 500

    return jsonify({
        "status": "ok",
        "model": _metadata.get("model_name"),
        "threshold": _metadata.get("classification_threshold"),
        "diagnostics_model": _diag_metadata.get("model_name") if _diag_metadata else None,
        "load_time_seconds": _load_time,
        "model_path": str(MODEL_PATH),
        "diagnostics_model_path": str(DIAG_MODEL_PATH),
        "metadata_path": str(METADATA_PATH),
    }), 200


@app.route("/api/model-info")
def api_model_info():
    """Available evaluation metrics, diagnostics and feature importance information."""
    return jsonify({
        "metadata": _metadata,
        "diagnostics_metadata": _diag_metadata,
        "model_comparison": _model_comparison,
        "feature_importance": _feature_importance,
        "permutation_importance": _permutation_importance,
        "diagnostics_evaluation": _diagnostics_evaluation,
        "example_machines": _score_machines(),
    }), 200


@app.route("/api/machines")
def api_machines():
    """Scored example machines sorted by risk (highest first)."""
    return jsonify(_score_machines()), 200


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
