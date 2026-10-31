"""
MaintenIQ — Multi-Label Diagnostics & Explainability Training Script
Trains separate multi-output diagnostic model for failure modes:
TWF, HDF, PWF, OSF, RNF.
Computes genuine evaluation metrics, baselines, and permutation importance.
"""

import json
import pathlib
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, precision_recall_fscore_support
from sklearn.inspection import permutation_importance

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
MODEL_DIR = BASE_DIR / "model"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA_PATH)
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
target_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]

X = df[features]
Y = df[target_modes]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 2. Pipeline architecture
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), ["Type"]),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ])
    ]
)

base_rf = RandomForestClassifier(
    n_estimators=120,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

diag_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", MultiOutputClassifier(base_rf))
])

print("Training multi-label diagnostic pipeline...")
diag_pipeline.fit(X_train, Y_train)

# 3. Predict on test set
probs_list = diag_pipeline.predict_proba(X_test)

# Per-label calibrated thresholds
thresholds = {
    "TWF": 0.15,
    "HDF": 0.35,
    "PWF": 0.20,
    "OSF": 0.35,
    "RNF": 0.50,
}

Y_test_pred = np.zeros_like(Y_test.values)
per_label_metrics = {}

for idx, label in enumerate(target_modes):
    p_arr = probs_list[idx][:, 1]
    th = thresholds[label]
    pred_col = (p_arr >= th).astype(int)
    Y_test_pred[:, idx] = pred_col
    prec, rec, f1, sup = precision_recall_fscore_support(
        Y_test.iloc[:, idx], pred_col, average="binary", zero_division=0
    )
    per_label_metrics[label] = {
        "threshold": th,
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "support": int(Y_test.iloc[:, idx].sum()),
    }

sub_acc = float(accuracy_score(Y_test, Y_test_pred))
h_loss = float(hamming_loss(Y_test, Y_test_pred))
micro_f1 = float(f1_score(Y_test, Y_test_pred, average="micro", zero_division=0))
macro_f1 = float(f1_score(Y_test, Y_test_pred, average="macro", zero_division=0))

# 4. Baselines
baselines = {
    "Dummy Prior": DummyClassifier(strategy="prior"),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42)
}
baseline_metrics = {}

for b_name, b_clf in baselines.items():
    b_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", MultiOutputClassifier(b_clf))
    ])
    b_pipe.fit(X_train, Y_train)
    b_pred = b_pipe.predict(X_test)
    b_h_loss = float(hamming_loss(Y_test, b_pred))
    b_sub_acc = float(accuracy_score(Y_test, b_pred))
    b_micro_f1 = float(f1_score(Y_test, b_pred, average="micro", zero_division=0))
    b_macro_f1 = float(f1_score(Y_test, b_pred, average="macro", zero_division=0))
    b_per_lbl = {}
    for i, t in enumerate(target_modes):
        p, r, f, s = precision_recall_fscore_support(
            Y_test.iloc[:, i], b_pred[:, i], average="binary", zero_division=0
        )
        b_per_lbl[t] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f), 4)
        }
    baseline_metrics[b_name] = {
        "exact_match_accuracy": round(b_sub_acc, 4),
        "hamming_loss": round(b_h_loss, 4),
        "micro_f1": round(b_micro_f1, 4),
        "macro_f1": round(b_macro_f1, 4),
        "per_label": b_per_lbl
    }

# 5. Export model & metadata
model_path = MODEL_DIR / "mainteniq_diagnostics_pipeline.joblib"
joblib.dump(diag_pipeline, model_path)
print(f"Diagnostics model saved to: {model_path}")

diagnostics_metadata = {
    "project": "MaintenIQ",
    "model_name": "MultiOutput Random Forest Diagnostics",
    "input_features": features,
    "target_modes": target_modes,
    "mode_descriptions": {
        "TWF": "Tool Wear Failure",
        "HDF": "Heat Dissipation Failure",
        "PWF": "Power Failure",
        "OSF": "Overstrain Failure",
        "RNF": "Random Failure"
    },
    "thresholds": thresholds,
    "test_metrics": {
        "exact_match_accuracy": round(sub_acc, 4),
        "hamming_loss": round(h_loss, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "per_label": per_label_metrics
    },
    "documented_limitations": {
        "RNF": "Random Failure (RNF) is defined in AI4I 2020 as 0.1% independent uniform noise. Because it has no correlation with sensor readings, machine learning models cannot reliably predict it from operating data (recall/precision = 0).",
        "TWF": "Tool Wear Failure has only 46 positive cases in 10,000 samples and overlaps closely with normal high wear, leading to lower recall on standard thresholds.",
        "physical_interpretation": "Outputs represent statistical failure mode estimates based on operating patterns, not direct physical sensor confirmations."
    }
}

with open(MODEL_DIR / "diagnostics_metadata.json", "w", encoding="utf-8") as f:
    json.dump(diagnostics_metadata, f, indent=4)
print("Diagnostics metadata saved.")

eval_report = {
    "model": "MultiOutput Random Forest Diagnostics",
    "exact_match_accuracy": round(sub_acc, 4),
    "hamming_loss": round(h_loss, 4),
    "micro_f1": round(micro_f1, 4),
    "macro_f1": round(macro_f1, 4),
    "per_label": per_label_metrics,
    "baselines": baseline_metrics
}
with open(REPORTS_DIR / "diagnostics_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(eval_report, f, indent=4)
print("Evaluation report saved.")

# 6. Permutation Importance for Binary Model
binary_pipeline = joblib.load(MODEL_DIR / "mainteniq_pipeline.joblib")
y_bin = df["Machine failure"]
_, X_bin_test, _, y_bin_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

perm_result = permutation_importance(
    binary_pipeline, X_bin_test, y_bin_test, n_repeats=10, random_state=42, scoring="roc_auc"
)

perm_dict = []
for i in perm_result.importances_mean.argsort()[::-1]:
    perm_dict.append({
        "feature": features[i],
        "importance_mean": round(float(perm_result.importances_mean[i]), 4),
        "importance_std": round(float(perm_result.importances_std[i]), 4)
    })

with open(REPORTS_DIR / "permutation_importance.json", "w", encoding="utf-8") as f:
    json.dump(perm_dict, f, indent=4)
print("Permutation importance saved.")
print("Stage 2, 3, 4 ML artifacts completed successfully.")
