import json
import base64
import os
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "MaintainIQ Project.ipynb"
OUTPUT_DIR = BASE_DIR / "frontend" / "static" / "img" / "notebook"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

extracted = []

# Mapping cells to filenames and descriptions
cell_map = {
    12: [("target_distribution.png", "Machine Failure Class Distribution")],
    13: [("feature_distributions_hist.png", "Distribution of Machine Operating Features (Histograms)")],
    14: [
        ("boxplot_air_temperature.png", "Air Temperature by Machine Failure Status"),
        ("boxplot_process_temperature.png", "Process Temperature by Machine Failure Status"),
        ("boxplot_rotational_speed.png", "Rotational Speed by Machine Failure Status"),
        ("boxplot_torque.png", "Torque by Machine Failure Status"),
        ("boxplot_tool_wear.png", "Tool Wear by Machine Failure Status")
    ],
    15: [("correlation_heatmap.png", "Correlation Heatmap of Operating Features & Failure")],
    22: [("model_comparison_cv.png", "5-Fold Cross-Validation Model Comparison")],
    26: [("threshold_tuning_curve.png", "Classification Threshold Tuning Curve")],
    31: [("confusion_matrix_heatmap.png", "Final Model Confusion Matrix Heatmap")],
    33: [("feature_importance_barplot.png", "Permutation Feature Importance Bar Chart")]
}

for cell_idx, specs in cell_map.items():
    cell = nb["cells"][cell_idx]
    outputs = cell.get("outputs", [])
    img_outputs = [out for out in outputs if "image/png" in out.get("data", {})]
    
    for i, (fname, title) in enumerate(specs):
        if i < len(img_outputs):
            raw_b64 = img_outputs[i]["data"]["image/png"]
            img_bytes = base64.b64decode(raw_b64)
            target_path = OUTPUT_DIR / fname
            with open(target_path, "wb") as img_file:
                img_file.write(img_bytes)
            extracted.append((fname, title, len(img_bytes)))

print(f"Successfully extracted {len(extracted)} images to {OUTPUT_DIR}:")
for fname, title, size in extracted:
    print(f" - {fname} ({size} bytes): {title}")
