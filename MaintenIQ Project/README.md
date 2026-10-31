# MaintenIQ — Predictive Maintenance and Machine Failure Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)

A student-friendly predictive maintenance web application that estimates machine failure risk
from six operating conditions using a trained Decision Tree classifier.
Built on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — a synthetic benchmark for educational purposes only.

> **Educational disclaimer:** This model is trained on a synthetic dataset and must **not** be used
> as the sole basis for real factory safety or maintenance decisions.

---

## Features

- 🔮 **Real-time failure prediction** — enter six operating conditions, get an estimated failure probability
- 📊 **Interactive dashboard** — KPI cards, machine priority table, Chart.js analytics
- 🏭 **Machine inspection priority** — eight example machines ranked by predicted risk
- 🧪 **Model comparison** — 5-fold cross-validation results for 7 classifiers
- 📈 **Feature importance** — permutation importance on the test set (ROC-AUC scoring)
- ✅ **Full test suite** — pytest integration tests for all API endpoints

---

## Project Structure

```
MaintenIQ Project/
├── backend/
│   ├── app.py                  # Flask application
│   └── requirements.txt        # Python dependencies
├── data/
│   └── ai4i2020.csv            # Source dataset
├── frontend/
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/
│   │       ├── charts.js       # Chart.js helpers
│   │       ├── dashboard.js    # Dashboard init
│   │       └── predict.js      # Prediction form logic
│   └── templates/
│       ├── base.html
│       ├── index.html          # Dashboard
│       └── predict.html        # Prediction form
├── model/
│   ├── mainteniq_pipeline.joblib  # Trained scikit-learn pipeline
│   └── model_metadata.json        # Feature list, threshold, test metrics
├── notebooks/
│   └── MaintainIQ Project.ipynb  # Source of truth for training
├── reports/
│   ├── model_comparison.json   # 5-fold CV results for 7 models
│   └── feature_importance.json # Permutation importance results
├── tests/
│   └── test_predictions.py     # pytest integration tests
├── .gitignore
├── README.md
└── render.yaml                 # Render deployment config
```

---

## Input Features

The model accepts exactly these six features (no others):

| Feature | Type | Example |
|---|---|---|
| Type | Categorical (L/M/H) | M |
| Air temperature [K] | Numeric | 298.1 |
| Process temperature [K] | Numeric | 308.6 |
| Rotational speed [rpm] | Numeric | 1551 |
| Torque [Nm] | Numeric | 42.8 |
| Tool wear [min] | Numeric | 0 |

Features `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are **excluded** to prevent data leakage.

---

## Quick Start (Local)

### Prerequisites
- Python 3.10 or later
- pip

### 1. Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Run the Flask development server

```bash
python backend/app.py
```

The app will be available at **http://localhost:5000**

### 4. Run the test suite

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard (HTML) |
| GET | `/predict` | Prediction form (HTML) |
| POST | `/api/predict` | JSON prediction — see below |
| GET | `/api/health` | Application and model health |
| GET | `/api/model-info` | Model metadata, comparison, importance |
| GET | `/api/machines` | Scored example machines |

### POST /api/predict

**Request body (JSON):**
```json
{
  "Type": "M",
  "Air temperature [K]": 302.3,
  "Process temperature [K]": 312.1,
  "Rotational speed [rpm]": 1350,
  "Torque [Nm]": 65.2,
  "Tool wear [min]": 180
}
```

**Response:**
```json
{
  "failure_probability": 0.8571,
  "failure_probability_pct": 85.7,
  "predicted_failure": 1,
  "level": "critical",
  "label": "Critical Risk",
  "threshold_used": 0.05,
  "explanation": "The model assigns a 85.7% estimated failure probability...",
  "disclaimer": "This prediction is an estimate...",
  "warnings": []
}
```

---

## Deployment on Render

1. Push the repository to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml` and configure the service

Or configure manually:
- **Build command:** `pip install -r backend/requirements.txt`
- **Start command:** `gunicorn --chdir backend app:app`
- **Environment:** Python 3

---

## Model

- **Algorithm:** Decision Tree (selected by 5-fold cross-validation, best F1-score)
- **Classification threshold:** 0.05 (optimised for recall ≥ 70%)
- **Preprocessing:** StandardScaler for numeric features, OneHotEncoder for `Type`
- **Test F1-score:** 0.645 | **Test Recall:** 0.588 | **Test ROC-AUC:** 0.790
- **Training set:** 8,000 records | **Test set:** 2,000 records

The pipeline was trained in `notebooks/MaintainIQ Project.ipynb` and saved with `joblib.dump`.
**Do not retrain or replace the model without re-running the notebook.**

---

## Regenerating Reports

The `reports/` directory contains JSON files extracted from the notebook outputs.
To regenerate them, re-run all cells in `notebooks/MaintainIQ Project.ipynb` and update:
- `reports/model_comparison.json` — cross-validation results (Cell 21 output)
- `reports/feature_importance.json` — permutation importance (Cell 32 output)

---

## Dataset

The [AI4I 2020 Predictive Maintenance Dataset](https://doi.org/10.24432/C5HS5C)
is a synthetic dataset with 10,000 records and a ~3.4% failure rate.
It was designed to reflect real predictive maintenance scenarios for educational use.
It does **not** represent any specific real-world machine or production system.
