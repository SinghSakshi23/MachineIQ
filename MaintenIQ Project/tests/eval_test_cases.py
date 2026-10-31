import pandas as pd
import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold

# Load existing pipeline
pipe = joblib.load('model/mainteniq_pipeline.joblib')
classes = list(getattr(pipe, 'classes_', pipe.named_steps['classifier'].classes_))
class_1_idx = classes.index(1) if 1 in classes else 1
threshold = 0.05

# Test inputs
test_cases = [
    {
        'name': 'Case 1: Normal Low-Risk (Preset Low)',
        'payload': {'Type': 'L', 'Air temperature [K]': 298.1, 'Process temperature [K]': 308.6, 'Rotational speed [rpm]': 1551, 'Torque [Nm]': 42.8, 'Tool wear [min]': 0}
    },
    {
        'name': 'Case 2: Extreme High-Risk (Preset High)',
        'payload': {'Type': 'H', 'Air temperature [K]': 304.5, 'Process temperature [K]': 316.2, 'Rotational speed [rpm]': 1200, 'Torque [Nm]': 70.0, 'Tool wear [min]': 220}
    },
    {
        'name': 'Case 3: High Heat Dissipation Stress (HDF)',
        'payload': {'Type': 'M', 'Air temperature [K]': 303.5, 'Process temperature [K]': 309.0, 'Rotational speed [rpm]': 1350, 'Torque [Nm]': 48.0, 'Tool wear [min]': 110}
    },
    {
        'name': 'Case 4: Power Surge (PWF: High speed & torque)',
        'payload': {'Type': 'L', 'Air temperature [K]': 299.0, 'Process temperature [K]': 309.5, 'Rotational speed [rpm]': 2850, 'Torque [Nm]': 68.0, 'Tool wear [min]': 30}
    },
    {
        'name': 'Case 5: High Tool Wear & Mechanical Strain (OSF)',
        'payload': {'Type': 'L', 'Air temperature [K]': 298.5, 'Process temperature [K]': 308.8, 'Rotational speed [rpm]': 1420, 'Torque [Nm]': 62.0, 'Tool wear [min]': 215}
    },
    {
        'name': 'Case 6: Moderate Operating Load (EX-005)',
        'payload': {'Type': 'M', 'Air temperature [K]': 299.9, 'Process temperature [K]': 310.0, 'Rotational speed [rpm]': 1480, 'Torque [Nm]': 55.3, 'Tool wear [min]': 130}
    }
]

print('=== Current Trained Pipeline (Decision Tree) ===')
for tc in test_cases:
    df = pd.DataFrame([tc['payload']])
    raw_pred = pipe.predict(df)[0]
    raw_proba = pipe.predict_proba(df)[0]
    failure_prob = float(raw_proba[class_1_idx])
    displayed_pct = f'{failure_prob * 100:.1f}%'
    pred_class_thresh = int(failure_prob >= threshold)
    print(tc['name'])
    print(f'   raw predict(): {raw_pred}')
    print(f'   raw predict_proba(): {raw_proba.tolist()}')
    print(f'   extracted failure probability (class 1): {failure_prob:.4f}')
    print(f'   displayed percentage: {displayed_pct}')
    print(f'   final predicted class (at threshold {threshold}): {pred_class_thresh}')

# Also evaluate calibrated version on training data
df_data = pd.read_csv('data/ai4i2020.csv')
features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X = df_data[features]
y = df_data['Machine failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
cal_model = CalibratedClassifierCV(estimator=pipe, method='sigmoid', cv=5)
cal_model.fit(X_train, y_train)

print('\n=== Calibrated Pipeline (Sigmoid via 5-Fold CV on Training Set) ===')
cal_classes = list(cal_model.classes_)
cal_class_1_idx = cal_classes.index(1) if 1 in cal_classes else 1
for tc in test_cases:
    df = pd.DataFrame([tc['payload']])
    raw_pred = cal_model.predict(df)[0]
    raw_proba = cal_model.predict_proba(df)[0]
    failure_prob = float(raw_proba[cal_class_1_idx])
    displayed_pct = f'{failure_prob * 100:.1f}%'
    pred_class_thresh = int(failure_prob >= threshold)
    print(tc['name'])
    print(f'   raw predict(): {raw_pred}')
    print(f'   raw predict_proba(): {[round(x, 4) for x in raw_proba.tolist()]}')
    print(f'   extracted failure probability (class 1): {failure_prob:.4f}')
    print(f'   displayed percentage: {displayed_pct}')
    print(f'   final predicted class (at threshold {threshold}): {pred_class_thresh}')
