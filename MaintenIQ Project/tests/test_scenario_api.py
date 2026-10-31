import urllib.request
import json

scenarios = {
    'Normal': {'Type': 'L', 'Air temperature [K]': 298.2, 'Process temperature [K]': 308.7, 'Rotational speed [rpm]': 1520, 'Torque [Nm]': 41.5, 'Tool wear [min]': 25},
    'Thermal Stress': {'Type': 'L', 'Air temperature [K]': 303.8, 'Process temperature [K]': 310.2, 'Rotational speed [rpm]': 1310, 'Torque [Nm]': 50.0, 'Tool wear [min]': 85},
    'Power Surge': {'Type': 'L', 'Air temperature [K]': 299.1, 'Process temperature [K]': 309.4, 'Rotational speed [rpm]': 2450, 'Torque [Nm]': 64.0, 'Tool wear [min]': 60},
    'Mechanical Overstrain': {'Type': 'L', 'Air temperature [K]': 298.6, 'Process temperature [K]': 308.9, 'Rotational speed [rpm]': 1360, 'Torque [Nm]': 63.5, 'Tool wear [min]': 212},
    'Moderate Thermal/Torque': {'Type': 'L', 'Air temperature [K]': 301.2, 'Process temperature [K]': 310.5, 'Rotational speed [rpm]': 1410, 'Torque [Nm]': 58.0, 'Tool wear [min]': 155},
}

for name, payload in scenarios.items():
    req = urllib.request.Request(
        'http://127.0.0.1:5000/api/predict',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        modes = [m['code'] for m in res.get('diagnostics', {}).get('predicted_modes', [])]
        prob = res.get('failure_probability_pct', 0.0)
        level = res.get('level', 'unknown')
        label = res.get('label', 'unknown')
        print(f"{name:25s} -> Prob: {prob:5.2f}% | Level: {level:8s} | Label: {label:10s} | Modes: {modes}")
