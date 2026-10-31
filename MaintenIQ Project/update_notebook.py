import json

nb_path = 'notebooks/MaintainIQ Project.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

markdown_stage2 = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        "## 11. Multi-Label Failure Diagnostics Model\n",
        "\n",
        "In addition to predicting general machine failure, we build a dedicated **multi-label diagnostic model**.\n",
        "The dataset provides 5 failure modes:\n",
        "- **TWF**: Tool Wear Failure\n",
        "- **HDF**: Heat Dissipation Failure\n",
        "- **PWF**: Power Failure\n",
        "- **OSF**: Overstrain Failure\n",
        "- **RNF**: Random Failure\n",
        "\n",
        "> **Methodology & Constraints**:\n",
        "> 1. The binary machine failure classifier is kept separate and untouched.\n",
        "> 2. Only the six valid operating condition features (`Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`) are used as inputs.\n",
        "> 3. Failure-mode labels are used **only** as targets for the diagnostic model, never as inputs.\n",
        "> 4. A `MultiOutputClassifier` wrapping `RandomForestClassifier` is employed with balanced class weights to support multiple co-occurring failure modes."
    ]
}

code_stage2 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import joblib\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.multioutput import MultiOutputClassifier\n",
        "from sklearn.dummy import DummyClassifier\n",
        "from sklearn.metrics import accuracy_score, hamming_loss, f1_score, precision_recall_fscore_support\n",
        "\n",
        "target_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']\n",
        "Y_diag = df[target_modes]\n",
        "\n",
        "# Train-test split (80/20 with random_state=42)\n",
        "X_train_d, X_test_d, Y_train_d, Y_test_d = train_test_split(\n",
        "    X, Y_diag, test_size=0.2, random_state=42\n",
        ")\n",
        "\n",
        "diag_rf = RandomForestClassifier(n_estimators=120, class_weight='balanced', random_state=42, n_jobs=-1)\n",
        "diag_pipeline = Pipeline([\n",
        "    ('preprocessor', preprocessor),\n",
        "    ('classifier', MultiOutputClassifier(diag_rf))\n",
        "])\n",
        "\n",
        "diag_pipeline.fit(X_train_d, Y_train_d)\n",
        "print('Multi-label diagnostic model trained successfully!')\n"
    ]
}

markdown_eval = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        "### Evaluation & Comparison against Baselines\n",
        "We evaluate multi-label performance using Exact Match Accuracy, Hamming Loss, Micro/Macro F1-score, and per-label precision/recall."
    ]
}

code_eval = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        "probs_d = diag_pipeline.predict_proba(X_test_d)\n",
        "thresholds = {'TWF': 0.15, 'HDF': 0.35, 'PWF': 0.20, 'OSF': 0.35, 'RNF': 0.50}\n",
        "\n",
        "Y_pred_d = np.zeros_like(Y_test_d.values)\n",
        "print('Per-label diagnostic metrics on test set:')\n",
        "for idx, lbl in enumerate(target_modes):\n",
        "    p_col = probs_d[idx][:, 1]\n",
        "    th = thresholds[lbl]\n",
        "    pred_col = (p_col >= th).astype(int)\n",
        "    Y_pred_d[:, idx] = pred_col\n",
        "    prec, rec, f1, sup = precision_recall_fscore_support(Y_test_d.iloc[:, idx], pred_col, average='binary', zero_division=0)\n",
        "    print(f'{lbl:4s} -> Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f} (Support: {sup})')\n",
        "\n",
        "print(f'Exact Match Accuracy: {accuracy_score(Y_test_d, Y_pred_d):.4f}')\n",
        "print(f'Hamming Loss: {hamming_loss(Y_test_d, Y_pred_d):.4f}')\n",
        "print(f'Micro F1: {f1_score(Y_test_d, Y_pred_d, average=\"micro\"):.4f}')\n",
        "print(f'Macro F1: {f1_score(Y_test_d, Y_pred_d, average=\"macro\"):.4f}')\n"
    ]
}

markdown_export = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        "### Exporting Diagnostic Pipeline\n",
        "We export the diagnostic pipeline as an independent artifact `model/mainteniq_diagnostics_pipeline.joblib`."
    ]
}

code_export = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        "diag_model_path = model_dir / 'mainteniq_diagnostics_pipeline.joblib'\n",
        "joblib.dump(diag_pipeline, diag_model_path)\n",
        "print('Diagnostic pipeline saved at:', diag_model_path.resolve())\n"
    ]
}

current_source = ''.join([''.join(c.get('source', [])) for c in nb['cells']])
if 'mainteniq_diagnostics_pipeline' not in current_source:
    nb['cells'].extend([markdown_stage2, code_stage2, markdown_eval, code_eval, markdown_export, code_export])
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated with Stage 2 cells.")
else:
    print("Notebook already has Stage 2 cells.")
