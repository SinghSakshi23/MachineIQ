/**
 * MaintenIQ — charts.js
 * Chart.js helpers shared by the dashboard.
 * All data comes from window.MAINTENIQ_DATA set server-side.
 */

'use strict';

// ── Chart.js global defaults (adaptive light/dark theme) ──────────────────────
function applyChartsTheme(theme) {
  const isDark = (theme === 'dark');
  Chart.defaults.color = isDark ? '#94a3b8' : '#4b5563';
  Chart.defaults.borderColor = isDark ? 'rgba(255, 255, 255, 0.08)' : '#e5e7eb';
  Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
  Chart.defaults.font.size = 12;
}

applyChartsTheme(document.documentElement.getAttribute('data-theme') || 'light');

window.addEventListener('mainteniq-theme-change', (e) => {
  applyChartsTheme(e.detail && e.detail.theme ? e.detail.theme : (document.documentElement.getAttribute('data-theme') || 'light'));
});

// ── Colour palette (matches light theme CSS tokens) ──────────────────────────
const COLOURS = {
  accent:   '#2563eb',   // --blue
  low:      '#16a34a',   // --risk-low
  medium:   '#d97706',   // --risk-medium
  high:     '#dc2626',   // --risk-high
  critical: '#7c3aed',   // --risk-critical
  navy300:  '#2952a3',
  navy200:  '#1a3a6e',
  surface2: '#f3f4f6',
  surface3: '#e5e7eb',
};

function hexAlpha(hex, alpha) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Failure Distribution Chart ────────────────────────────────────────────────
function buildFailureDistChart(canvasId) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['No Failure', 'Machine Failure'],
      datasets: [{
        label: 'Records',
        data: [9661, 339],
        backgroundColor: [hexAlpha(COLOURS.accent, 0.7), hexAlpha(COLOURS.high, 0.75)],
        borderColor:     [COLOURS.accent, COLOURS.high],
        borderWidth: 1.5,
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.y.toLocaleString()} records`
          }
        }
      },
      scales: {
        x: { grid: { display: false } },
        y: {
          beginAtZero: true,
          ticks: {
            callback: v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v
          }
        }
      }
    }
  });
}

// ── Risk Distribution Doughnut ────────────────────────────────────────────────
function buildRiskDistChart(canvasId, machines) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const counts = { low: 0, medium: 0, high: 0, critical: 0 };
  machines.forEach(m => { if (counts[m.level] !== undefined) counts[m.level]++; });

  const labels = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'];
  const data   = [counts.low, counts.medium, counts.high, counts.critical];
  const colours = [COLOURS.low, COLOURS.medium, COLOURS.high, COLOURS.critical];

  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: colours.map(c => hexAlpha(c, 0.8)),
        borderColor: colours,
        borderWidth: 1.5,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      cutout: '60%',
      plugins: {
        legend: {
          position: 'right',
          labels: { padding: 14, usePointStyle: true, pointStyle: 'circle' }
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.parsed} machines`
          }
        }
      }
    }
  });
}

// ── Model Comparison Chart ────────────────────────────────────────────────────
function buildModelComparisonChart(canvasId, comparisonData) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  if (!comparisonData || !comparisonData.results || comparisonData.results.length === 0) {
    ctx.closest('.chart-card').innerHTML = `
      <div class="card-title">Model Comparison — 5-Fold Cross-Validation</div>
      <div class="empty-state">
        <div class="empty-icon">📭</div>
        <p>No model comparison data found in <code>reports/model_comparison.json</code>.</p>
      </div>`;
    return;
  }

  const results = comparisonData.results;
  const models  = results.map(r => r.model);

  const metricKeys = ['precision', 'recall', 'f1_score', 'roc_auc'];
  const metricLabels = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC'];
  const metricColours = [COLOURS.accent, COLOURS.medium, COLOURS.low, COLOURS.navy300];

  const datasets = metricKeys.map((key, i) => ({
    label: metricLabels[i],
    data: results.map(r => parseFloat((r[key] || 0).toFixed(4))),
    backgroundColor: hexAlpha(metricColours[i], 0.75),
    borderColor: metricColours[i],
    borderWidth: 1.5,
    borderRadius: 4,
  }));

  new Chart(ctx, {
    type: 'bar',
    data: { labels: models, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { position: 'top' },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`
          }
        }
      },
      scales: {
        x: {
          ticks: { maxRotation: 30 },
          grid: { display: false }
        },
        y: {
          beginAtZero: true,
          max: 1.0,
          ticks: { callback: v => v.toFixed(1) }
        }
      }
    }
  });
}

// ── Feature Importance Chart ──────────────────────────────────────────────────
function buildFeatureImportanceChart(canvasId, importanceData) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  if (!importanceData || !importanceData.features || importanceData.features.length === 0) {
    ctx.closest('.chart-card').innerHTML = `
      <div class="card-title">Permutation Feature Importance</div>
      <div class="empty-state">
        <div class="empty-icon">📭</div>
        <p>No feature importance data found in <code>reports/feature_importance.json</code>.</p>
      </div>`;
    return;
  }

  const features = importanceData.features;
  // Already sorted descending from notebook
  const labels  = features.map(f => f.feature);
  const values  = features.map(f => parseFloat(f.importance.toFixed(4)));
  const stds    = features.map(f => parseFloat(f.std.toFixed(4)));

  const barColours = values.map(v =>
    v > 0.10 ? hexAlpha(COLOURS.high,   0.8) :
    v > 0.05 ? hexAlpha(COLOURS.medium, 0.8) :
    v >= 0   ? hexAlpha(COLOURS.accent, 0.8) :
               hexAlpha(COLOURS.navy200, 0.5)
  );

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Mean Δ ROC-AUC',
        data: values,
        backgroundColor: barColours,
        borderColor: barColours.map(c => c.replace(/[\d.]+\)$/, '1)')),
        borderWidth: 1,
        borderRadius: 4,
        errorBars: stds.map(s => ({ plus: s, minus: s })),
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` Mean Δ ROC-AUC: ${ctx.parsed.x.toFixed(4)}`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Mean change in ROC-AUC' },
          ticks: { callback: v => v.toFixed(3) }
        },
        y: { grid: { display: false } }
      }
    }
  });
}

// ── Public API ────────────────────────────────────────────────────────────────
window.MaintenIQCharts = {
  buildFailureDistChart,
  buildRiskDistChart,
  buildModelComparisonChart,
  buildFeatureImportanceChart,
};
