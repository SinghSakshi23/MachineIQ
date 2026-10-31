/**
 * MaintenIQ — predict.js
 * User-friendly assessment logic, plain-English explanations,
 * failure-mode screening, and expandable technical details.
 */

'use strict';

// ── Plain-English Failure Mode Labels ─────────────────────────────────────────
const FAILURE_MODE_NAMES = {
  TWF: 'Tool wear problem',
  HDF: 'Overheating or heat dissipation problem',
  PWF: 'Power-related problem',
  OSF: 'Excessive mechanical strain',
  RNF: 'Other detected failure pattern',
};

// ── Field Configurations ──────────────────────────────────────────────────────
const FIELDS = [
  {
    id: 'input-air-temp',
    name: 'Air temperature [K]',
    errorId: 'air-temp-error',
    label: 'Air temperature',
  },
  {
    id: 'input-proc-temp',
    name: 'Process temperature [K]',
    errorId: 'proc-temp-error',
    label: 'Process temperature',
  },
  {
    id: 'input-rot-speed',
    name: 'Rotational speed [rpm]',
    errorId: 'rot-speed-error',
    label: 'Rotational speed',
  },
  {
    id: 'input-torque',
    name: 'Torque [Nm]',
    errorId: 'torque-error',
    label: 'Torque',
  },
  {
    id: 'input-tool-wear',
    name: 'Tool wear [min]',
    errorId: 'tool-wear-error',
    label: 'Tool wear',
  },
];

// ── DOM References ────────────────────────────────────────────────────────────
const form             = document.getElementById('predict-form');
const btnPredict       = document.getElementById('btn-predict');
const btnReset         = document.getElementById('btn-reset');
const formAlert        = document.getElementById('form-alert');
const resultPanel      = document.getElementById('result-panel');
const placeholder      = document.getElementById('result-placeholder');
const btnEditReadings  = document.getElementById('btn-edit-readings');
const btnCheckAnother  = document.getElementById('btn-check-another');

// ── Validation Helpers ────────────────────────────────────────────────────────
function showFieldError(errorId, msg) {
  const el = document.getElementById(errorId);
  if (el) { el.textContent = msg; el.classList.add('visible'); }
}

function clearFieldError(errorId) {
  const el = document.getElementById(errorId);
  if (el) { el.textContent = ''; el.classList.remove('visible'); }
}

function clearAllErrors() {
  FIELDS.forEach(f => clearFieldError(f.errorId));
  clearFieldError('type-error');
  if (formAlert) {
    formAlert.classList.remove('visible');
    formAlert.textContent = '';
  }
  FIELDS.forEach(f => {
    const inp = document.getElementById(f.id);
    if (inp) inp.classList.remove('invalid');
  });
  const typeEl = document.getElementById('input-type');
  if (typeEl) typeEl.classList.remove('invalid');
}

function validateForm() {
  clearAllErrors();
  let valid = true;

  // Machine Type
  const typeEl = document.getElementById('input-type');
  if (!typeEl || !typeEl.value) {
    showFieldError('type-error', 'Please select a machine type.');
    if (typeEl) typeEl.classList.add('invalid');
    valid = false;
  }

  // Numeric fields
  FIELDS.forEach(f => {
    const inp = document.getElementById(f.id);
    if (!inp) return;
    const raw = inp.value.trim();
    if (raw === '') {
      showFieldError(f.errorId, `${f.label} is required.`);
      inp.classList.add('invalid');
      valid = false;
    } else if (isNaN(parseFloat(raw))) {
      showFieldError(f.errorId, `${f.label} must be a number.`);
      inp.classList.add('invalid');
      valid = false;
    }
  });

  return valid;
}

function buildPayload() {
  const payload = {
    Type: document.getElementById('input-type').value.trim().toUpperCase(),
  };
  FIELDS.forEach(f => {
    payload[f.name] = parseFloat(document.getElementById(f.id).value);
  });
  return payload;
}

// ── Reference Range Evaluation Helper ─────────────────────────────────────────
function evaluateReferenceRanges(payload, limits) {
  const issues = [];
  if (!limits) return { allWithinRange: true, issues: [] };

  for (const [key, limit] of Object.entries(limits)) {
    const val = payload[key];
    if (val !== undefined && val !== null && !isNaN(val)) {
      const cleanName = key.replace(/ \[.*\]/, '');
      if (val < limit.dataset_min) {
        issues.push({
          name: cleanName,
          value: val,
          unit: limit.unit,
          min: limit.dataset_min,
          max: limit.dataset_max,
          type: 'below',
          text: `${cleanName} (${val} ${limit.unit}) is below the typical reference range (${limit.dataset_min}–${limit.dataset_max} ${limit.unit}).`
        });
      } else if (val > limit.dataset_max) {
        issues.push({
          name: cleanName,
          value: val,
          unit: limit.unit,
          min: limit.dataset_min,
          max: limit.dataset_max,
          type: 'above',
          text: `${cleanName} (${val} ${limit.unit}) is above the typical reference range (${limit.dataset_min}–${limit.dataset_max} ${limit.unit}).`
        });
      }
    }
  }

  return {
    allWithinRange: issues.length === 0,
    issues: issues
  };
}

// ── Render 1: Main Prediction ─────────────────────────────────────────────────
function renderMainPrediction(data) {
  const headingEl    = document.getElementById('result-main-heading');
  const iconBadgeEl  = document.getElementById('result-icon-badge');
  const statusChipEl = document.getElementById('result-status-chip');
  const probNumEl    = document.getElementById('result-prob-number');
  const sentenceEl   = document.getElementById('result-prob-sentence');
  const zeroNoteEl   = document.getElementById('result-zero-prob-note');
  const warningsBox  = document.getElementById('result-warnings-box');
  const warningsText = document.getElementById('result-warnings-text');

  // Friendly headline according to actual risk level and prediction outcome
  if (data.level === 'critical' || data.level === 'high') {
    headingEl.textContent = 'The model predicts a possible failure';
    iconBadgeEl.textContent = '⚠️';
    iconBadgeEl.className = 'result-icon-badge badge-high';
  } else if (data.level === 'medium') {
    headingEl.textContent = 'The machine may need attention';
    iconBadgeEl.textContent = '⚡';
    iconBadgeEl.className = 'result-icon-badge badge-medium';
  } else if (data.predicted_failure === 1) {
    headingEl.textContent = 'Operational anomaly flagged for inspection';
    iconBadgeEl.textContent = '⚡';
    iconBadgeEl.className = 'result-icon-badge badge-medium';
  } else {
    headingEl.textContent = 'The model estimates a low chance of failure';
    iconBadgeEl.textContent = '✔';
    iconBadgeEl.className = 'result-icon-badge badge-low';
  }

  // Risk badge label: Low, Moderate, High, or Critical
  let badgeLabel = 'Low Risk';
  if (data.level === 'medium') badgeLabel = 'Moderate Risk';
  else if (data.level === 'high') badgeLabel = 'High Risk';
  else if (data.level === 'critical') badgeLabel = 'Critical Risk';

  statusChipEl.textContent = badgeLabel;
  statusChipEl.className = `risk-badge ${data.level}`;

  // Estimated probability percentage
  const pctStr = (data.failure_probability_pct ?? (data.failure_probability * 100)).toFixed(1) + '%';
  probNumEl.textContent = pctStr;

  // One short explanatory sentence
  sentenceEl.textContent =
    `Under the entered conditions, the model calculates an estimated ${pctStr} chance of machine failure based on patterns it learned from the dataset.`;

  // Zero-percent note
  if (data.failure_probability === 0 || data.failure_probability_pct === 0) {
    zeroNoteEl.style.display = 'block';
  } else {
    zeroNoteEl.style.display = 'none';
  }

  // Warnings for inputs exceeding training ranges
  if (data.warnings && data.warnings.length > 0) {
    const cleanWarnings = data.warnings.map(w => {
      const match = w.match(/'([^']+)' value ([\d.]+) is outside/);
      if (match) return `${match[1]} (${match[2]}) is outside standard ranges.`;
      return w;
    });
    warningsText.innerHTML = '<strong>Advisory Notice:</strong> ' + cleanWarnings.join(' ');
    warningsBox.style.display = 'flex';
  } else {
    warningsBox.style.display = 'none';
  }
}

// ── Render 2: "Why this result?" Section ──────────────────────────────────────
function renderWhyThisResult(payload, data) {
  const rangeIconEl   = document.getElementById('why-range-icon');
  const rangeTextEl   = document.getElementById('why-range-text');
  const influenceText = document.getElementById('why-influence-text');

  // Check actual payload against reference limits
  const evalResult = evaluateReferenceRanges(payload, window.REFERENCE_LIMITS);

  if (evalResult.allWithinRange) {
    rangeIconEl.textContent = '✔';
    rangeIconEl.className = 'why-item-icon icon-ok';
    rangeTextEl.textContent =
      'All entered sensor readings fall within typical baseline reference ranges from the dataset.';
  } else {
    rangeIconEl.textContent = '⚠️';
    rangeIconEl.className = 'why-item-icon icon-warn';
    const issuesList = evalResult.issues.map(item => item.text).join(' ');
    rangeTextEl.textContent =
      `One or more sensor readings are outside typical reference ranges: ${issuesList}`;
  }

  // Feature influence from exact model decision path
  const xai = data.local_explanation;
  if (xai && xai.available) {
    if (xai.top_risk_drivers && xai.top_risk_drivers.length > 0) {
      const topNames = xai.top_risk_drivers.slice(0, 2).map(d => d.feature).join(' and ');
      influenceText.textContent =
        `${topNames} had the strongest upward influence on the model's failure estimate for this machine.`;
    } else if (data.level === 'low') {
      influenceText.textContent =
        'Operating readings followed decision paths in the model associated with stable, low-risk conditions.';
    } else {
      influenceText.textContent =
        'Decision influence is distributed evenly across the entered operating conditions.';
    }
  } else {
    influenceText.textContent =
      'A detailed local feature breakdown is currently unavailable for these readings.';
  }
}

// ── Render 3: "Possible machine problems" Section ─────────────────────────────
function renderPossibleProblems(diagnostics) {
  const container = document.getElementById('possible-problems-container');
  if (!container) return;
  container.innerHTML = '';

  if (!diagnostics || !diagnostics.available) {
    const errorCard = document.createElement('div');
    errorCard.className = 'problem-card problem-card-neutral';
    errorCard.innerHTML = `
      <div class="problem-card-title">Diagnostic screening unavailable</div>
      <p class="problem-card-desc">The diagnostic model could not screen for specific failure modes at this time.</p>
    `;
    container.appendChild(errorCard);
    return;
  }

  const flaggedModes = diagnostics.predicted_modes || [];

  if (flaggedModes.length > 0) {
    flaggedModes.forEach(mode => {
      const friendlyName = FAILURE_MODE_NAMES[mode.code] || mode.name || mode.code;
      const probPct = mode.probability_pct ?? (mode.probability * 100).toFixed(1);

      const card = document.createElement('div');
      card.className = 'problem-card problem-card-flagged';
      card.innerHTML = `
        <div class="problem-header-row">
          <div class="problem-title-wrap">
            <span class="problem-warning-icon" aria-hidden="true">⚠️</span>
            <span class="problem-friendly-name">${friendlyName}</span>
            <span class="tech-tag">(${mode.code})</span>
          </div>
          <span class="problem-match-pill">${probPct}% pattern match</span>
        </div>
        <p class="problem-card-desc">
          Possible pattern identified by the diagnostic model from the dataset based on current readings. This is a statistical pattern match, not a confirmed physical diagnosis.
        </p>
      `;
      container.appendChild(card);
    });
  } else {
    const cleanCard = document.createElement('div');
    cleanCard.className = 'problem-card problem-card-clean';
    cleanCard.innerHTML = `
      <div class="problem-header-row">
        <div class="problem-title-wrap">
          <span class="problem-clean-icon" aria-hidden="true">✔</span>
          <span class="problem-friendly-name">No specific failure type was identified from the information entered.</span>
        </div>
      </div>
      <p class="problem-card-desc">
        The diagnostic model did not detect a recognized problem pattern (such as tool wear breakdown, thermal stress, or power surge) from these values. Please note that the absence of a detected pattern does not prove that the machine is healthy or free of defects.
      </p>
    `;
    container.appendChild(cleanCard);
  }
}

// ── Render 4: Expandable Technical Details ────────────────────────────────────
function renderTechnicalDetails(data) {
  // 1. Root probability
  const baseProbEl = document.getElementById('tech-base-prob');
  if (baseProbEl && data.local_explanation) {
    const baseVal = data.local_explanation.base_root_probability;
    if (baseVal !== undefined) {
      baseProbEl.textContent = (baseVal * 100).toFixed(1) + '%';
    }
  }

  // 2. Local feature attribution bars
  const contribContainer = document.getElementById('tech-contributions-list');
  if (contribContainer && data.local_explanation) {
    contribContainer.innerHTML = '';
    const contributions = data.local_explanation.feature_contributions || {};
    const entries = Object.entries(contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

    if (entries.length === 0) {
      contribContainer.innerHTML = '<p class="tech-empty-note">No local attribution details available.</p>';
    } else {
      entries.forEach(([feature, val]) => {
        const isUpward = val >= 0;
        const absVal = Math.min(100, Math.abs(val) * 100);
        const sign = isUpward ? '+' : '';
        const barClass = isUpward ? 'bar-upward' : 'bar-downward';
        const textClass = isUpward ? 'text-upward' : 'text-downward';

        const row = document.createElement('div');
        row.className = 'tech-attribution-row';
        row.innerHTML = `
          <div class="attr-header">
            <span class="attr-name">${feature}</span>
            <span class="attr-val ${textClass}">
              ${sign}${(val * 100).toFixed(1)}% probability shift
            </span>
          </div>
          <div class="attr-bar-track">
            <div class="attr-bar-fill ${barClass}" style="width: ${absVal.toFixed(1)}%;"></div>
          </div>
        `;
        contribContainer.appendChild(row);
      });
    }
  }

  // 3. Derived Engineering Calculations Grid
  const physGrid = document.getElementById('tech-physics-grid');
  if (physGrid && data.physical_metrics) {
    physGrid.innerHTML = '';
    const p = data.physical_metrics;

    const cards = [
      {
        title: 'Thermal Difference (ΔT)',
        formula: 'Process Temp − Air Temp',
        value: `${p.delta_t?.value ?? '—'} K`,
        condition: 'ΔT < 8.6 K with speed < 1,380 rpm flags reduced heat dissipation.',
        flagged: p.delta_t?.hdf_risk,
      },
      {
        title: 'Electrical Power (P)',
        formula: 'P = (2π × Speed / 60) × Torque',
        value: `${p.power?.value ?? '—'} W`,
        condition: 'Typical operating range is between 3,500 W and 9,000 W.',
        flagged: p.power?.pwf_risk,
      },
      {
        title: 'Mechanical Overstrain',
        formula: 'Tool wear × Torque',
        value: `${p.overstrain?.value ?? '—'} min·Nm`,
        condition: p.overstrain?.documented_threshold
          ? `Capacity limit: ${p.overstrain.documented_threshold}.`
          : 'Limit depends on machine grade (Type L: 11,000, M: 12,000, H: 13,000 min·Nm).',
        flagged: p.overstrain?.osf_risk,
      },
      {
        title: 'Tool Wear State',
        formula: 'Cumulative insert cut time',
        value: `${p.tool_wear_zone?.value ?? '—'} min`,
        condition: 'Typical replacement window is 200–240 minutes.',
        flagged: p.tool_wear_zone?.twf_risk,
      },
    ];

    cards.forEach(c => {
      const cardEl = document.createElement('div');
      cardEl.className = `tech-calc-card ${c.flagged ? 'calc-warning' : 'calc-normal'}`;
      cardEl.innerHTML = `
        <div class="calc-title">${c.title}</div>
        <div class="calc-formula">${c.formula}</div>
        <div class="calc-value">${c.value}</div>
        <div class="calc-rule ${c.flagged ? 'rule-flagged' : ''}">
          ${c.flagged ? '⚠️ Near or exceeds benchmark condition: ' : ''}${c.condition}
        </div>
      `;
      physGrid.appendChild(cardEl);
    });
  }
}

// ── Complete Result Orchestrator ──────────────────────────────────────────────
function renderResult(payload, data) {
  if (placeholder) placeholder.style.display = 'none';
  if (resultPanel) {
    resultPanel.style.display = 'flex';
    resultPanel.classList.add('visible');
  }

  // Smooth scroll into view on mobile
  if (window.innerWidth < 1024 && resultPanel) {
    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // Render 4 key blocks in DOM sequence
  renderMainPrediction(data);
  renderPossibleProblems(data.diagnostics);
  renderWhyThisResult(payload, data);
  renderTechnicalDetails(data);
}

// ── Submit Prediction API ─────────────────────────────────────────────────────
async function submitPrediction(payload) {
  btnPredict.classList.add('loading');
  btnPredict.disabled = true;

  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await resp.json();

    if (!resp.ok) {
      if (formAlert) {
        formAlert.textContent = data.error || 'The check failed. Please review your inputs and try again.';
        formAlert.classList.add('visible');
      }
      return;
    }

    renderResult(payload, data);

  } catch (err) {
    if (formAlert) {
      formAlert.textContent = 'Could not connect to the server. Make sure the application is running and try again.';
      formAlert.classList.add('visible');
    }
  } finally {
    btnPredict.classList.remove('loading');
    btnPredict.disabled = false;
  }
}

// ── Event Listeners ───────────────────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!validateForm()) return;
  const payload = buildPayload();
  await submitPrediction(payload);
});

btnReset.addEventListener('click', () => {
  form.reset();
  clearAllErrors();
  if (resultPanel) {
    resultPanel.style.display = 'none';
    resultPanel.classList.remove('visible');
  }
  if (placeholder) placeholder.style.display = 'block';
});

if (btnEditReadings) {
  btnEditReadings.addEventListener('click', () => {
    const firstInput = document.getElementById('input-type');
    if (firstInput) {
      firstInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
      firstInput.focus();
    }
  });
}

if (btnCheckAnother) {
  btnCheckAnother.addEventListener('click', () => {
    form.reset();
    clearAllErrors();
    if (resultPanel) {
      resultPanel.style.display = 'none';
      resultPanel.classList.remove('visible');
    }
    if (placeholder) placeholder.style.display = 'block';
    const firstInput = document.getElementById('input-type');
    if (firstInput) {
      firstInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
      firstInput.focus();
    }
  });
}

// Live error clearing
const typeSelect = document.getElementById('input-type');
if (typeSelect) {
  typeSelect.addEventListener('change', () => {
    clearFieldError('type-error');
    typeSelect.classList.remove('invalid');
  });
}

FIELDS.forEach(f => {
  const inp = document.getElementById(f.id);
  if (inp) {
    inp.addEventListener('input', () => {
      clearFieldError(f.errorId);
      inp.classList.remove('invalid');
    });
  }
});

// URL Query Param prefill from dashboard
(function handleUrlParams() {
  const params = new URLSearchParams(window.location.search);
  if (!params.has('type') && !params.has('air')) return;
  const t = params.get('type');
  if (t && typeSelect) typeSelect.value = t.toUpperCase();
  const fieldMap = {
    air: 'input-air-temp',
    proc: 'input-proc-temp',
    speed: 'input-rot-speed',
    torque: 'input-torque',
    wear: 'input-tool-wear',
  };
  for (const [param, id] of Object.entries(fieldMap)) {
    const val = params.get(param);
    const el = document.getElementById(id);
    if (val !== null && el) el.value = val;
  }
  if (params.get('autocheck') === '1' && form) {
    setTimeout(() => {
      form.dispatchEvent(new Event('submit'));
    }, 200);
  }
})();
