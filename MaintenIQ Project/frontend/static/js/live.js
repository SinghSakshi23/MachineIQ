/**
 * MaintenIQ — live.js
 * Live predictive maintenance telemetry simulation engine.
 * Connects generated streaming sensor readings to the Flask ML prediction API
 * and multi-label diagnostics, updating real-time telemetry cards and Chart.js streams.
 */

'use strict';

(function () {
  // ── Configuration & State ──────────────────────────────────────────────────
  const MAX_HISTORY_POINTS = 20;
  const MAX_TABLE_ROWS = 8;

  let isRunning = false;
  let timerId = null;
  let isFetching = false;
  let tickCount = 0;
  let currentSeed = null;
  let scenarioChanged = true;

  // Active state telemetry readings
  let currentState = {
    Type: 'L',
    'Air temperature [K]': 298.2,
    'Process temperature [K]': 308.7,
    'Rotational speed [rpm]': 1530,
    'Torque [Nm]': 41.5,
    'Tool wear [min]': 20,
  };

  // ── DOM References ─────────────────────────────────────────────────────────
  const btnToggle    = document.getElementById('btn-stream-toggle');
  const btnToggleTxt = document.getElementById('btn-stream-text');
  const selScenario  = document.getElementById('select-scenario');
  const selSpeed     = document.getElementById('select-speed');
  const inpSeed      = document.getElementById('input-seed');
  const btnSeed      = document.getElementById('btn-seed-apply');
  const statusChip   = document.getElementById('stream-status-chip');
  const statusText   = document.getElementById('stream-status-text');
  const alertBox     = document.getElementById('stream-alert');
  const btnClearLog  = document.getElementById('btn-clear-log');
  const logTableBody = document.getElementById('table-log-body');

  // Telemetry cards
  const elProb       = document.getElementById('tel-prob');
  const elRiskBadge  = document.getElementById('tel-risk-badge');
  const elUpdateTime = document.getElementById('tel-update-time');
  const elRiskReason = document.getElementById('tel-risk-reason');
  const elAirTemp    = document.getElementById('tel-air-temp');
  const elProcTemp   = document.getElementById('tel-proc-temp');
  const elDeltaT     = document.getElementById('tel-delta-t');
  const elSpeed      = document.getElementById('tel-speed');
  const elTorque     = document.getElementById('tel-torque');
  const elPower      = document.getElementById('tel-power');
  const elWear       = document.getElementById('tel-wear');
  const elStrain     = document.getElementById('tel-strain');
  const elDiag       = document.getElementById('tel-diag-summary');

  // ── Chart.js Global Initialization ──────────────────────────────────────────
  let chartRisk = null;
  let chartSensors = null;

  function initCharts() {
    const ctxRisk = document.getElementById('chart-risk-trend');
    const ctxSensors = document.getElementById('chart-sensors-trend');
    if (!ctxRisk || !ctxSensors || typeof Chart === 'undefined') return;

    // Chart 1: Failure Probability Trend
    chartRisk = new Chart(ctxRisk, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Estimated Failure Risk (%)',
            data: [],
            borderColor: '#2563eb',
            backgroundColor: 'rgba(37, 99, 235, 0.08)',
            fill: true,
            tension: 0.3,
            borderWidth: 2.5,
            pointRadius: 3,
            pointHoverRadius: 5,
          },
          {
            label: 'Cutoff Threshold (5%)',
            data: [],
            borderColor: '#dc2626',
            borderDash: [5, 5],
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 250 },
        scales: {
          y: {
            min: 0,
            max: 100,
            title: { display: true, text: 'Failure Probability (%)' },
            grid: { color: '#f3f4f6' }
          },
          x: {
            title: { display: false },
            grid: { display: false }
          }
        },
        plugins: {
          legend: { position: 'top', align: 'end' }
        }
      }
    });

    // Chart 2: Sensors Telemetry
    chartSensors = new Chart(ctxSensors, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Proc Temp (K)',
            data: [],
            borderColor: '#ea580c',
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 2,
            yAxisID: 'yTemp',
          },
          {
            label: 'Torque (Nm)',
            data: [],
            borderColor: '#7c3aed',
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 2,
            yAxisID: 'yMech',
          },
          {
            label: 'Speed (rpm / 10)',
            data: [],
            borderColor: '#0284c7',
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 2,
            yAxisID: 'yMech',
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 250 },
        scales: {
          yTemp: {
            type: 'linear',
            position: 'left',
            min: 290,
            max: 330,
            title: { display: true, text: 'Temperature (K)' },
            grid: { color: '#f3f4f6' }
          },
          yMech: {
            type: 'linear',
            position: 'right',
            min: 0,
            max: 300,
            title: { display: true, text: 'Speed/10 & Torque' },
            grid: { display: false }
          },
          x: {
            grid: { display: false }
          }
        },
        plugins: {
          legend: { position: 'top', align: 'end' }
        }
      }
    });

    updateChartTheme(document.documentElement.getAttribute('data-theme') || 'light');
  }

  function updateChartTheme(theme) {
    if (!chartRisk || !chartSensors) return;
    const isDark = (theme === 'dark');
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : '#f3f4f6';
    const textColor = isDark ? '#94a3b8' : '#4b5563';

    if (chartRisk.options && chartRisk.options.scales && chartRisk.options.scales.y) {
      chartRisk.options.scales.y.grid.color = gridColor;
      chartRisk.options.scales.y.ticks = chartRisk.options.scales.y.ticks || {};
      chartRisk.options.scales.y.ticks.color = textColor;
      if (chartRisk.options.scales.y.title) chartRisk.options.scales.y.title.color = textColor;
    }
    if (chartRisk.options && chartRisk.options.plugins && chartRisk.options.plugins.legend) {
      chartRisk.options.plugins.legend.labels = chartRisk.options.plugins.legend.labels || {};
      chartRisk.options.plugins.legend.labels.color = textColor;
    }
    chartRisk.update('none');

    if (chartSensors.options && chartSensors.options.scales) {
      if (chartSensors.options.scales.yTemp) {
        chartSensors.options.scales.yTemp.grid.color = gridColor;
        chartSensors.options.scales.yTemp.ticks = chartSensors.options.scales.yTemp.ticks || {};
        chartSensors.options.scales.yTemp.ticks.color = textColor;
        if (chartSensors.options.scales.yTemp.title) chartSensors.options.scales.yTemp.title.color = textColor;
      }
      if (chartSensors.options.scales.yMech) {
        chartSensors.options.scales.yMech.ticks = chartSensors.options.scales.yMech.ticks || {};
        chartSensors.options.scales.yMech.ticks.color = textColor;
        if (chartSensors.options.scales.yMech.title) chartSensors.options.scales.yMech.title.color = textColor;
      }
    }
    if (chartSensors.options && chartSensors.options.plugins && chartSensors.options.plugins.legend) {
      chartSensors.options.plugins.legend.labels = chartSensors.options.plugins.legend.labels || {};
      chartSensors.options.plugins.legend.labels.color = textColor;
    }
    chartSensors.update('none');
  }

  window.addEventListener('mainteniq-theme-change', (e) => {
    updateChartTheme(e.detail && e.detail.theme ? e.detail.theme : (document.documentElement.getAttribute('data-theme') || 'light'));
  });

  // ── Update Chart Data ───────────────────────────────────────────────────────
  function pushChartData(timeLabel, probPct, procTemp, torque, speedRpm) {
    if (!chartRisk || !chartSensors) return;

    // Update Risk Chart
    chartRisk.data.labels.push(timeLabel);
    chartRisk.data.datasets[0].data.push(probPct);
    chartRisk.data.datasets[1].data.push(5.0); // 5% classification threshold

    if (chartRisk.data.labels.length > MAX_HISTORY_POINTS) {
      chartRisk.data.labels.shift();
      chartRisk.data.datasets[0].data.shift();
      chartRisk.data.datasets[1].data.shift();
    }
    chartRisk.update('none');

    // Update Sensors Chart
    chartSensors.data.labels.push(timeLabel);
    chartSensors.data.datasets[0].data.push(procTemp);
    chartSensors.data.datasets[1].data.push(torque);
    chartSensors.data.datasets[2].data.push(Math.round(speedRpm / 10));

    if (chartSensors.data.labels.length > MAX_HISTORY_POINTS) {
      chartSensors.data.labels.shift();
      chartSensors.data.datasets[0].data.shift();
      chartSensors.data.datasets[1].data.shift();
      chartSensors.data.datasets[2].data.shift();
    }
    chartSensors.update('none');
  }

  // ── Add Row to Event Log Table ──────────────────────────────────────────────
  function addLogRow(timeStr, input, probPct, level, label, diagnostics) {
    if (!logTableBody) return;

    const emptyRow = logTableBody.querySelector('.empty-row');
    if (emptyRow) emptyRow.remove();

    let diagSnippet = 'Clean';
    if (diagnostics && diagnostics.predicted_modes && diagnostics.predicted_modes.length > 0) {
      diagSnippet = diagnostics.predicted_modes.map(m => m.code).join(', ');
    }

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${timeStr}</strong></td>
      <td>${input.Type}</td>
      <td>${input['Air temperature [K]']} K</td>
      <td>${input['Process temperature [K]']} K</td>
      <td>${input['Rotational speed [rpm]']}</td>
      <td>${input['Torque [Nm]']}</td>
      <td>${input['Tool wear [min]']}</td>
      <td><strong>${probPct}%</strong></td>
      <td><span class="risk-badge ${level}">${label}</span></td>
      <td><span class="diag-tag ${diagSnippet !== 'Clean' ? 'alert' : ''}">${diagSnippet}</span></td>
    `;

    logTableBody.insertBefore(tr, logTableBody.firstChild);

    while (logTableBody.children.length > MAX_TABLE_ROWS) {
      logTableBody.removeChild(logTableBody.lastChild);
    }
  }

  // ── Step Simulation Tick ───────────────────────────────────────────────────
  async function stepSimulation() {
    if (!isRunning || isFetching) return;
    isFetching = true;

    try {
      tickCount++;
      const scenario = selScenario.value;

      // 1. Fetch next realistic reading from backend simulator endpoint
      const simReqPayload = {
        ...currentState,
        scenario: scenario,
        step: tickCount,
        scenario_changed: scenarioChanged,
        seed: currentSeed !== null ? currentSeed : undefined,
      };
      scenarioChanged = false;

      const simResp = await fetch('/api/simulate/next', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(simReqPayload),
      });

      if (!simResp.ok) {
        throw new Error(`Telemetry simulation endpoint returned HTTP ${simResp.status}`);
      }

      const nextReading = await simResp.json();
      currentState = nextReading;

      // 2. Score with the actual calibrated ML prediction endpoint
      const predictPayload = {
        Type: nextReading.Type,
        'Air temperature [K]': nextReading['Air temperature [K]'],
        'Process temperature [K]': nextReading['Process temperature [K]'],
        'Rotational speed [rpm]': nextReading['Rotational speed [rpm]'],
        'Torque [Nm]': nextReading['Torque [Nm]'],
        'Tool wear [min]': nextReading['Tool wear [min]'],
      };

      const predResp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(predictPayload),
      });

      if (!predResp.ok) {
        throw new Error(`Prediction model service returned HTTP ${predResp.status}`);
      }

      // Check if simulation was paused while fetch was in flight
      if (!isRunning) return;

      const result = await predResp.json();

      // Clear prior error alert
      if (alertBox) alertBox.style.display = 'none';

      // 3. Update DOM Telemetry Cards
      const timeStr = nextReading.timestamp || new Date().toLocaleTimeString();
      const probPct = result.failure_probability_pct ?? 0;
      const level = result.level || 'low';
      const label = result.label || 'Low Risk';
      const modes = result.diagnostics?.predicted_modes || [];

      if (elProb) elProb.textContent = `${probPct.toFixed(1)}%`;
      if (elRiskBadge) {
        elRiskBadge.textContent = label;
        elRiskBadge.className = `risk-badge ${level}`;
      }

      const deltaVal = (nextReading['Process temperature [K]'] - nextReading['Air temperature [K]']).toFixed(1);
      const powerVal = result.physical_metrics?.power?.value ?? Math.round(nextReading['Torque [Nm]'] * nextReading['Rotational speed [rpm]'] * 0.10472);
      const strainVal = result.physical_metrics?.overstrain?.value ?? Math.round(nextReading['Tool wear [min]'] * nextReading['Torque [Nm]']);

      if (elAirTemp)  elAirTemp.textContent = nextReading['Air temperature [K]'];
      if (elProcTemp) elProcTemp.textContent = nextReading['Process temperature [K]'];
      if (elDeltaT)   elDeltaT.textContent = deltaVal;

      if (elSpeed)    elSpeed.textContent = nextReading['Rotational speed [rpm]'];
      if (elTorque)   elTorque.textContent = nextReading['Torque [Nm]'];
      if (elPower)    elPower.textContent = powerVal;

      if (elWear)     elWear.textContent = nextReading['Tool wear [min]'];
      if (elStrain)   elStrain.textContent = strainVal;

      // Diagnostics Card
      if (elDiag) {
        if (modes.length > 0) {
          elDiag.innerHTML = modes.map(m => `
            <span class="active-mode-pill">⚠ ${m.name} (${m.code})</span>
          `).join(' ');
        } else {
          elDiag.innerHTML = `<span class="clean-diag">✔ No failure mode identified</span>`;
        }
      }

      // Timestamp
      if (elUpdateTime) {
        elUpdateTime.textContent = `Updated: ${timeStr}`;
      }

      // Plain-English Explanation
      if (elRiskReason) {
        let reasonMsg = '';
        const modeListStr = modes.map(m => m.name).join(', ');
        if (level === 'critical') {
          reasonMsg = `🚨 Critical Risk: Model predicts severe failure likelihood (${probPct.toFixed(1)}%). ${modeListStr ? `Identified modes: ${modeListStr}. ` : ''}Extreme operating stress detected (thermal dissipation deficit or cutter overstrain). Immediate maintenance action recommended.`;
        } else if (level === 'high') {
          reasonMsg = `⚠ High Risk: Operational thresholds exceeded (${probPct.toFixed(1)}%). ${modeListStr ? `Identified modes: ${modeListStr}. ` : ''}High mechanical power (>9000 W) or elevated wear-torque strain requires scheduled inspection.`;
        } else if (level === 'medium') {
          reasonMsg = `⚠ Moderate Risk: Early stress indicators detected (${probPct.toFixed(1)}%). Torque or thermal margins are near advisory limits. Continued telemetry monitoring advised.`;
        } else {
          reasonMsg = `✔ Low Risk: Operating smoothly within nominal design parameters (${probPct.toFixed(1)}%). Spindle speed, torque, and thermal dissipation are balanced.`;
        }
        elRiskReason.textContent = reasonMsg;
      }

      // 4. Update Charts
      pushChartData(
        timeStr,
        probPct,
        nextReading['Process temperature [K]'],
        nextReading['Torque [Nm]'],
        nextReading['Rotational speed [rpm]']
      );

      // 5. Add to event table
      addLogRow(timeStr, nextReading, probPct.toFixed(1), level, label, result.diagnostics);

    } catch (err) {
      console.warn('Simulation frame error:', err);
      if (alertBox) {
        alertBox.textContent = `Simulation stream notice: ${err.message}. Retrying on next tick...`;
        alertBox.style.display = 'block';
      }
    } finally {
      isFetching = false;
    }
  }

  // ── Start / Stop Engine ────────────────────────────────────────────────────
  function startSimulation() {
    if (isRunning) return;
    isRunning = true;

    btnToggle.classList.remove('btn-primary');
    btnToggle.classList.add('btn-danger');
    btnToggleTxt.textContent = 'Stop Simulation';
    btnToggle.querySelector('.btn-icon').textContent = '⏹';

    statusChip.className = 'stream-status-chip active';
    statusText.textContent = 'Simulating Telemetry';

    const interval = parseInt(selSpeed.value, 10) || 1000;
    stepSimulation(); // Immediate initial step
    timerId = setInterval(stepSimulation, interval);
  }

  function stopSimulation() {
    if (!isRunning) return;
    isRunning = false;

    if (timerId) {
      clearInterval(timerId);
      timerId = null;
    }

    btnToggle.classList.remove('btn-danger');
    btnToggle.classList.add('btn-primary');
    btnToggleTxt.textContent = 'Start Live Simulation';
    btnToggle.querySelector('.btn-icon').textContent = '▶';

    statusChip.className = 'stream-status-chip idle';
    statusText.textContent = 'Simulation Paused';
  }

  // ── Event Handlers ─────────────────────────────────────────────────────────
  window.toggleSimulation = function() {
    if (isRunning) {
      stopSimulation();
    } else {
      startSimulation();
    }
  };

  window.setScenario = function(scen) {
    if (selScenario) {
      selScenario.value = scen;
      scenarioChanged = true;
      tickCount = 0;
      if (isRunning) {
        stepSimulation();
      }
    }
  };

  btnToggle.addEventListener('click', window.toggleSimulation);

  // Scenario Change: switch scenario immediately on next tick
  selScenario.addEventListener('change', () => {
    scenarioChanged = true;
    tickCount = 0;
    if (isRunning) {
      stepSimulation();
    }
  });

  // Sampling Interval Change
  selSpeed.addEventListener('change', () => {
    if (isRunning) {
      clearInterval(timerId);
      const interval = parseInt(selSpeed.value, 10) || 1000;
      timerId = setInterval(stepSimulation, interval);
    }
  });

  // Random Seed Lock
  btnSeed.addEventListener('click', () => {
    const val = parseInt(inpSeed.value, 10);
    if (!isNaN(val) && val > 0) {
      currentSeed = val;
      btnSeed.textContent = 'Locked ✓';
      btnSeed.classList.add('btn-active');
    } else {
      currentSeed = null;
      inpSeed.value = '';
      btnSeed.textContent = 'Lock';
      btnSeed.classList.remove('btn-active');
    }
  });

  // Clear Table History
  btnClearLog.addEventListener('click', () => {
    logTableBody.innerHTML = `
      <tr class="empty-row">
        <td colspan="10" style="text-align:center; padding:20px; color:var(--gray-500);">
          Log cleared. Next incoming frame will appear here.
        </td>
      </tr>
    `;
  });

  // Stop cleanly if user navigates away
  window.addEventListener('beforeunload', stopSimulation);

  // Initialize Charts
  initCharts();
})();
