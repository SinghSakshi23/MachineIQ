/**
 * MaintenIQ — Premium Dashboard Interactive Logic (dashboard.js)
 * Implements:
 * - Real-time Machine Intelligence Sandbox with live backend ML prediction
 * - Instant physics telemetry calculations (Power, Delta T, Overstrain)
 * - Industrial scenario presets (Normal, Thermal Stress, Overstrain, Power Surge, Tool Wear)
 * - Tabbed technical model insights switcher
 * - Smooth anchor scrolling and priority machine link generator
 */

'use strict';

(function initMaintenIQDashboard() {
  document.addEventListener('DOMContentLoaded', () => {

    // ── 1. Interactive Machine Intelligence Sandbox ────────────────
    const sliders = {
      air:    document.getElementById('sb-air-temp'),
      proc:   document.getElementById('sb-proc-temp'),
      speed:  document.getElementById('sb-speed'),
      torque: document.getElementById('sb-torque'),
      wear:   document.getElementById('sb-wear'),
    };

    const displays = {
      air:    document.getElementById('sb-val-air'),
      proc:   document.getElementById('sb-val-proc'),
      speed:  document.getElementById('sb-val-speed'),
      torque: document.getElementById('sb-val-torque'),
      wear:   document.getElementById('sb-val-wear'),
    };

    const readouts = {
      probVal:     document.getElementById('sb-prob-val'),
      probSub:     document.getElementById('sb-prob-sub'),
      probFill:    document.getElementById('sb-prob-fill'),
      riskBadge:   document.getElementById('sb-risk-badge'),
      diagBox:     document.getElementById('sb-diag-summary'),
      diagTags:    document.getElementById('sb-diag-tags'),
      powerVal:    document.getElementById('sb-power-val'),
      deltaTVal:   document.getElementById('sb-deltat-val'),
      strainVal:   document.getElementById('sb-strain-val'),
      ctaInspect:  document.getElementById('sb-cta-inspect'),
    };

    let selectedType = 'L';
    let predictTimeout = null;

    // Presets definitions matching industrial failure modes
    const PRESETS = {
      normal: {
        type: 'L',
        air: 298.1,
        proc: 308.6,
        speed: 1551,
        torque: 42.8,
        wear: 20,
      },
      thermal: {
        type: 'M',
        air: 303.8,
        proc: 310.5,
        speed: 1310,
        torque: 52.0,
        wear: 95,
      },
      overstrain: {
        type: 'L',
        air: 298.6,
        proc: 308.9,
        speed: 1360,
        torque: 64.0,
        wear: 215,
      },
      power: {
        type: 'H',
        air: 299.0,
        proc: 309.2,
        speed: 2450,
        torque: 64.5,
        wear: 65,
      },
      wear: {
        type: 'L',
        air: 298.5,
        proc: 308.8,
        speed: 1440,
        torque: 58.0,
        wear: 240,
      },
    };

    // Update slider UI labels
    function updateDisplayLabels() {
      if (sliders.air && displays.air) displays.air.textContent = parseFloat(sliders.air.value).toFixed(1) + ' K';
      if (sliders.proc && displays.proc) displays.proc.textContent = parseFloat(sliders.proc.value).toFixed(1) + ' K';
      if (sliders.speed && displays.speed) displays.speed.textContent = parseInt(sliders.speed.value, 10).toLocaleString() + ' rpm';
      if (sliders.torque && displays.torque) displays.torque.textContent = parseFloat(sliders.torque.value).toFixed(1) + ' Nm';
      if (sliders.wear && displays.wear) displays.wear.textContent = parseInt(sliders.wear.value, 10) + ' min';
    }

    // Local physics checks
    function calculatePhysics(air, proc, speed, torque, wear) {
      // Power (W) = Torque (Nm) * speed (rad/s)
      const omega = (2 * Math.PI * speed) / 60;
      const powerW = Math.round(torque * omega);

      // Delta T = Process Temp - Air Temp
      const deltaT = (proc - air);

      // Overstrain product
      const strainProduct = Math.round(wear * torque);

      if (readouts.powerVal) {
        readouts.powerVal.textContent = powerW.toLocaleString() + ' W';
        readouts.powerVal.style.color = (powerW < 3500 || powerW > 9000) ? 'var(--mq-crimson)' : 'var(--mq-text-primary)';
      }

      if (readouts.deltaTVal) {
        readouts.deltaTVal.textContent = deltaT.toFixed(1) + ' K';
        readouts.deltaTVal.style.color = (deltaT < 8.6) ? 'var(--mq-amber)' : 'var(--mq-text-primary)';
      }

      if (readouts.strainVal) {
        readouts.strainVal.textContent = strainProduct.toLocaleString();
        readouts.strainVal.style.color = (strainProduct > 11000) ? 'var(--mq-crimson)' : 'var(--mq-text-primary)';
      }
    }

    // Apply values to sliders
    function applyValues(values) {
      selectedType = values.type || 'L';

      // Update machine type pills
      document.querySelectorAll('.mq-type-pill').forEach(pill => {
        if (pill.dataset.type === selectedType) {
          pill.classList.add('active');
        } else {
          pill.classList.remove('active');
        }
      });

      if (sliders.air) sliders.air.value = values.air;
      if (sliders.proc) sliders.proc.value = values.proc;
      if (sliders.speed) sliders.speed.value = values.speed;
      if (sliders.torque) sliders.torque.value = values.torque;
      if (sliders.wear) sliders.wear.value = values.wear;

      updateDisplayLabels();
      triggerPrediction();
    }

    // Trigger backend evaluation with debouncing
    function triggerPrediction() {
      if (!sliders.air) return;

      const air    = parseFloat(sliders.air.value);
      const proc   = parseFloat(sliders.proc.value);
      const speed  = parseFloat(sliders.speed.value);
      const torque = parseFloat(sliders.torque.value);
      const wear   = parseFloat(sliders.wear.value);

      // Instant local physics calculation
      calculatePhysics(air, proc, speed, torque, wear);

      // Update link to /predict
      if (readouts.ctaInspect) {
        readouts.ctaInspect.href = `/predict?type=${encodeURIComponent(selectedType)}&air=${air}&proc=${proc}&speed=${speed}&torque=${torque}&wear=${wear}&autocheck=1`;
      }

      // Debounce server API call
      clearTimeout(predictTimeout);
      predictTimeout = setTimeout(async () => {
        try {
          const payload = {
            "Type": selectedType,
            "Air temperature [K]": air,
            "Process temperature [K]": proc,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
          };

          const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });

          if (!res.ok) throw new Error('Prediction API failed');
          const data = await res.json();
          renderPredictionResult(data);
        } catch (err) {
          console.warn('Sandbox live prediction fallback:', err);
          // Fallback heuristic estimation if network fails
          renderFallbackEstimate(air, proc, speed, torque, wear);
        }
      }, 120);
    }

    // Render results from /api/predict
    function renderPredictionResult(data) {
      const pct = data.failure_probability_pct ?? Math.round(data.failure_probability * 1000) / 10;
      const level = (data.level || 'low').toLowerCase();
      const label = data.label || 'Low';

      if (readouts.probVal) readouts.probVal.textContent = pct.toFixed(1) + '%';
      if (readouts.probFill) {
        readouts.probFill.style.width = Math.min(100, Math.max(2, pct)) + '%';
        readouts.probFill.style.backgroundColor = data.colour || getRiskColor(level);
      }

      if (readouts.riskBadge) {
        readouts.riskBadge.textContent = label + ' Risk';
        readouts.riskBadge.className = `mq-risk-badge ${level}`;
      }

      if (readouts.probSub) {
        if (level === 'critical') {
          readouts.probSub.textContent = 'Immediate mechanical inspection mandated.';
        } else if (level === 'high') {
          readouts.probSub.textContent = 'High probability of imminent machine failure.';
        } else if (level === 'medium') {
          readouts.probSub.textContent = 'Moderate load. Operating near advisory threshold.';
        } else {
          readouts.probSub.textContent = 'Healthy nominal operating state.';
        }
      }

      // Diagnostics summary
      if (readouts.diagBox && data.diagnostics) {
        const diag = data.diagnostics;
        if (diag.predicted_modes && diag.predicted_modes.length > 0) {
          const modeBadges = diag.predicted_modes.map(m => 
            `<span class="mq-step-pill" style="border-color: var(--mq-crimson); color: #fca5a5; background: rgba(239,68,68,0.15)">⚠️ ${m.name} (${m.code})</span>`
          ).join(' ');
          readouts.diagBox.innerHTML = `
            <div class="mq-diagnostics-box-title">🚨 Active Failure Diagnostics</div>
            <p style="margin-bottom: 8px;">${diag.summary}</p>
            <div style="display:flex; flex-wrap:wrap; gap:6px;">${modeBadges}</div>
          `;
        } else {
          readouts.diagBox.innerHTML = `
            <div class="mq-diagnostics-box-title">✅ Diagnostics Nominal</div>
            <p style="color: var(--mq-text-secondary); margin:0;">No specific physical failure modes (TWF, HDF, PWF, OSF, RNF) triggered at current operational setpoints.</p>
          `;
        }
      }
    }

    // Heuristic color getter
    function getRiskColor(level) {
      switch(level) {
        case 'critical': return 'var(--mq-violet-light)';
        case 'high':     return 'var(--mq-crimson)';
        case 'medium':   return 'var(--mq-amber)';
        default:         return 'var(--mq-lime)';
      }
    }

    // Fallback if offline
    function renderFallbackEstimate(air, proc, speed, torque, wear) {
      let score = 0.015;
      const omega = (2 * Math.PI * speed) / 60;
      const power = torque * omega;
      const deltaT = proc - air;
      const strain = wear * torque;

      if (deltaT < 8.6 && speed < 1380) score += 0.55;
      if (strain > 11000) score += 0.65;
      if (power < 3500 || power > 9000) score += 0.70;
      if (wear > 200) score += 0.35;

      const pct = Math.min(96.0, score * 100);
      let level = 'low';
      let label = 'Low';
      if (pct >= 55) { level = 'critical'; label = 'Critical'; }
      else if (pct >= 35) { level = 'high'; label = 'High'; }
      else if (pct >= 15) { level = 'medium'; label = 'Medium'; }

      renderPredictionResult({
        failure_probability_pct: pct,
        level: level,
        label: label,
        colour: getRiskColor(level)
      });
    }

    // Bind slider events
    Object.values(sliders).forEach(slider => {
      if (!slider) return;
      slider.addEventListener('input', () => {
        updateDisplayLabels();
        // Remove preset highlight when customized
        document.querySelectorAll('.mq-preset-btn').forEach(b => b.classList.remove('active'));
        triggerPrediction();
      });
    });

    // Preset buttons
    document.querySelectorAll('.mq-preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.mq-preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const key = btn.dataset.preset;
        if (PRESETS[key]) {
          applyValues(PRESETS[key]);
        }
      });
    });

    // Machine type selector in Sandbox
    document.querySelectorAll('.mq-type-pill').forEach(pill => {
      pill.addEventListener('click', () => {
        document.querySelectorAll('.mq-type-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        selectedType = pill.dataset.type || 'L';
        triggerPrediction();
      });
    });

    // Initial trigger for sandbox
    if (sliders.air) {
      updateDisplayLabels();
      triggerPrediction();
    }


    // ── 2. Tabbed Technical Model Insights ────────────────────────
    const tabBtns = document.querySelectorAll('.mq-tab-btn');
    const tabContents = document.querySelectorAll('.mq-tab-content');

    tabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetId = btn.dataset.tab;
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));

        btn.classList.add('active');
        const targetContent = document.getElementById(targetId);
        if (targetContent) targetContent.classList.add('active');
      });
    });


    // ── 3. Smooth Anchor Scrolling ────────────────────────────────
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function(e) {
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });

    // ── 4. Theme Change Listener ─────────────────────────────────
    window.addEventListener('mainteniq-theme-change', () => {
      triggerPrediction();
    });

  });
})();
