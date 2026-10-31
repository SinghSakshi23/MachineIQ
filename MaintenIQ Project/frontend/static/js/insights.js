/**
 * MaintenIQ — insights.js
 * Initialises charts on the Model Insights page.
 * Data is read from window.INSIGHTS_DATA set inline by the template.
 */

'use strict';

(function initInsights() {
  try {
    const { feature_importance, model_comparison } = window.INSIGHTS_DATA || {};

    // Feature importance chart
    if (document.getElementById('chart-feature-importance')) {
      MaintenIQCharts.buildFeatureImportanceChart('chart-feature-importance', feature_importance);
    }

    // Model comparison chart
    if (document.getElementById('chart-model-comparison')) {
      MaintenIQCharts.buildModelComparisonChart('chart-model-comparison', model_comparison);
    }
  } catch (err) {
    console.error('Insights page init error:', err);
  }
})();
