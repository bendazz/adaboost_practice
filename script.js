// Synthetic dataset generator and table renderer for AdaBoost practice.
// No frameworks. Runs on page load.

(function () {
  'use strict';

  // Utility: seeded-ish RNG using crypto if available for better randomness
  function rand() {
    if (window.crypto && window.crypto.getRandomValues) {
      const arr = new Uint32Array(1);
      window.crypto.getRandomValues(arr);
      return (arr[0] / 0xffffffff);
    }
    return Math.random();
  }

  // Generate points from two slightly overlapping Gaussian blobs, then label
  // using a linear boundary on f1 and f2 with small noise. The dataset will
  // have either 2 features (f1, f2) or 3 features (f1, f2, f3), chosen once
  // per regeneration.
  function generateDataset(n = 50, featureCount = 2) {
    const points = [];

    // Parameters for two clusters
    const clusterA = { cx: -1.0, cy: -0.5, sigma: 0.8 };
    const clusterB = { cx: 1.2, cy: 0.7, sigma: 0.8 };

    // Linear decision boundary: w1 * x1 + w2 * x2 + b > 0 => label 1
    const w1 = 0.9, w2 = -0.7, b = 0.1;

    for (let i = 0; i < n; i++) {
      // Pick a cluster
      const fromB = rand() < 0.5;
      const { cx, cy, sigma } = fromB ? clusterB : clusterA;

      // Sample 2D Gaussian via Box–Muller
      const u1 = rand() || 1e-9;
      const u2 = rand();
      const R = Math.sqrt(-2 * Math.log(u1)) * sigma;
      const theta = 2 * Math.PI * u2;
      const x1 = cx + R * Math.cos(theta);
      const x2 = cy + R * Math.sin(theta);

      // Optional third feature: mildly correlated with label
      let x3 = null;
      if (featureCount === 3) {
        // x3 is a noisy combination of x1, x2
        x3 = 0.4 * x1 - 0.2 * x2 + (rand() - 0.5) * 0.5;
      }

      // Compute linear score and add small label noise
      const score = w1 * x1 + w2 * x2 + b + (rand() - 0.5) * 0.2;
      const y = score > 0 ? 1 : 0;

      points.push({ id: i + 1, f1: x1, f2: x2, f3: x3, y });
    }

    return points;
  }

  function formatNumber(x) {
    return typeof x === 'number' ? x.toFixed(3) : '';
  }

  function renderTable(rows) {
    const container = document.getElementById('tableContainer');
    if (!container) return;

    // Determine if any row has f3 (for consistent header)
    const hasF3 = rows.some(r => typeof r.f3 === 'number');

    const table = document.createElement('table');
    table.className = 'data-table';

    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');

    const headers = ['#', 'f1', 'f2'];
    if (hasF3) headers.push('f3');
    headers.push('y');

    headers.forEach(h => {
      const th = document.createElement('th');
      th.textContent = h;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);

    const tbody = document.createElement('tbody');
    rows.forEach(r => {
      const tr = document.createElement('tr');
      const cells = [r.id, formatNumber(r.f1), formatNumber(r.f2)];
      if (hasF3) cells.push(formatNumber(r.f3));
      cells.push(r.y);
      cells.forEach(c => {
        const td = document.createElement('td');
        td.textContent = c;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });

    table.appendChild(thead);
    table.appendChild(tbody);

    container.innerHTML = '';
    container.appendChild(table);

    // Summary
    const counts = rows.reduce((acc, r) => { acc[r.y] = (acc[r.y] || 0) + 1; return acc; }, {});
    const summary = document.createElement('div');
    summary.className = 'summary';
    summary.textContent = `Rows: ${rows.length} · Features: ${hasF3 ? 3 : 2} · Class 0: ${counts[0] || 0} · Class 1: ${counts[1] || 0}`;
    container.appendChild(summary);
  }

  // Train a decision stump (max_depth=1) using a Gini-based split selection similar to scikit-learn's default.
  function trainDecisionStumpGini(rows) {
    const N = rows.length;
    if (N === 0) return null;

    const featureNames = ['f1', 'f2'];
    if (rows.some(r => typeof r.f3 === 'number')) featureNames.push('f3');

    // Helper to compute weighted Gini for a split
    function giniOfCounts(count0, count1) {
      const total = count0 + count1;
      if (total === 0) return 0;
      const p0 = count0 / total;
      const p1 = count1 / total;
      return 1 - (p0 * p0 + p1 * p1);
    }

    let best = {
      feature: null,
      threshold: null,
      impurity: Infinity,
      leftPred: 0,
      rightPred: 0,
    };

    for (const fname of featureNames) {
      // Build (x, y) pairs and sort by x
      const pairs = rows
        .map(r => ({ x: r[fname], y: r.y }))
        .filter(p => typeof p.x === 'number')
        .sort((a, b) => a.x - b.x);

      if (pairs.length === 0) continue;

      // Precompute total class counts
      let total0 = 0, total1 = 0;
      for (const p of pairs) {
        if (p.y === 0) total0++; else total1++;
      }

      // Prefix counts for left side
      let left0 = 0, left1 = 0;
      for (let i = 0; i < pairs.length - 1; i++) {
        const p = pairs[i];
        if (p.y === 0) left0++; else left1++;
        const next = pairs[i + 1];
        if (p.x === next.x) continue; // identical values can't form a valid threshold

        const right0 = total0 - left0;
        const right1 = total1 - left1;

        const gLeft = giniOfCounts(left0, left1);
        const gRight = giniOfCounts(right0, right1);
        const nLeft = left0 + left1;
        const nRight = right0 + right1;
        const weightedImpurity = (nLeft * gLeft + nRight * gRight) / (nLeft + nRight);

        if (weightedImpurity < best.impurity) {
          const threshold = (p.x + next.x) / 2;
          const leftPred = left1 >= left0 ? 1 : 0;
          const rightPred = right1 >= right0 ? 1 : 0;
          best = { feature: fname, threshold, impurity: weightedImpurity, leftPred, rightPred };
        }
      }
    }

    // If no split improved impurity (e.g., constant features), fall back to single leaf majority
    if (!best.feature) {
      // Majority class overall
      const counts = rows.reduce((acc, r) => { acc[r.y] = (acc[r.y] || 0) + 1; return acc; }, {});
      const maj = (counts[1] || 0) >= (counts[0] || 0) ? 1 : 0;
      const errCount = rows.reduce((e, r) => e + (r.y === maj ? 0 : 1), 0);
      return {
        feature: null,
        threshold: null,
        leftPred: maj,
        rightPred: maj,
        errorCount: errCount,
        errorRate: errCount / rows.length,
      };
    }

    // Compute training error using the chosen split
    let err = 0;
    for (const r of rows) {
      const x = r[best.feature];
      const pred = x <= best.threshold ? best.leftPred : best.rightPred;
      if (pred !== r.y) err++;
    }

    return {
      feature: best.feature,
      threshold: best.threshold,
      leftPred: best.leftPred,
      rightPred: best.rightPred,
      errorCount: err,
      errorRate: err / rows.length,
    };
  }

  function computeSay(errorRate) {
    // alpha = 0.5 * ln((1 - err) / err)
    if (errorRate <= 0) return Infinity;
    if (errorRate >= 1) return -Infinity;
    return 0.5 * Math.log((1 - errorRate) / errorRate);
  }

  function init() {
    const regenBtn = document.getElementById('regenBtn');
    const downloadBtn = document.getElementById('downloadCsvBtn');
    const revealBtn = document.getElementById('revealAnswerBtn');
    const answerText = document.getElementById('answerText');

    // Hold onto the current dataset for export
    let currentData = [];
    let currentFeatureCount = 2;

    function regenerate() {
      const n = 15; // fixed row count as requested
      currentFeatureCount = rand() < 0.5 ? 2 : 3; // decide once per regeneration
      currentData = generateDataset(n, currentFeatureCount);
      renderTable(currentData);
      if (answerText) answerText.textContent = '';
    }

    function toCSV(rows) {
      if (!rows || rows.length === 0) return '';
      const hasF3 = rows.some(r => typeof r.f3 === 'number');
      const headers = ['id', 'f1', 'f2'];
      if (hasF3) headers.push('f3');
      headers.push('y');

      const lines = [headers.join(',')];
      for (const r of rows) {
        const values = [
          String(r.id),
          formatNumber(r.f1),
          formatNumber(r.f2)
        ];
        if (hasF3) values.push(formatNumber(r.f3));
        values.push(String(r.y));
        lines.push(values.join(','));
      }
      return lines.join('\n');
    }

    function downloadCSV() {
      if (!currentData || currentData.length === 0) return;
      const csv = toCSV(currentData);
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);

      const now = new Date();
      const pad = (n) => n.toString().padStart(2, '0');
      const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
      const fname = `adaboost_dataset_${currentData.length}rows_${currentFeatureCount}f_${stamp}.csv`;

      const a = document.createElement('a');
      a.href = url;
      a.download = fname;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }

    regenBtn.addEventListener('click', regenerate);
    if (downloadBtn) downloadBtn.addEventListener('click', downloadCSV);
    if (revealBtn) {
      revealBtn.addEventListener('click', () => {
        if (!currentData || currentData.length === 0) return;
        const stump = trainDecisionStumpGini(currentData);
        if (!stump) return;
        const say = computeSay(stump.errorRate);
        const errPct = (stump.errorRate * 100).toFixed(1);
        const parts = [];
        parts.push(`Say: ${Number.isFinite(say) ? say.toFixed(3) : (say > 0 ? '+∞' : '-∞')}`);
        parts.push(`Training error: ${errPct}% (${stump.errorCount}/${currentData.length})`);
        if (stump.feature) {
          parts.push(`Stump: ${stump.feature} ≤ ${formatNumber(stump.threshold)} → ${stump.leftPred}, else ${stump.rightPred}`);
        } else {
          parts.push(`Stump: constant prediction = ${stump.leftPred}`);
        }
        if (answerText) answerText.textContent = parts.join(' · ');
      });
    }
    regenerate(); // initial render on load
  }

  window.addEventListener('DOMContentLoaded', init);
})();
