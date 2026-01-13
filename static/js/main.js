// main.js: renders the risk donut and dataset stats
window.addEventListener('DOMContentLoaded', function () {
  const riskCanvas = document.getElementById('riskChart');
  if (riskCanvas) {
    const risk = parseFloat(riskCanvas.dataset.risk) || 0;
    new Chart(riskCanvas, {
      type: 'doughnut',
      data: {
        labels: ['Risk', 'Remaining'],
        datasets: [{
          data: [risk, Math.max(0, 100 - risk)],
          backgroundColor: ['#ef4444', '#e6eefc'],
          borderWidth: 0
        }]
      },
      options: {
        cutout: '70%',
        plugins: { legend: { display: false } },
      }
    });
  }

  // Fetch dataset stats and render a pie chart
  const statsCanvas = document.getElementById('statsChart');
  if (statsCanvas) {
    fetch('/api/stats')
      .then(res => res.json())
      .then(json => {
        if (json.error) { console.error(json.error); return; }
        const cardio = json.cardio || { '0': 0, '1': 0 };
        const labels = ['No CVD', 'Has CVD'];
        const data = [cardio['0'] || 0, cardio['1'] || 0];
        new Chart(statsCanvas, {
          type: 'pie',
          data: {
            labels: labels,
            datasets: [{ data: data, backgroundColor: ['#06b6d4', '#fb923c'] }]
          },
          options: { plugins: { legend: { position: 'bottom' } } }
        });
      })
      .catch(err => console.error('Failed to load stats', err));
  }
});