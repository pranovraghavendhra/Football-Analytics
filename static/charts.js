function initCharts(passData, playerStats) {

  const chartDefaults = {
    plugins: {
      legend: { labels: { color: '#aaa', font: { size: 12 } } }
    },
    scales: {
      x: { ticks: { color: '#aaa' }, grid: { color: '#2a2a4a' } },
      y: { ticks: { color: '#aaa' }, grid: { color: '#2a2a4a' } }
    }
  };

  // Pass chart
  new Chart(document.getElementById('passChart'), {
    type: 'bar',
    data: {
      labels: ['Team 1 Passes', 'Team 2 Passes', 'Interceptions'],
      datasets: [{
        label: 'Count',
        data: [passData.team1_passes, passData.team2_passes, passData.interceptions],
        backgroundColor: ['#3b82f6', '#ef4444', '#f59e0b'],
        borderRadius: 8,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: chartDefaults.scales
    }
  });

  // Prepare player data
  const playerIds = Object.keys(playerStats).filter(id => playerStats[id].max_speed > 0);
  const speeds = playerIds.map(id => playerStats[id].max_speed);
  const distances = playerIds.map(id => playerStats[id].total_distance);
  const teamColors = playerIds.map(id =>
    playerStats[id].team === 1 ? '#3b82f6' : '#ef4444'
  );

  // Speed chart
  new Chart(document.getElementById('speedChart'), {
    type: 'bar',
    data: {
      labels: playerIds.map(id => `P${id}`),
      datasets: [{
        label: 'Max Speed (km/h)',
        data: speeds,
        backgroundColor: teamColors,
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.parsed.y} km/h`
          }
        }
      },
      scales: {
        x: { ticks: { color: '#aaa', font: { size: 10 } }, grid: { color: '#2a2a4a' } },
        y: {
          ticks: { color: '#aaa' },
          grid: { color: '#2a2a4a' },
          title: { display: true, text: 'km/h', color: '#888' }
        }
      }
    }
  });

  // Distance chart
  new Chart(document.getElementById('distChart'), {
    type: 'bar',
    data: {
      labels: playerIds.map(id => `P${id}`),
      datasets: [{
        label: 'Distance (m)',
        data: distances,
        backgroundColor: teamColors,
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.parsed.y} m`
          }
        }
      },
      scales: {
        x: { ticks: { color: '#aaa', font: { size: 10 } }, grid: { color: '#2a2a4a' } },
        y: {
          ticks: { color: '#aaa' },
          grid: { color: '#2a2a4a' },
          title: { display: true, text: 'metres', color: '#888' }
        }
      }
    }
  });
}