const snapshot = {
  models: [
    { name: "Seasonal naive", mae: 7.5000, rmse: 7.5000, mape: 1.15, status: "Reference" },
    { name: "Linear regression", mae: 9.3398, rmse: 10.0411, mape: 1.41, status: "Best demo RMSE" },
    { name: "Random Forest", mae: 22.5467, rmse: 30.3171, mape: 3.82, status: "Benchmark" }
  ],
  uncertainty: [
    ["Nominal coverage", "80%, 90%, 95%"],
    ["Calibration size", "8 observations"],
    ["Observed demo coverage", "50.00%"],
    ["Interval radius", "11.4716"]
  ],
  simulation: [
    ["Paths", "100"],
    ["Horizon", "6 steps"],
    ["Median final value", "513.20"],
    ["10%-90% final range", "389.16–650.33"],
    ["Crossing probability", "1.00% at threshold 700"]
  ],
  roadmap: [
    ["0–6", "Foundation to simulation", "done"],
    ["7", "Dashboard", "next"],
    ["8", "Hybrid research", "planned"],
    ["9", "Dissertation release", "planned"],
    ["10", "Future extensions", "planned"]
  ]
};

document.querySelector("#model-table").innerHTML = snapshot.models.map(model => `
  <tr><td><strong>${model.name}</strong></td><td>${model.mae.toFixed(4)}</td><td>${model.rmse.toFixed(4)}</td><td>${model.mape.toFixed(2)}%</td><td class="status">${model.status}</td></tr>
`).join("");

function renderMetrics(id, metrics) {
  document.querySelector(`#${id}`).innerHTML = metrics.map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join("");
}
renderMetrics("uncertainty", snapshot.uncertainty);
renderMetrics("simulation", snapshot.simulation);
document.querySelector("#roadmap").innerHTML = snapshot.roadmap.map(([phase, title, status]) => `
  <div class="milestone ${status}"><b><span class="dot"></span>Phase ${phase}</b><small>${title}</small></div>
`).join("");
