const snapshot = {
  models: [
    { name: "Linear ML + residual state", mae: 5.6077, rmse: 6.7111, mape: null, coverage: "100.00%", width: 97.9382, status: "Best demo RMSE" },
    { name: "Seasonal naive", mae: 7.5000, rmse: 7.5000, mape: 1.15, coverage: null, width: null, status: "Reference" },
    { name: "Linear regression", mae: 9.3398, rmse: 10.0411, mape: 1.41, coverage: null, width: null, status: "Benchmark" },
    { name: "Random Forest", mae: 22.5467, rmse: 30.3171, mape: 3.82, coverage: null, width: null, status: "Benchmark" },
    { name: "Neural state-space", mae: 67.1321, rmse: 73.3443, mape: 10.19, coverage: "100.00%", width: 267.8903, status: "Initial neural run" },
    { name: "Local-level state-space", mae: 68.6303, rmse: 79.5340, mape: null, coverage: "100.00%", width: 278.2685, status: "Stochastic baseline" },
    { name: "Local-linear-trend state-space", mae: 75.6830, rmse: 103.4272, mape: null, coverage: "100.00%", width: 393.3754, status: "Stochastic baseline" }
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
  rolling: {
    summary: [["Mean RMSE", "9.1505"], ["RMSE standard deviation", "7.2493"], ["Mean coverage", "95.83%"], ["Mean interval width", "74.3880"]],
    folds: [[21, 23.2248, "75.00%", 36.1335], [25, 12.9887, "100.00%", 86.6492], [29, 6.1640, "100.00%", 86.1735], [33, 1.3800, "100.00%", 82.5466], [37, 7.5218, "100.00%", 78.4465], [41, 3.6238, "100.00%", 76.3788]]
  },
  roadmap: [
    ["0–6", "Foundation to simulation", "done"],
    ["7", "Dashboard", "next"],
    ["8", "Hybrid research", "planned"],
    ["9", "Dissertation release", "planned"],
    ["10", "Future extensions", "planned"]
  ]
};

document.querySelector("#model-table").innerHTML = snapshot.models.map(model => `
  <tr><td><strong>${model.name}</strong></td><td>${model.mae.toFixed(4)}</td><td>${model.rmse.toFixed(4)}</td><td>${model.mape === null ? "—" : model.mape.toFixed(2) + "%"}</td><td>${model.coverage ?? "—"}</td><td>${model.width === null ? "—" : model.width.toFixed(4)}</td><td class="status">${model.status}</td></tr>
`).join("");

function renderMetrics(id, metrics) {
  document.querySelector(`#${id}`).innerHTML = metrics.map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join("");
}
renderMetrics("uncertainty", snapshot.uncertainty);
renderMetrics("simulation", snapshot.simulation);
renderMetrics("rolling-summary", snapshot.rolling.summary);
document.querySelector("#rolling-table").innerHTML = snapshot.rolling.folds.map((fold, index) => `
  <tr><td>${index + 1}</td><td>${fold[0]}</td><td>${fold[1].toFixed(4)}</td><td>${fold[2]}</td><td>${fold[3].toFixed(4)}</td></tr>
`).join("");
document.querySelector("#roadmap").innerHTML = snapshot.roadmap.map(([phase, title, status]) => `
  <div class="milestone ${status}"><b><span class="dot"></span>Phase ${phase}</b><small>${title}</small></div>
`).join("");
