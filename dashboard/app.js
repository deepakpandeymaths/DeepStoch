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
  realWorld: {
    summary: [["Mean RMSE", "91.1869"], ["RMSE standard deviation", "23.5358"], ["Mean empirical coverage", "94.76%"], ["Absolute coverage error", "4.76%"], ["Mean interval width", "331.2871"]],
    nominal: 0.90,
    coverage: [0.8973, 0.9524, 0.9286, 0.9524, 0.9688, 0.9048, 0.9658, 0.9792, 0.9554, 0.9747, 0.9449],
    widths: [338.49, 326.83, 345.13, 371.61, 299.28, 306.52, 337.52, 344.72, 367.87, 271.32, 334.86]
  },
  roadmap: [
    ["0–6", "Foundation to simulation", "done"],
    ["7", "Dashboard", "next"],
    ["8", "Hybrid research", "planned"],
    ["9", "Dissertation release", "planned"],
    ["10", "Future extensions", "planned"]
  ]
};

let liveRunContext = null;

const copilotAnswers = [
  { keys: ["what is deepstoch", "purpose", "meaning", "do"], answer: "DeepStoch is a research platform for forecasting time-series data under uncertainty. It learns a likely future value, produces an interval around that forecast, simulates plausible future paths, and checks the interval on future observations. The research goal is calibrated forecasting—not just a low error score." },
  { keys: ["uncertainty", "interval", "trust", "coverage"], answer: "A point forecast can say 620, but it does not say how risky that estimate is. DeepStoch adds a prediction interval such as [580, 665]. Coverage asks how often future observations actually fall inside those intervals. The current real-world neural experiment achieved 94.76% mean empirical coverage against a 90% nominal target across 11 chronological folds." },
  { keys: ["model", "neural", "gru", "machine learning"], answer: "DeepStoch compares several families: transparent baselines, linear regression, Random Forest, local-level and local-trend state-space models, an ML-plus-residual-state hybrid, and a probabilistic GRU neural state-space model. A model is not accepted as better merely because it is deep; it must be compared on accuracy, coverage, interval width, stability, and computational cost." },
  { keys: ["real world", "appliances", "dataset"], answer: "The first real-world experiment uses the UCI Appliances Energy Prediction series. The neural model was retrained chronologically at 11 origins. Across those folds, mean RMSE was 91.1869, mean empirical coverage was 94.76%, and mean interval width was 331.2871. These are research measurements, not universal performance guarantees." },
  { keys: ["next", "future", "build", "roadmap"], answer: "The next research steps are to compare all model families under the same rolling protocol, add calibration-versus-width plots, run ablations, repeat across additional domains, and freeze reproducible experiment configurations for the dissertation. The chatbot interface is currently a local explanatory layer; a live LLM service can be added after the experiment API is stable." },
  { keys: ["chatbot", "llm", "ai"], answer: "The current Copilot is intentionally rule-based and local. It explains the documented DeepStoch snapshot without sending your data to an external service. Later, we can connect an LLM through a controlled backend that receives only approved experiment summaries and cites the underlying metrics." }
];

function answerCopilot(question) {
  const normalized = question.toLowerCase();
  if (liveRunContext && (normalized.includes("current") || normalized.includes("selected") || normalized.includes("live") || normalized.includes("result") || normalized.includes("best") || normalized.includes("simulation"))) {
    const best = [...liveRunContext.models].sort((a, b) => Number(a.rmse) - Number(b.rmse))[0];
    const thresholdText = liveRunContext.simulation.probability_exceeds_threshold === null ? "No threshold was requested." : `The threshold-crossing probability is ${(liveRunContext.simulation.probability_exceeds_threshold * 100).toFixed(2)}%.`;
    return `For the current live run on ${liveRunContext.rows} rows, ${liveRunContext.models.length} models were compared. The lowest holdout RMSE is ${Number(best.rmse).toFixed(4)} from ${friendlyModelName(best.model)}. Empirical coverage is ${(liveRunContext.interval.empirical_coverage * 100).toFixed(2)}% at nominal ${(liveRunContext.interval.nominal_coverage * 100).toFixed(0)}%, with mean interval width ${Number(liveRunContext.interval.mean_width).toFixed(4)}. The simulation used ${liveRunContext.simulation.paths} paths over ${liveRunContext.simulation.horizon} steps. ${thresholdText}`;
  }
  const match = copilotAnswers.find(item => item.keys.some(key => normalized.includes(key)));
  return match ? match.answer : "DeepStoch is building a trustworthy forecasting workflow: compare models, quantify uncertainty, test calibration on future data, and document the evidence. Try asking about the purpose, uncertainty, neural model, real-world experiment, or next steps.";
}

function addChatMessage(role, text) {
  const messages = document.querySelector("#chat-messages");
  const bubble = document.createElement("div");
  bubble.className = `chat-message ${role}`;
  bubble.innerHTML = `<span class="chat-role">${role === "assistant" ? "DeepStoch Copilot" : "You"}</span><p>${text}</p>`;
  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;
}

function askCopilot(question) {
  const cleaned = question.trim();
  if (!cleaned) return;
  addChatMessage("user", cleaned);
  addChatMessage("assistant", answerCopilot(cleaned));
  document.querySelector("#chat-input").value = "";
}

const suggestions = ["What is the purpose of DeepStoch?", "Why do we need uncertainty?", "What do the real-world results mean?", "What should we build next?"];
document.querySelector("#chat-suggestions").innerHTML = suggestions.map(question => `<button type="button" data-question="${question}">${question}</button>`).join("");
document.querySelectorAll("#chat-suggestions button").forEach(button => button.addEventListener("click", () => askCopilot(button.dataset.question)));
document.querySelector("#chat-form").addEventListener("submit", event => { event.preventDefault(); askCopilot(document.querySelector("#chat-input").value); });
addChatMessage("assistant", "Welcome. I can explain what DeepStoch is building, why uncertainty matters, what the current real-world numbers mean, and what comes next.");

function friendlyModelName(name) {
  return {
    linear_regression: "Linear regression",
    random_forest: "Random Forest",
    linear_ml_plus_residual_state: "Linear ML + residual state",
    local_level_state_space: "Local-level state-space",
    local_linear_trend_state_space: "Local-linear-trend state-space",
    neural_state_space: "Neural state-space"
  }[name] || name.replaceAll("_", " ");
}

function renderModels(models) {
  document.querySelector("#model-table").innerHTML = models.map(model => `
    <tr><td><strong>${model.name}</strong></td><td>${model.mae === null ? "—" : Number(model.mae).toFixed(4)}</td><td>${model.rmse === null ? "—" : Number(model.rmse).toFixed(4)}</td><td>${model.mape === null ? "—" : Number(model.mape).toFixed(2) + "%"}</td><td>${model.coverage === null || model.coverage === undefined ? "—" : (typeof model.coverage === "number" ? (model.coverage * 100).toFixed(2) + "%" : model.coverage)}</td><td>${model.width === null || model.width === undefined ? "—" : Number(model.width).toFixed(4)}</td><td class="status">${model.status}</td></tr>
  `).join("");
}
renderModels(snapshot.models);

async function connectLocalApi() {
  const base = document.querySelector("#api-url").value.trim().replace(/\/$/, "");
  await runComparison(base);
}

async function runComparison(baseOverride = null) {
  const base = (baseOverride || document.querySelector("#api-url").value.trim()).replace(/\/$/, "");
  const status = document.querySelector("#api-status");
  const button = document.querySelector("#api-connect");
  const runButton = document.querySelector("#run-comparison");
  const runStatus = document.querySelector("#run-status");
  const path = document.querySelector("#dataset-path").value.trim();
  const target = document.querySelector("#target-column").value.trim() || "target";
  const time = document.querySelector("#time-column").value.trim() || "timestamp";
  const testSize = Number(document.querySelector("#test-size").value);
  const horizon = Number(document.querySelector("#simulation-horizon").value);
  const paths = Number(document.querySelector("#simulation-paths").value);
  const thresholdText = document.querySelector("#simulation-threshold").value.trim();
  const threshold = thresholdText === "" ? null : Number(thresholdText);
  if (!path || !Number.isInteger(testSize) || testSize < 1 || !Number.isInteger(horizon) || horizon < 1 || !Number.isInteger(paths) || paths < 1 || (threshold !== null && !Number.isFinite(threshold))) {
    runStatus.textContent = "Enter a dataset path and valid positive experiment values.";
    return;
  }
  button.disabled = true;
  runButton.disabled = true;
  status.textContent = "Connecting…";
  runStatus.textContent = "Validating dataset and running comparison…";
  try {
    const health = await fetch(`${base}/health`);
    if (!health.ok) throw new Error("health check failed");
    const summaryResponse = await fetch(`${base}/dataset/summary?path=${encodeURIComponent(path)}&target_column=${encodeURIComponent(target)}&time_column=${encodeURIComponent(time)}`);
    if (!summaryResponse.ok) throw new Error("dataset validation request failed");
    const summary = await summaryResponse.json();
    if (!summary.validation.valid) throw new Error(summary.validation.errors.join("; "));
    const response = await fetch(`${base}/models/compare?path=${encodeURIComponent(path)}&target_column=${encodeURIComponent(target)}&time_column=${encodeURIComponent(time)}&test_size=${testSize}`);
    if (!response.ok) throw new Error("comparison request failed");
    const payload = await response.json();
    const uncertaintyResponse = await fetch(`${base}/uncertainty/report?path=${encodeURIComponent(path)}&target_column=${encodeURIComponent(target)}&time_column=${encodeURIComponent(time)}&calibration_size=${Math.min(8, Math.max(1, testSize))}&test_size=${testSize}&coverage=0.9`);
    if (!uncertaintyResponse.ok) throw new Error("uncertainty request failed");
    const uncertaintyPayload = await uncertaintyResponse.json();
    const simulationQuery = `${base}/simulation/report?path=${encodeURIComponent(path)}&target_column=${encodeURIComponent(target)}&horizon=${horizon}&paths=${paths}&seed=42${threshold === null ? "" : `&threshold=${encodeURIComponent(threshold)}`}`;
    const simulationResponse = await fetch(simulationQuery);
    if (!simulationResponse.ok) throw new Error("simulation request failed");
    const simulationPayload = await simulationResponse.json();
    renderModels(payload.models.map(model => ({ ...model, name: friendlyModelName(model.model), width: model.interval_width, status: "Live API" })));
    const interval = uncertaintyPayload.interval;
    renderMetrics("uncertainty", [["Nominal coverage", `${(interval.nominal_coverage * 100).toFixed(0)}%`], ["Calibration size", `${interval.calibration_size} observations`], ["Empirical coverage", `${(interval.empirical_coverage * 100).toFixed(2)}%`], ["Mean interval width", Number(interval.mean_width).toFixed(4)]]);
    const simulation = simulationPayload.simulation;
    const finalMedian = simulation.quantiles["0.5"][simulation.horizon - 1];
    const finalLow = simulation.quantiles["0.1"][simulation.horizon - 1];
    const finalHigh = simulation.quantiles["0.9"][simulation.horizon - 1];
    const crossing = simulation.probability_exceeds_threshold === null ? "not requested" : `${(simulation.probability_exceeds_threshold * 100).toFixed(2)}%`;
    renderMetrics("simulation", [["Paths", String(simulation.paths)], ["Horizon", `${simulation.horizon} steps`], ["Median final value", Number(finalMedian).toFixed(2)], ["10%-90% final range", `${Number(finalLow).toFixed(2)}–${Number(finalHigh).toFixed(2)}`], ["Threshold probability", crossing]]);
    liveRunContext = { rows: summary.rows, models: payload.models, interval, simulation };
    document.querySelector("#data-mode").textContent = "Live API";
    status.textContent = "Connected";
    status.className = "api-status connected";
    const split = summary.split;
    document.querySelector("#run-status").textContent = "Comparison complete";
    document.querySelector("#run-summary").textContent = `${summary.rows} rows validated · ${split.train} train · ${split.validation} validation · ${split.test} test · ${payload.models.length} models · ${(interval.empirical_coverage * 100).toFixed(2)}% coverage · ${simulation.paths} simulated paths.`;
    addChatMessage("assistant", `Live run complete for ${summary.rows} rows. Ask me about the current best model, empirical coverage, interval width, or simulation result.`);
  } catch (error) {
    status.textContent = "Snapshot mode";
    status.className = "api-status error";
    runStatus.textContent = "Run failed";
    document.querySelector("#run-summary").textContent = `${error.message}. The snapshot remains visible.`;
  } finally {
    button.disabled = false;
    runButton.disabled = false;
  }
}
document.querySelector("#api-connect").addEventListener("click", connectLocalApi);
document.querySelector("#run-comparison").addEventListener("click", () => runComparison());

function renderMetrics(id, metrics) {
  document.querySelector(`#${id}`).innerHTML = metrics.map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join("");
}
renderMetrics("uncertainty", snapshot.uncertainty);
renderMetrics("simulation", snapshot.simulation);
renderMetrics("rolling-summary", snapshot.rolling.summary);
renderMetrics("real-world-summary", snapshot.realWorld.summary);
document.querySelector("#rolling-table").innerHTML = snapshot.rolling.folds.map((fold, index) => `
  <tr><td>${index + 1}</td><td>${fold[0]}</td><td>${fold[1].toFixed(4)}</td><td>${fold[2]}</td><td>${fold[3].toFixed(4)}</td></tr>
`).join("");
document.querySelector("#roadmap").innerHTML = snapshot.roadmap.map(([phase, title, status]) => `
  <div class="milestone ${status}"><b><span class="dot"></span>Phase ${phase}</b><small>${title}</small></div>
`).join("");

function renderCalibrationChart(target, calibration) {
  const width = 960;
  const height = 250;
  const left = 48;
  const right = 20;
  const top = 22;
  const bottom = 34;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const x = index => left + (index / Math.max(1, calibration.coverage.length - 1)) * plotWidth;
  const yCoverage = value => top + (1 - value) * plotHeight;
  const widthMin = Math.min(...calibration.widths);
  const widthMax = Math.max(...calibration.widths);
  const yWidth = value => top + plotHeight + 55 + (1 - (value - widthMin) / Math.max(1, widthMax - widthMin)) * 70;
  const coveragePoints = calibration.coverage.map((value, index) => `${x(index).toFixed(1)},${yCoverage(value).toFixed(1)}`).join(" ");
  const widthPoints = calibration.widths.map((value, index) => `${x(index).toFixed(1)},${yWidth(value).toFixed(1)}`).join(" ");
  const nominalY = yCoverage(calibration.nominal);
  target.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">
    <rect class="chart-bg" x="0" y="0" width="${width}" height="${height}" rx="12"/>
    <text class="chart-title" x="${left}" y="15">Coverage by fold</text>
    <line class="nominal-line" x1="${left}" y1="${nominalY}" x2="${width - right}" y2="${nominalY}"/>
    <polyline class="coverage-line" points="${coveragePoints}"/>
    <text class="chart-label" x="${width - right}" y="${nominalY - 6}" text-anchor="end">Nominal 90%</text>
    <text class="chart-title" x="${left}" y="${height - 8}">Mean interval width</text>
    <polyline class="width-line" points="${widthPoints}"/>
  </svg>`;
}
renderCalibrationChart(document.querySelector("#calibration-chart"), snapshot.realWorld);
