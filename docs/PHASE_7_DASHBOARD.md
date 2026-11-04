# Phase 7 — Research dashboard prototype

Phase 7 adds the first browser-facing dashboard for DeepStoch. It presents the
current reproducible demo snapshot as a research interface containing model
comparison, uncertainty calibration, simulation summaries, and roadmap status.

Open `dashboard/index.html` in a browser, or serve the repository with any
static HTTP server:

```powershell
python -m http.server 8000
```

Then open `http://localhost:8000/dashboard/`.

The current dashboard is deliberately a static prototype. It does not claim to
be connected to a live experiment database. The next dashboard milestone will
connect generated JSON reports, add forecast and interval charts, and expose
dataset/model controls. Phase 8 will then add hybrid neural state-space and
neural stochastic research modules.

The dashboard now also displays rolling-origin hybrid evaluation, including
aggregate RMSE variation, empirical coverage, interval width, and each temporal
fold. It is therefore an evidence viewer as well as a project-status prototype.

The model table now contains the unified comparison across baselines, classical
ML, state-space, hybrid, and neural models, including both point-error metrics
and uncertainty metrics where available.
