# Real-world experiment 001 — UCI Appliances Energy Prediction

## Dataset

The source is UCI's Appliances Energy Prediction dataset. The downloaded file
was obtained from the dataset author's linked public repository, which UCI
identifies as the associated data repository. The UCI record remains the
authoritative source for attribution, dataset description, DOI, and licence.

The converted local file contains 19,735 chronological rows with:

```text
timestamp, target, T1, RH_1, T2, RH_2, T_out, RH_out, Windspeed, Visibility
```

The canonical mapping is `date -> timestamp` and `Appliances -> target`.

## Validation

```text
Rows: 19,735
Validation status: ok
Errors: 0
Warnings: 0
```

## Holdout protocol

The final 1,974 observations were used as the chronological test block. No
random shuffling was used. The demo neural checkpoint was excluded because it
was trained on a different dataset.

## Initial comparison

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Linear regression | 27.5744 | 61.2531 | 24.5687% |
| Linear ML + residual state | 27.0742 | 61.3412 | not reported by hybrid interface |
| Random Forest | 30.5107 | 66.1631 | 27.5373% |
| Historical mean | 52.9499 | 93.9461 | 56.9851% |
| Moving average | 46.2715 | 100.7713 | 31.4941% |
| Local level state-space | 46.3359 | 101.2918 | not reported by hybrid interface |
| Last value | 46.4235 | 102.0271 | 30.0765% |
| Seasonal naive | 47.3050 | 102.5189 | 30.9315% |

These are initial single-holdout results, not final dissertation conclusions.
The hybrid intervals were extremely wide, so coverage must be considered
together with interval width. The next experiment is rolling-origin evaluation
on this real-world series, followed by training a dataset-specific neural
checkpoint before comparing neural results.
