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

## Dataset-specific neural state-space run

A separate probabilistic GRU checkpoint was trained using only this UCI
dataset. The checkpoint was not reused from the synthetic demonstration
dataset.

```text
Training sequences:       15,780
Validation sequences:      3,946
Initial validation NLL:   1.564926
Final validation NLL:     1.166559
Best validation NLL:      1.046790
```

Evaluation on the same final 1,974-observation chronological test block:

| Model | MAE | RMSE | MAPE | Empirical coverage | Mean interval width |
|---|---:|---:|---:|---:|---:|
| Dataset-specific neural state-space | 42.1574 | 89.7262 | 36.8310% | 95.74% | 317.9312 |

The nominal interval level was 90%. Empirical coverage was therefore above
the nominal level on this holdout, while the interval was substantially
narrower than the initial ML-plus-residual-state interval. The neural point
forecast did not outperform the linear regression or hybrid point forecast
on this first holdout. These results are evidence of a working
dataset-specific pipeline, not a final model-selection claim; rolling-origin
neural evaluation and further calibration are required.

## Rolling-origin neural evaluation

To test temporal stability, the neural model was retrained independently at
each origin using the following configuration:

```text
Sequence length: 6
Minimum training rows: 12,000
Test block per fold: 672
Step between origins: 672
Training epochs: 8
Early-stopping patience: 3
Folds: 11
Nominal coverage: 90%
```

Aggregate results across all 11 chronological folds:

| Metric | Result |
|---|---:|
| Mean RMSE | 91.1869 |
| RMSE standard deviation | 23.5358 |
| Mean empirical coverage | 94.7646% |
| Mean interval width | 331.2871 |

The rolling result provides stronger evidence than a single holdout that the
probabilistic intervals are generally conservative on this series. The RMSE
variation across folds also shows that performance changes with the temporal
regime. This is an initial rolling experiment: repeated seeds, fold-level
calibration plots, comparisons against rolling classical baselines, and
additional datasets are still needed before drawing broad conclusions.
