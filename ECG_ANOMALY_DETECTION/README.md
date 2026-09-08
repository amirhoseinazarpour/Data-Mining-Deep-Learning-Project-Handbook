# RNN-1 — Multivariate Weather Forecasting

**Authors:** Sobhan Moghimi, Amirhossein Azarpour

Data Mining, Deep Learning (Dr. Hadi Farahani, Spring 2026). Core project, 25 points.

**Files in this submission**

- `weather_forecasting_colab.ipynb` — code
- the PDF report — write-up
- this README

Given the past **72 hourly readings** from the Jena Climate station, forecast the next **12 hours** of:

- temperature `T (degC)` (°C)
- relative humidity `rh (%)` (percentage points)

## Original contribution

**Horizon-specific additive attention** over LSTM encoder states. Each forecast lead (1–12 h) has its own learned query. Hypothesis, stated before training: attention should help most at longer horizons by selecting recent trends and daily lags (24 / 48 / 72 h), not at 1 h where persistence is already strong.

**Result of the reported run.** Direct GRU is the best model. Attention is worse at 1–6 h and only catches up at 12 h (slightly best on 12 h humidity RMSE). The hypothesis is mostly rejected. Full numbers and discussion are in the PDF report.

| Model | Params | Test NMAE ↓ | Test T MAE 1 h / 12 h |
| --- | ---: | ---: | ---: |
| Persistence | 0 | 0.514 | 0.643 / 4.188 °C |
| Direct LSTM | 23,064 | 0.252 | 0.662 / 1.836 °C |
| **Direct GRU** | **17,880** | **0.243** | **0.553 / 1.783 °C** |
| Seq2Seq LSTM | 54,658 | 0.250 | 0.643 / 1.842 °C |
| Horizon Attention LSTM | 47,170 | 0.255 | 0.833 / 1.806 °C |

NMAE is mean MAE divided by each target’s training standard deviation. It is only a ranking score; °C and humidity points are never averaged as a physical-unit error.

The notebook is the source of truth. Do not edit results by hand; re-run it.

## How to reproduce

1. Open `weather_forecasting_colab.ipynb` in Google Colab.
2. Runtime → Change runtime type → **GPU**.
3. Leave `FAST_MODE = False` and `SEED = 2026` (these are the reported settings).
4. Run all cells top to bottom. The Jena dataset is downloaded automatically from the official Keras URL.

The notebook writes `/content/jena_rnn_outputs/` and offers `/content/jena_rnn_outputs.zip` for download. Figures used in the PDF are generated there.

Full mode takes on the order of **5–10 minutes** on a Colab GPU (four models, up to 30 epochs, early stopping). The reported run used TensorFlow 2.20.0.

For a smoke test only, set `FAST_MODE = True` (stride 3, 32 units, 10 epochs). Do not treat FAST numbers as the course result.

## Protocol (reported run)

- Seed `2026` (`random`, NumPy, TensorFlow); deterministic ops requested
- Chronological 70 / 15 / 15 split; scaler fit on training rows only
- Lookback 72 h, horizon 12 h, stride 1, 19 input features
- Shared training: Adam `1e-3`, batch 128, dropout 0.1, 64 units, MSE on standardized targets, early stopping on `val_loss` (patience 6)
- Metrics: per-target, per-horizon MAE and RMSE after inverse transform
- Model selection for the interval stretch uses **validation NMAE only**

## What the notebook writes

After a full run, `jena_rnn_outputs/` contains:

**Tables**

- `forecast_metrics.csv` — MAE / RMSE by model, split, target, horizon
- `model_comparison.csv`, `metrics_aggregate_normalized.csv`
- `training_summary.csv`, `split_summary.csv`
- `interval_metrics.csv` — 90% validation-residual coverage and width
- `attention_diagnostics.csv`, `error_vs_change.csv`, `failure_case_index.csv`

**Figures** (same plots as in the PDF)

- `figures/data_overview.png`
- `figures/training_curves.png`
- `figures/horizon_mae.png`, `figures/horizon_rmse.png`
- `figures/prediction_examples.png`
- `figures/attention_lookback_summary.png`, `figures/attention_heatmaps.png`
- `figures/error_vs_weather_change.png`, `figures/failure_cases.png`
- `figures/prediction_intervals.png`

## Data

Jena Climate, Max Planck Institute for Biogeochemistry, via the Keras weather-forecasting example:

https://keras.io/examples/timeseries/timeseries_weather_forecasting/
