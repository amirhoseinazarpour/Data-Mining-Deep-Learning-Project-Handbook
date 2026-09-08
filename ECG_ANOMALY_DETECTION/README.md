# ECG5000 Anomaly Detection: Controlled Conv1D vs LSTM Autoencoders

A leakage-aware anomaly-detection study on the **ECG5000** time-series dataset. The project compares a **Conv1D Autoencoder** and an **LSTM Autoencoder** under a controlled experimental protocol: both models are trained only on normal ECG beats, score samples using reconstruction mean squared error (MSE), and are evaluated on the same untouched final test set.

> **Primary artifact:** `ecg_anomaly_detection_colab-1.ipynb`  
> **Runtime:** Google Colab or a local Python environment with TensorFlow.

## Overview

The task is to detect abnormal ECG beats from reconstruction error:

- **Normal class:** ECG5000 class `1`
- **Anomalous classes:** ECG5000 classes `2`–`5`
- **Training protocol:** Train autoencoders exclusively on normal beats.
- **Anomaly score:** Per-beat reconstruction MSE across the 140 time points.
- **Core comparison:** Conv1D AE versus LSTM AE under matched data splits, preprocessing, training-only-on-normal policy, optimizer family, reconstruction loss, early-stopping policy, and threshold-calibration data.

The notebook is deliberately designed as a controlled comparison—not as a claim that either architecture is universally best for ECG anomaly detection.

## Research Question

**Do convolutional local waveform features or recurrent temporal representations provide the better speed–quality trade-off for reconstruction-based anomaly detection on ECG5000?**

- The **Conv1D Autoencoder** uses convolutional filters to encode local, shift-tolerant waveform motifs.
- The **LSTM Autoencoder** uses recurrent processing to encode the ordered temporal context of each beat.

The final conclusion should be based on the generated final-test table: F1 under an identical threshold rule, precision, recall, Average Precision, parameter count, training time, anomaly-type recall, and failure cases.

## Experimental Protocol

The notebook follows a leakage-aware evaluation design:

1. Loads and combines the ECG5000 archive splits.
2. Creates a reproducible, stratified **20% final test split**. This split is held out from model selection and threshold calibration.
3. Splits the remaining development set into a labeled **calibration split** and a model-development split, stratified across the five original ECG classes.
4. Keeps class-1 beats only from model-development data and divides them into normal training and normal validation subsets.
5. Fits feature-wise standardization using **only normal training beats**.
6. Trains each Autoencoder using only normalized normal training beats.
7. Performs early stopping using normal-validation reconstruction loss only.
8. Selects anomaly thresholds from non-test data.
9. Evaluates once on the untouched final test set.

This separation matters: neither final-test labels nor final-test reconstruction scores are used to choose a model, a training checkpoint, or a threshold.

## Dataset

The project loads **ECG5000** through the [`aeon`](https://www.aeon-toolkit.org/) time-series toolkit; no Kaggle API key is required.

| Property | Value |
|---|---|
| Dataset | ECG5000 |
| Input | Univariate ECG beat time series |
| Sequence length | 140 time points |
| Original labels | 5 classes |
| Normal class | 1 |
| Anomaly classes | 2–5 |
| Task | Binary normal-versus-anomaly detection |

The notebook maps the original classes to the following interpretation:

| Class | Interpretation |
|---:|---|
| 1 | Normal |
| 2 | R-on-T PVC |
| 3 | PVC |
| 4 | SP/EB |
| 5 | Unclassified |

## Models

Both models receive a beat with shape `(140, 1)` and minimize reconstruction MSE.

| Model | Representation hypothesis |
|---|---|
| Conv1D Autoencoder | Short local waveform motifs can distinguish normal from anomalous morphology efficiently. |
| LSTM Autoencoder | Sequential dependencies across the full beat improve reconstruction-based discrimination. |

The notebook exposes `FAST_MODE` for a practical Colab execution path. In this mode, model width and training duration are reduced; disable it for a more computationally intensive run.

## Threshold Strategies

Each model receives two independently calibrated thresholds:

| Rule | Calibration source | Interpretation |
|---|---|---|
| **Normal-validation P95** | Scores of held-out normal validation beats | Threshold at the 95th percentile of normal reconstruction error; does not require anomaly labels. |
| **Labeled-calibration F1 optimum** | Separate labeled calibration split | Threshold that maximizes F1 with representative normal and anomalous examples. |

The P95 rule is relevant when labeled anomalies are unavailable. The F1-optimal rule can be useful when representative labeled anomalies are available, but it may be sensitive to calibration-set prevalence and anomaly composition.

## Evaluation

### Classification metrics

The final-test evaluation includes:

- Precision
- Recall / Sensitivity
- F1-score
- Specificity / True Negative Rate
- Negative Predictive Value (NPV)
- Average Precision
- Confusion matrices
- Precision–Recall curves

### Diagnostic analysis

The notebook also provides:

- Final normal-versus-anomaly reconstruction-error distributions.
- Per-anomaly-class detection performance for classes 2–5.
- False-positive and false-negative waveform inspection.
- Runtime and parameter-count comparison for Conv1D and LSTM models.

### Reference baselines

As additional reference models—not as part of the required neural comparison—the notebook fits:

- One-Class SVM with RBF kernel
- Isolation Forest

Both baselines are trained on the same normalized normal-training data and use F1-optimized thresholds from the same labeled calibration split.

## Installation

### Google Colab

Open the notebook in Colab and run the installation cell:

```python
!pip install aeon
```

The notebook installs or imports the remaining dependencies through the Colab environment.

### Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install aeon tensorflow numpy pandas matplotlib seaborn scikit-learn
pip install jupyter
```

## Usage

Start Jupyter and run the notebook sequentially:

```bash
jupyter lab
```

Then open:

```text
ecg_anomaly_detection_colab-1.ipynb
```

### Recommended execution order

1. Install `aeon`.
2. Run the configuration and reproducibility cell.
3. Load ECG5000 and inspect the exploratory plots.
4. Create the leakage-safe train/validation/calibration/final-test splits.
5. Build and train the Conv1D and LSTM Autoencoders.
6. Calibrate the P95 and F1-optimal thresholds without using final-test data.
7. Run final-test classification, per-class, and failure-case analyses.
8. Optionally run One-Class SVM and Isolation Forest baselines.
9. Export the reproducibility metadata, model artifacts, scores, and ZIP archive.

## Outputs

The notebook creates artifacts such as:

```text
artifacts/
├── Conv1D_AE_best.keras
├── LSTM_AE_best.keras
├── config.json
├── scaler_statistics.npz
├── final_test_scores.csv
├── results.csv
└── ecg5000_anomaly_detection_artifacts.zip
```

Exact output filenames may vary slightly with notebook configuration.

## Reproducibility

The notebook fixes random seeds across Python, NumPy, and TensorFlow, records the experiment configuration, saves scaler statistics, persists model checkpoints, and exports final prediction scores and result tables.

For a quick Colab run, keep:

```python
FAST_MODE = True
```

For a more thorough run, set:

```python
FAST_MODE = False
```

and expect longer training time.

## Limitations

- ECG5000 consists of short, pre-segmented, preprocessed beats; results may not transfer to continuous, noisy, multi-lead, or differently sampled clinical ECG streams.
- Reconstruction error is not guaranteed to be higher for all anomalies. A sufficiently expressive autoencoder can reconstruct some abnormal beats well, producing false negatives.
- F1 is prevalence-dependent and weights precision and recall equally. Real clinical or operational deployments may require sensitivity constraints, cost-sensitive thresholding, calibrated alert rates, or patient-level metrics.
- Patient identifiers are not available in the dataset. The beat-level stratified split cannot rule out patient-level information leakage if multiple beats came from the same patient.
- A single split and random seed do not quantify uncertainty. Repeated grouped splits, bootstrapping, and external validation would provide stronger evidence.
- This repository is an educational/research implementation and is **not** a clinical decision-support system.

## Technologies

`Python` · `TensorFlow / Keras` · `aeon` · `NumPy` · `Pandas` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `Google Colab` · `Jupyter`

## Citation

If you use the ECG5000 dataset, cite its original source and follow the relevant dataset and toolkit terms. This repository does not redistribute the dataset.

## License

Add a license appropriate for your intended use before public release (for example, MIT for permissive code reuse).
