# TF-1: Transformer Text Classification — Data-Efficiency Study

## What this does

Fine-tunes **DistilBERT** on **AG News** (4-class topic classification) at six
training-set sizes — 1%, 5%, 10%, 25%, 50%, 100% — and compares it at each
size against a **TF-IDF + Logistic Regression** baseline trained on the same
subset.

**Hypothesis (the original contribution):** a pretrained Transformer needs
far less labeled data than a classical bag-of-words baseline to reach
comparable accuracy, because its representations already encode general
language structure from pretraining. The data-efficiency curve is the
evidence for or against this.

## How to run

**As a script:**
```bash
pip install -r requirements.txt
python tf1_text_classification.py
```

**In Colab / Jupyter:** just run the cell/file as-is. There are no
command-line flags — the script uses plain config variables near the top
(`EPOCHS`, `SEED`, `FRACTIONS`) instead of `argparse`, because `argparse`
breaks inside notebook kernels (it collides with the kernel's own `-f
...json` argument). Edit those variables directly if you want to change
anything.

Optional: for a quick smoke test on fewer fractions (much faster), edit
`FRACTIONS` near the top of the file to:
```python
FRACTIONS = [0.01, 0.1, 1.0]
```

A GPU is strongly recommended — the 100% fraction alone is ~120k training
examples. On CPU only, use the smoke-test config above.

## Outputs (all in `results/`)

| File | What it's for |
|---|---|
| `data_efficiency_results.json` | Raw accuracy/macro-F1 for TF-IDF and DistilBERT at every fraction |
| `data_efficiency_curve.png` | The plot — put this straight in the report |
| `error_analysis.txt` | 15 misclassified examples from the full-data DistilBERT model, for the qualitative error section |
| `per_class_report_full_model.txt` | Per-class precision/recall/F1 (AG News is roughly balanced, but report this anyway per the handbook's instructions) |

## Notes on fairness of the comparison

- At small fractions (≤5000 examples) DistilBERT gets more epochs
  (`epochs * 3`, capped at 15) so it isn't handicapped relative to the
  100%-data run — few-shot fine-tuning normally needs more passes over a
  small set to converge. This is called out explicitly in the code
  (`effective_epochs` in `run_transformer`) so it's easy to defend in the
  report.
- Both models see the exact same subsample at each fraction (same seed),
  so the comparison at each point is apples-to-apples.
- Same fixed seed (`--seed`, default 42) everywhere for reproducibility.

## Mapping to the report outline

1. **Problem & data** — AG News, 4-class topic classification, from
   Hugging Face Hub.
2. **Method** — DistilBERT via `transformers`, TF-IDF + Logistic Regression
   baseline via `scikit-learn`.
3. **Original contribution** — the data-efficiency sweep (this is the
   graded core; state the hypothesis before showing the curve).
4. **Experiments** — the six fractions, both models, same seed/splits.
5. **Results & analysis** — `data_efficiency_curve.png` +
   `per_class_report_full_model.txt`; discuss where the gap between the two
   models is largest/smallest and why.
6. **Discussion & observation** — answer the two handbook questions
   (convolution vs. attention) separately; this script doesn't cover them,
   they're conceptual.
7. **Limitations** — e.g. AG News is a relatively easy, clean dataset where
   even TF-IDF does well; the gap you find here may not generalize to
   noisier text (GoEmotions/Jigsaw).

## Stretch ideas (optional, for extra bonus)

- Swap in GoEmotions or Jigsaw and rerun — noisier/imbalanced labels should
  widen the gap between TF-IDF and DistilBERT more than on AG News.
- Add a calibration analysis (reliability diagram) on the full-data model.
- Wrap the full-data model in a 5-line Gradio app for the demo bonus.
