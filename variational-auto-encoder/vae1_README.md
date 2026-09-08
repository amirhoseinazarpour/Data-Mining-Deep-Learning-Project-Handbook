# VAE-1: Generative Modeling and Latent-Space Exploration

## What this does

Trains a convolutional VAE on **Fashion-MNIST**, then:
1. Shows reconstructions, random samples from the prior, and an
   interpolation between two real images' encodings.
2. Visualizes the latent space with a 2-D PCA projection colored by class.
3. **Original contribution — quantitative attribute manipulation:** trains a
   small independent CNN classifier ("oracle"), computes per-class mean
   latent vectors, defines attribute-transfer directions
   (`mean(target_class) - mean(source_class)`), and moves real latents
   along that direction at increasing strength. The oracle then scores
   what fraction of the decoded images are classified as the target class,
   giving a quantitative "flip rate vs. manipulation strength" curve —
   not just eyeballed samples.

**Hypothesis being tested:** linear directions in the VAE's latent space
correspond to semantically meaningful, class-level attributes, so moving
along `mean(target) - mean(source)` should smoothly shift the decoder's
output toward the target class in a way an independent classifier can
measure.

Transfer pairs tested by default: Sneaker→Ankle boot, T-shirt/top→Shirt,
Pullover→Coat (chosen because each pair is visually related but distinct).

## How to run

**As a script:**
```bash
pip install -r requirements.txt
python vae1_generative_latent.py
```

**In Colab / Jupyter:** just run the cell/file as-is. There are no
command-line flags — everything is controlled by the config variables near
the top (`LATENT_DIM`, `VAE_EPOCHS`, `TRANSFER_PAIRS`, `ALPHAS`, etc.),
following the same pattern as the TF-1 script so there's no `argparse`
conflict with the notebook kernel.

Fashion-MNIST downloads automatically via `torchvision.datasets` on first
run. A GPU speeds things up but this is small enough to also run on CPU
(20 epochs on Fashion-MNIST with this architecture is a few minutes on
either).

## Outputs (all in `results/`)

| File | What it's for |
|---|---|
| `vae_reconstructions.png` | Real vs. reconstructed images — sanity check the VAE learned anything |
| `vae_random_samples.png` | 64 samples drawn from the prior `N(0, I)` |
| `vae_interpolation.png` | Interpolation between two real images' latent encodings |
| `latent_space_pca.png` | 2-D PCA scatter of the latent space, colored by class — this is the "visualize the latent space" deliverable |
| `attribute_manipulation_*.png` | Qualitative grid per transfer pair: the same example image at each alpha |
| `manipulation_success_curve.png` | **The core evidence for the original contribution** — oracle flip-rate vs. alpha, one line per transfer pair |
| `metrics.json` | Everything numeric: full ELBO training history, oracle accuracy, and all flip-rate numbers |

## Notes on the design

- **KL warmup (`KL_WARMUP_EPOCHS`):** beta ramps linearly from 0 to 1 over
  the first few epochs instead of being fixed at 1 from the start. Without
  this, vanilla VAEs on image data commonly suffer "posterior collapse"
  early in training (the KL term dominates before the decoder has learned
  anything, so it just ignores the latent code). This is worth mentioning
  in the report if you compare with/without warmup as a mini-ablation.
- **`LATENT_DIM = 20`:** big enough for reasonable sample quality (a
  2-D latent bottleneck alone tends to blur too much on Fashion-MNIST),
  while still projectable to 2-D via PCA for the required visualization.
- **The oracle classifier is deliberately separate from the VAE** — it's
  only there to give you an objective, automated judge for the
  manipulation experiment, so "does attribute transfer work" isn't a
  question you're answering by eye.
- Same fixed seed (`SEED = 42`) everywhere for reproducibility.

## Mapping to the report outline

1. **Problem & data** — Fashion-MNIST, generative modeling + latent-space
   structure.
2. **Method** — convolutional VAE architecture, KL warmup schedule.
3. **Original contribution** — the attribute-manipulation study; state the
   hypothesis before showing `manipulation_success_curve.png`.
4. **Experiments** — ELBO curves (`metrics.json`), sample quality,
   interpolation, the three transfer pairs.
5. **Results & analysis** — discuss which transfer pairs get the highest
   flip rate and why (e.g. Sneaker→Ankle boot are more visually similar
   than T-shirt→Shirt, so expect an easier flip); look at where the
   qualitative grids and the quantitative curve agree or disagree.
6. **Discussion & observation** — answer the two handbook questions
   (interpolation smoothness / KL term geometry; alternative priors) —
   this script doesn't cover them, they're conceptual.
7. **Limitations** — PCA is a linear projection and can hide non-linear
   structure in a 20-D latent space; the oracle classifier itself has
   <100% accuracy so some "non-flips" may be oracle errors, not manipulation
   failures — worth flagging with the oracle's own test accuracy from
   `metrics.json`.

## Stretch ideas (optional, for extra bonus)

- Quantify sample quality with FID instead of eyeballing
  `vae_random_samples.png` (handbook's suggested stretch for this project).
- Sweep `LATENT_DIM` (e.g. 2, 10, 20, 50) and plot final test ELBO and
  flip-rate quality against it.
- Add a second interpolation figure specifically between two *different*
  classes (e.g. Sneaker → Ankle boot) to visually complement the
  quantitative manipulation curve.
