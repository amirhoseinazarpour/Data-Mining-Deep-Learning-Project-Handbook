# RL-1: Steering GPT-2 with PPO (RLHF-lite)

**Bonus Challenge · 25 points · does not count toward the required 100 or the breadth rule**

Sobhan Moghimi & Amirhossein Azarpoor

## What this does

Fine-tunes GPT-2 (`lvwerra/gpt2-imdb`, already LM-finetuned on IMDB) with PPO so its
continuations of movie-review prompts are steered toward positive sentiment, using
a DistilBERT sentiment classifier (`lvwerra/distilbert-imdb`) as the reward signal —
a small, self-contained version of RLHF.

**Original contribution:** trains two runs — **with** and **without** a KL-to-reference
penalty — and compares reward climb against fluency (perplexity) in each, to directly
study reward hacking: does the policy learn to game the classifier at the cost of
producing fluent text?

## Why this notebook doesn't use `trl.PPOTrainer` directly

Modern Hugging Face TRL (≥0.12) replaced the classic `PPOTrainer.step(queries,
responses, rewards)` loop with a `Trainer.train()`-style API that scores a
**causal-LM** reward model on the same token ids as the policy. That's incompatible
with `lvwerra/distilbert-imdb` (a BERT-family classifier with its own tokenizer) —
which is exactly the reward model this assignment calls for. So the notebook uses
TRL's `AutoModelForCausalLMWithValueHead` building block plus a compact, from-scratch
PPO update (clipped surrogate objective + GAE + value loss + KL-shaped reward) that
matches the algorithm from the original TRL GPT-2-sentiment tutorial. This is
explained in the first markdown cell of the notebook and is worth mentioning in the
report if anyone asks why the code doesn't look like a stock `trl` example.

## How to run

Open the notebook in Colab (GPU runtime) or Kaggle and run top to bottom.

1. **Cell 1** installs/upgrades `trl`, `transformers`, `datasets`, `accelerate`,
   `peft`. If this cell upgrades any package, **restart the runtime** and re-run from
   the imports cell — stale package versions are the most common source of import
   errors here.
2. **Section 1 (Data)** builds ~6,000 IMDB prompts, each truncated to 8–16 tokens
   (matches the original TRL tutorial setup).
3. **Section 2 (Models and reward)** loads the policy (GPT-2 + value head), the
   frozen reference copy, and the DistilBERT sentiment pipeline used as the reward.
4. **Section 3** trains two independent runs via `run_ppo(use_kl_penalty=...)`:
   - `model_kl, logs_kl = run_ppo(use_kl_penalty=True)` — `kl_coef=0.2` (TRL default)
   - `model_nokl, logs_nokl = run_ppo(use_kl_penalty=False)` — `kl_coef=0.0`

   Default budget is 60 PPO steps × batch size 16, sized for a single Colab GPU. If
   you hit an out-of-memory error, drop `batch_size` to 8 in the `run_ppo(...)` calls.
5. **Section 4** scores generations from both models with perplexity **under the
   frozen base GPT-2** (not the tuned model), so it's a fair, independent fluency
   check, and prints reward vs. perplexity for both runs side by side.
6. **Section 5** prints qualitative before/after continuations for 5 prompts, +KL vs
   -KL, so you can eyeball the reward-hacking pattern in raw text.
7. **Section 6 (Discussion & observation)** answers both handbook discussion
   questions inline — already written into the notebook.

## Key result (from the included run)

| | Final reward (mean POSITIVE logit) | Perplexity (fluency) |
|---|---|---|
| **With KL penalty** (`kl_coef=0.2`) | 1.671 | 89.3 |
| **Without KL penalty** (`kl_coef=0.0`) | 0.639 (lower) | 225.1 (much worse) |

This is the reward-hacking signature the experiment was designed to catch: removing
the KL penalty doesn't even reach a higher reward here — it reaches a *lower* reward
while fluency collapses, meaning the unconstrained policy wandered off the
distribution of natural language without even winning on the metric it was
optimizing. Worth double-checking and discussing this exact asymmetry in the report,
since naively one might expect no-KL to at least win on raw reward.

## Outputs to capture for the report

The notebook doesn't save files to disk by default — everything is inline. For your
report/repo submission, capture:
- The **reward-vs-step and KL-vs-step plot** from Section 4 (`fig` object) — save it
  with `fig.savefig("results/reward_kl_curves.png", dpi=150)` before/instead of
  `plt.show()`.
- The **printed reward/perplexity summary line** from Section 4.
- A few of the **qualitative before/after examples** from Section 5 (already printed;
  copy 2–3 representative pairs into the report, ideally one clean example and one
  visibly degenerate/repetitive one from the no-KL model).
- The **Discussion & observation** markdown cell (Section 6) — already written.

## Notes on reproducibility

- Fixed seed (`SEED = 42`) for `random`, `numpy`, and `torch`.
- `WANDB_DISABLED` and HF Hub progress bars are turned off for clean notebook output.
- Both PPO runs (`+KL` and `-KL`) train independent, freshly-initialized policies from
  the same `lvwerra/gpt2-imdb` checkpoint and see the same dataloader shuffling seed,
  so the only difference between the two runs is `kl_coef`.

## Mapping to the report outline

1. **Problem & data** — IMDB prompts, GPT-2 policy, DistilBERT reward model.
2. **Method** — PPO with value head, clipped surrogate + GAE, KL-shaped reward.
3. **Original contribution** — the with-KL vs. no-KL comparison; state the reward-
   hacking hypothesis before showing the reward/perplexity numbers.
4. **Experiments** — the two `run_ppo` calls, same seed/data, only `kl_coef` differs.
5. **Results & analysis** — reward/KL curves (Section 4 plot), the fluency table
   above, qualitative examples (Section 5).
6. **Discussion & observation** — already answered in Section 6 of the notebook;
   copy directly into the report.
7. **Limitations** — 60 steps is a short PPO run for a bonus challenge; note that
   longer training might change the magnitude (though not likely the direction) of
   the reward-hacking effect. Also note the reward is a single DistilBERT classifier
   with its own blind spots — relevant context for the Q2 discussion answer already
   in the notebook.

## Stretch idea (optional, per the handbook)

Train a small reward model from preference pairs you label yourself instead of using
the off-the-shelf DistilBERT classifier — the handbook lists this as the suggested
stretch goal for RL-1.
