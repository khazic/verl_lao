# Scoring-Based Credit Assignment

Per-turn advantages for long-horizon multi-turn RL, read off the policy by
re-scoring collected trajectories instead of spending extra rollouts (tree/branching
Monte Carlo) or training an auxiliary process reward model.

Two readouts, both pure teacher-forced log-prob evaluation:

- **Ablation credit (backward)**: `c_t = logp(a_final | ctx) - logp(a_final | ctx \ turn_t)`.
  How strongly the final decision depended on turn `t`. Normalized into weights
  `w_t` that reallocate the group-centered outcome reward across turns.
- **Answer-probe potential (forward)**: `Phi_t = (1/|y*|) logp(y* | ctx_<=t, probe)`.
  Whether the target has become more immediately emittable.

Combined per-turn advantage: `A_t = T * w_t * (R - R_bar) + beta * (Phi_t - Phi_{t-1})`.

Neither quantity is environment-level causal credit, and neither identity that
they satisfy makes the resulting update an unbiased policy gradient. Phase A
exists to find out how far the readouts are from what they are meant to measure,
**before** any training compute is spent.

## Phase A: offline validation (this directory)

Standalone scoring against a frozen policy. No trainer, no environment, no
gradient updates. One GPU is enough.

```
python -m examples.scoring_credit.offline_score --model /path/to/model --data trajs.jsonl --out runs/offline --tp 1
```

Useful flags:

| Flag | Why |
|---|---|
| `--dry-run` | Reports request count and token volume, then stops. Always run this first on a new dataset. |
| `--modes delete replace` | Scores both interventions in one pass; they share the unablated prefix, so this is much cheaper than two runs. |
| `--controls control_length` | Length-matched control. Subtracting it separates information loss from discontinuity and volume. |
| `--check-backend-agreement 64` | Re-scores the first 64 requests with an eager HF forward pass and reports the largest disagreement. Run once per model before trusting a go/no-go. |
| `--shapley-permutations 8` | Monte Carlo Shapley credit. Off by default; this is the only expensive option. |
| `--include-final-probe` | Off by default on purpose, see the note below. |

### Input schema

One JSON object per line:

```json
{
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "y_star": "forty two",
  "success": true,
  "gold_turns": [2],
  "pair_id": "q17",
  "variant": "restated",
  "origin_turn": 2,
  "restatement_turn": 5
}
```

`messages`, `y_star` and `success` are required. A turn is one assistant message
plus every message that follows it until the next assistant message, so tool
results and observations stay attached to the action that produced them
regardless of the role your scaffold gives them.

The optional fields drive specific studies:

- `gold_turns`: annotated decisive turns, for precision@k / MRR localization.
- `pair_id` / `variant` / `origin_turn` / `restatement_turn`: the
  provenance-versus-restatement gate. Emit two lines per prompt with the same
  `pair_id`, one `"base"` and one `"restated"`, differing only in whether a later
  turn restates the decisive information without adding anything.

### Two traps that silently invalidate results

**Do not probe past the final action.** `--include-final-probe` puts the answer
itself into the probe context, which makes `Phi` trivially high on successes and
inflates every separability number computed from it. The default stops probing at
the boundary before the final turn.

**Verify the backend before believing a go/no-go.** vLLM's prompt-logprob path and
the trainer's `compute_log_prob` are different code. If they disagree, an offline
decision does not transfer to training. `--check-backend-agreement` measures the
gap once so you can stop worrying about it.

### The scoring noise floor is not negligible

Measured on Qwen3-0.6B over the example trajectories, as the largest absolute
difference in mean log-prob per token:

| comparison | max abs difference |
|---|---|
| HF bfloat16 vs HF float32 | 0.236 |
| vLLM vs HF bfloat16 | 0.213 |
| vLLM vs HF float32 | 0.105 |
| vLLM prefix caching on vs off | 0.075 |

Three things follow. Reference scoring runs in float32 by default, since a
bfloat16 reference is noisier than the disagreement it is supposed to
adjudicate, and vLLM turns out to sit closer to float32 than a naive bfloat16
forward pass does. Prefix caching perturbs the numbers slightly; it stays on
because it is what makes the ablation variants affordable, but it is not free.
Most importantly, every credit `c_t` must be read against this floor: on a small
model, differences of a few tenths of a nat per token are indistinguishable from
arithmetic. Measure the floor for your own model with
`--check-backend-agreement` before interpreting any credit ranking, and prefer
summed over per-token quantities where the target span is long.

### What the summary reports

`<out>.summary.json` uses keys named after the paper's table columns:

- `estimators.<variant>.provenance`: `provenance` (credit peaks at the originating
  turn), `origin_beats_restatement`, `migration` (share pulled away from the
  origin by the restatement).
- `estimators.<variant>.localization`: `p_at_1`, `p_at_3`, `mrr` against `gold_turns`.
- `estimators.<variant>.confounds`: correlation of credit with turn length and
  position, `r2_explained_by_length_and_position`, and
  `partial_rho_with_gold_turns`, which is agreement after those are removed.
  A `null` partial correlation means the confounds explain the credit entirely.
- `estimators.<variant>.agreement_with_probe`: rank correlation between `c_t` and
  `Phi_t - Phi_{t-1}`. A diagnostic, not a validation: both come from one policy.
- `probe`: `auroc_total_rise`, `auroc_max_jump`, `fraction_below_floor`,
  `probe_ensemble_sd`, and `phi_quantiles` for picking a floor.

Variants are reported separately for each ablation mode, for each
`mode-minus-control` combination, and for `shapley` when enabled.

### Go / no-go

Run the gate on both task families, since the probe is predicted to behave very
differently on tasks whose progress converts into answer likelihood and on tasks
whose progress accumulates in the environment.

- If `provenance` is low and `migration` is high, credit is tracking salience
  rather than provenance; the reallocation term needs Shapley or a different
  intervention before it is worth training on.
- If `partial_rho_with_gold_turns` collapses relative to `rho_with_gold_turns`,
  the credit was mostly turn length and position.
- If `auroc_total_rise` is near 0.5 on a family, the probe carries no signal
  there and `beta` stays at 0 for it.

## Phase B: training integration (not in this branch)

1. `scoring.py`: build ablated / probe variants from a rollout batch and score
   them through the actor's `compute_log_prob` path between rollout and update.
2. Advantage estimator `scoring_credit` registered with
   `verl.trainer.ppo.core_algos.register_adv_est`, assembling `A_t` and mapping it
   onto each turn's response tokens.
3. Config under `algorithm.scoring_credit`: `kappa`, `beta`, `beta_warmup_steps`,
   `phi_floor`, `probe_ensemble`, `ablation_mode`, `terminal_potential`,
   `signed_credits`, and `token_split`.

Two of these need to exist from the first training run rather than being added
later:

- `token_split` (`broadcast` | `per_token`). Turn-level mass conservation is not
  token-level mass conservation. Broadcasting `A_t` to every token of a turn makes
  the token-level total depend on `sum_t n_t w_t`, which drifts from the uniform
  baseline exactly when credit correlates with turn length. `per_token` divides by
  the turn's token count to restore it.
- `max_t w_t` in the training metrics. Concentration and mass conservation
  together force large advantages: as `kappa` falls, the selected turn approaches
  `T * (R - R_bar)`, which is `T` times the GRPO scalar. Logging the maximum weight
  is what distinguishes a temperature that broke optimization from attribution that
  was simply wrong.

## Tests

```
python -m pytest examples/scoring_credit/tests/test_scoring_credit_on_cpu.py -q
```

CPU only, no model required. The scoring backend is faked so the true dependence
structure is known exactly, which lets the tests assert that each metric detects
the failure mode it exists to catch: a restatement capturing credit from the turn
that originated the information, credit that is really just turn length, and a
control that cancels pure discontinuity cost.
