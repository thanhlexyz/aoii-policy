# Convergence of the learned retransmission schedule, in Lean 4

Machine-checked proofs behind Theorem 1 (Section III-D, "Consistency of the
Learned Schedule") of the paper. The paper states the theorem in
mathematical notation only; this folder is the check behind it.

## Build

Lean `v4.33.1` and Mathlib `v4.33.1` through the shared package cache
`~/.cache/lake/packages` (see `lakefile.toml`, as in `~/Project/lean4aoi`).

    lake build                  # all three files
    lake env lean Check.lean    # prints the axioms of every theorem
    make proof                  # both, from code/

Every `#print axioms` line must list only `propext`, `Classical.choice`
and `Quot.sound`: no `sorryAx`.

## Files and what they prove

| File | Paper | Content |
|---|---|---|
| `Model.lean` | Section III-A, III-B, Theorem 1 (iii) | The two-state AoII chain of one device: the expected AoII area of a geometric sojourn (`expected_area`, via the binomial series), the stationary law (`piW_stationary`), the closed form eq. (aoii-model) as the renewal-reward product (`meanAoII_eq`), its strict decrease in the recovery rate (`meanAoII_strictAnti`), the recovery rate eq. (repair) of a schedule and its bounds (`recovery_ge_send`, `recovery_ge_reactive`, `recovery_mono`), the reactive closed form (`meanAoII_reactive`), the bound under a per-slot transmission probability `x > 0` (`meanAoII_background_le`) and the `x² (1 - p_ℓ) / p_E²` factor between them (`reactive_ge_background`). |
| `Stability.lean` | Theorem 1 (i), deterministic part | Finitely many scores converging to means with a unique maximizer have that maximizer eventually (`eventually_unique_max`); the gap is positive (`exists_gap`); scores within half the gap keep the maximizer (`lt_of_close`). |
| `Consistency.lean` | Theorem 1 (i) and (ii) | Under `Rewards.Stationary` (i.i.d. rewards per arm, every arm credited infinitely often), the posterior mean of every arm converges almost surely (`post_ae_tendsto`, strong law of large numbers along the random number of credited device-rounds) and the posterior-best arm is the optimal one for all but finitely many rounds (`greedy_ae_eventually`). Under `Sampling` (posterior mean plus a vanishing scale times a standardized draw), the noise vanishes in probability (`noise_tendsto_zero`, Chebyshev) and the probability of sampling a suboptimal arm tends to zero (`sample_suboptimal_tendsto_zero`). `scale_tendsto` shows that a bounded noise estimate over `√n` is such a scale. |

## What is modelled and what is proved

- The renewal-reward identity, mean AoII = sojourn rate × expected area,
  is the modelling step of Section III-A; `Model.lean` proves everything
  after it.
- The traffic assumption of the paper is `Consistency.Rewards.Stationary`:
  measurable, integrable, pairwise independent and identically distributed
  rewards credited to each arm, and every arm credited infinitely often.
  The last clause is a hypothesis here; Thompson sampling meets it because a
  Gaussian posterior sample gives every arm a positive probability in every
  round.
- The posterior sample is modelled as `post + c * Z` with `c → 0` almost
  surely and `Z` of mean zero and variance at most one
  (`Consistency.Sampling`); no independence between `c`, `Z` and the history
  is needed, only a union bound.
- Nothing is proved about the rate of convergence or the regret.
