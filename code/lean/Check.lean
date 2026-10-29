import Model
import Stability
import Consistency

/-! Prints the axioms of the theorems behind Theorem 1 of the paper.
Run with `lake env lean Check.lean` after `lake build`.
Every line must list only `propext`, `Classical.choice`, and `Quot.sound`. -/

-- Theorem 1 (iii): the AoII chain under a schedule
#print axioms Model.expected_area
#print axioms Model.piW_stationary
#print axioms Model.meanAoII_eq
#print axioms Model.meanAoII_strictAnti
#print axioms Model.meanAoII_reactive
#print axioms Model.meanAoII_background_le
#print axioms Model.reactive_ge_background

-- Theorem 1 (i): the posterior means and the posterior-best schedule
#print axioms Stability.eventually_unique_max
#print axioms Consistency.post_ae_tendsto
#print axioms Consistency.greedy_ae_eventually

-- Theorem 1 (ii): the posterior samples
#print axioms Consistency.scale_tendsto
#print axioms Consistency.noise_tendsto_zero
#print axioms Consistency.sample_suboptimal_tendsto_zero
