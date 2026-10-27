import Mathlib

/-! # Stability of a finite argmax under converging scores

The base station assigns each device the schedule with the largest score,
where the score of an arm is its posterior mean (when the learner is frozen)
or a posterior sample (while it learns). This file proves the deterministic
heart of the convergence theorem: when the scores of finitely many arms
converge to means with a unique maximizer, the maximizer of the scores is
that arm from some round on; and quantitatively, as soon as every score is
within half the gap of its mean.
-/

open Filter Finset
open scoped Topology

namespace Stability

variable {A : Type*}

/-- Scores converging arm by arm to means with a unique maximizer `a₀` have
`a₀` as their unique maximizer from some index on. -/
theorem eventually_unique_max [Finite A] {μ : A → ℝ} {s : ℕ → A → ℝ} {a₀ : A}
    (hgap : ∀ a, a ≠ a₀ → μ a < μ a₀)
    (hconv : ∀ a, Tendsto (fun t => s t a) atTop (𝓝 (μ a))) :
    ∀ᶠ t in atTop, ∀ a, a ≠ a₀ → s t a < s t a₀ := by
  rw [eventually_all]
  intro a
  by_cases h : a = a₀
  · exact Eventually.of_forall fun _ h' => absurd h h'
  · exact ((hconv a).eventually_lt (hconv a₀) (hgap a h)).mono fun _ h' _ => h'

/-- The gap between the best mean and the runner-up is positive. -/
theorem exists_gap [Fintype A] [DecidableEq A] {μ : A → ℝ} {a₀ : A}
    (hgap : ∀ a, a ≠ a₀ → μ a < μ a₀) :
    ∃ Δ : ℝ, 0 < Δ ∧ ∀ a, a ≠ a₀ → μ a + Δ ≤ μ a₀ := by
  by_cases hne : (univ.erase a₀).Nonempty
  · obtain ⟨b, hb, hmin⟩ := exists_min_image (univ.erase a₀) (fun a => μ a₀ - μ a) hne
    refine ⟨μ a₀ - μ b, ?_, fun a ha => ?_⟩
    · have := hgap b (mem_erase.1 hb).1; linarith
    · have := hmin a (mem_erase.2 ⟨ha, mem_univ a⟩); linarith
  · refine ⟨1, one_pos, fun a ha => ?_⟩
    exact absurd ⟨a, mem_erase.2 ⟨ha, mem_univ a⟩⟩ hne

/-- Scores within `Δ / 2` of means separated by at least `Δ` keep the
maximizer. -/
theorem lt_of_close {μ s : A → ℝ} {a₀ : A} {Δ : ℝ}
    (hgap : ∀ a, a ≠ a₀ → μ a + Δ ≤ μ a₀)
    (hclose : ∀ a, |s a - μ a| < Δ / 2) :
    ∀ a, a ≠ a₀ → s a < s a₀ := by
  intro a ha
  have h1 := abs_lt.1 (hclose a)
  have h2 := abs_lt.1 (hclose a₀)
  have := hgap a ha
  linarith [h1.2, h2.1]

end Stability
