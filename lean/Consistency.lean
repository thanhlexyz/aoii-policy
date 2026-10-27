import Mathlib
import Stability

/-! # Consistency of the learned retransmission schedule

Section III-C and III-D of the paper. The base station credits every device-round
reward to the schedule (arm) the device played, keeps a Gaussian posterior
per arm with the arm's mean reward and a scale that shrinks with the number
of credited device-rounds, and assigns each device the arm with the largest
posterior sample. Under stationary traffic the rewards credited to an arm
are independent and identically distributed, and this file proves:

* `post_ae_tendsto`: the posterior mean of every arm converges almost
  surely to the expected reward of that schedule (strong law of large
  numbers along the random number of credited device-rounds);
* `greedy_ae_eventually`: almost surely, the posterior-best schedule is the
  reward-optimal one for all but finitely many rounds;
* `sample_suboptimal_tendsto_zero`: the probability that a posterior sample
  assigns a device a suboptimal schedule tends to zero (Chebyshev on the
  standardized draw, convergence in probability of the posterior mean and
  scale).

The assumption that every arm is credited infinitely often is a hypothesis
here (`Stationary.explore`); Gaussian posterior samples give every arm a
positive probability in every round, which is why Thompson sampling meets
it.
-/

open MeasureTheory ProbabilityTheory Filter Finset
open scoped Topology ENNReal

set_option linter.unusedSectionVars false

namespace Consistency

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
variable {A : Type*} [Fintype A] [DecidableEq A]

/-- Rewards credited to the arms, in the order they are observed: `Y a i` is
the `i`-th device-round reward credited to arm `a`, and `n a t` the number of
device-rounds credited to `a` after `t` rounds. -/
structure Rewards (A Ω : Type*) where
  Y : A → ℕ → Ω → ℝ
  n : A → ℕ → Ω → ℕ

/-- Stationary traffic: the rewards credited to one arm are integrable,
pairwise independent and identically distributed, and every arm is credited
infinitely often. -/
structure Rewards.Stationary (R : Rewards A Ω) (P : Measure Ω) : Prop where
  meas : ∀ a i, Measurable (R.Y a i)
  int : ∀ a, Integrable (R.Y a 0) P
  indep : ∀ a, Pairwise (Function.onFun (· ⟂ᵢ[P] ·) (R.Y a))
  ident : ∀ a i, IdentDistrib (R.Y a i) (R.Y a 0) P P
  n_meas : ∀ a t, Measurable (R.n a t)
  explore : ∀ a, ∀ᵐ ω ∂P, Tendsto (fun t => R.n a t ω) atTop atTop

/-- Expected reward of a schedule. -/
noncomputable def Rewards.mean (R : Rewards A Ω) (P : Measure Ω) (a : A) : ℝ :=
  ∫ ω, R.Y a 0 ω ∂P

/-- Posterior mean of arm `a` after `t` rounds: the average of the credited
rewards. -/
noncomputable def Rewards.post (R : Rewards A Ω) (a : A) (t : ℕ) (ω : Ω) : ℝ :=
  ((R.n a t ω : ℕ) : ℝ)⁻¹ * ∑ i ∈ range (R.n a t ω), R.Y a i ω

variable {R : Rewards A Ω}

/-! ## The posterior means -/

/-- Almost surely, every posterior mean converges to the expected reward of
its schedule. -/
theorem post_ae_tendsto (hR : R.Stationary P) :
    ∀ᵐ ω ∂P, ∀ a, Tendsto (fun t => R.post a t ω) atTop (𝓝 (R.mean P a)) := by
  rw [ae_all_iff]
  intro a
  filter_upwards [strong_law_ae (R.Y a) (hR.int a) (hR.indep a) (hR.ident a), hR.explore a]
    with ω h1 h2
  have := h1.comp h2
  simpa [Function.comp_def, smul_eq_mul, Rewards.post, Rewards.mean] using this

/-- Almost surely, the posterior-best schedule is the reward-optimal one for
all but finitely many rounds. -/
theorem greedy_ae_eventually (hR : R.Stationary P) {a₀ : A}
    (hgap : ∀ a, a ≠ a₀ → R.mean P a < R.mean P a₀) :
    ∀ᵐ ω ∂P, ∀ᶠ t in atTop, ∀ a, a ≠ a₀ → R.post a t ω < R.post a₀ t ω := by
  filter_upwards [post_ae_tendsto hR] with ω hω
  exact Stability.eventually_unique_max hgap hω

lemma post_measurable (hR : R.Stationary P) (a : A) (t : ℕ) : Measurable (R.post a t) := by
  have hF : Measurable (fun p : Ω × ℕ => ((p.2 : ℕ) : ℝ)⁻¹ * ∑ i ∈ range p.2, R.Y a i p.1) := by
    apply measurable_from_prod_countable_left
    intro m
    show Measurable fun x : Ω => ((m : ℕ) : ℝ)⁻¹ * ∑ i ∈ range m, R.Y a i x
    exact (Finset.measurable_sum _ fun i _ => hR.meas a i).const_mul _
  exact hF.comp (measurable_id.prodMk (hR.n_meas a t))

/-- The posterior mean converges to the expected reward in probability. -/
theorem post_tendstoInMeasure (hR : R.Stationary P) (a : A) :
    TendstoInMeasure P (fun t => R.post a t) atTop (fun _ => R.mean P a) :=
  tendstoInMeasure_of_tendsto_ae (fun t => (post_measurable hR a t).aestronglyMeasurable)
    ((post_ae_tendsto hR).mono fun _ h => h a)

/-! ## The posterior samples -/

/-- Posterior sampling: the score of arm `a` for one device in round `t` is
the posterior mean plus the posterior scale `c a t` times a standardized draw
`Z a t`. The scale vanishes almost surely and the draw has mean zero and
variance at most one. -/
structure Sampling (R : Rewards A Ω) (P : Measure Ω) where
  c : A → ℕ → Ω → ℝ
  Z : A → ℕ → Ω → ℝ
  c_meas : ∀ a t, Measurable (c a t)
  c_tendsto : ∀ a, ∀ᵐ ω ∂P, Tendsto (fun t => c a t ω) atTop (𝓝 0)
  Z_memLp : ∀ a t, MemLp (Z a t) 2 P
  Z_mean : ∀ a t, ∫ ω, Z a t ω ∂P = 0
  Z_var : ∀ a t, variance (Z a t) P ≤ 1

/-- The posterior sample of arm `a` for one device in round `t`. -/
noncomputable def Sampling.score (S : Sampling R P) (a : A) (t : ℕ) (ω : Ω) : ℝ :=
  R.post a t ω + S.c a t ω * S.Z a t ω

/-- A posterior scale that is a bounded noise estimate over the square root
of the number of credited device-rounds vanishes almost surely. -/
theorem scale_tendsto (hR : R.Stationary P) {σ : ℕ → Ω → ℝ} {B : ℝ}
    (hσ : ∀ t ω, 0 ≤ σ t ω) (hB : ∀ t ω, σ t ω ≤ B) (a : A) :
    ∀ᵐ ω ∂P, Tendsto (fun t => σ t ω / Real.sqrt (R.n a t ω)) atTop (𝓝 0) := by
  filter_upwards [hR.explore a] with ω hn
  have hsqrt : Tendsto (fun t => Real.sqrt (R.n a t ω)) atTop atTop :=
    Real.tendsto_sqrt_atTop.comp (tendsto_natCast_atTop_atTop.comp hn)
  have hinv : Tendsto (fun t => (Real.sqrt (R.n a t ω))⁻¹) atTop (𝓝 0) :=
    tendsto_inv_atTop_zero.comp hsqrt
  have hbound : Tendsto (fun t => B * (Real.sqrt (R.n a t ω))⁻¹) atTop (𝓝 0) := by
    simpa using hinv.const_mul B
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hbound ?_ ?_
  · intro t
    exact div_nonneg (hσ t ω) (Real.sqrt_nonneg _)
  · intro t
    show σ t ω / Real.sqrt (R.n a t ω) ≤ B * (Real.sqrt (R.n a t ω))⁻¹
    rw [div_eq_mul_inv]
    exact mul_le_mul_of_nonneg_right (hB t ω) (inv_nonneg.2 (Real.sqrt_nonneg _))

omit [IsProbabilityMeasure P] in
/-- Convergence in probability of a real sequence, in the form used below. -/
private lemma tendsto_measure_abs_of_tendstoInMeasure {f : ℕ → Ω → ℝ} {g : Ω → ℝ}
    (h : TendstoInMeasure P f atTop g) {ε : ℝ} (hε : 0 < ε) :
    Tendsto (fun t => P {ω | ε ≤ |f t ω - g ω|}) atTop (𝓝 0) := by
  have := h (ENNReal.ofReal ε) (ENNReal.ofReal_pos.2 hε)
  have hset : ∀ t, {ω | ENNReal.ofReal ε ≤ edist (f t ω) (g ω)} = {ω | ε ≤ |f t ω - g ω|} := by
    intro t
    ext ω
    simp only [Set.mem_ofPred_eq, edist_dist, Real.dist_eq]
    rw [ENNReal.ofReal_le_ofReal_iff (abs_nonneg _)]
  simpa only [hset] using this

/-- The noise term `c Z` vanishes in probability: Chebyshev on the draw and
convergence in probability of the scale. -/
theorem noise_tendsto_zero (S : Sampling R P) (a : A) {ε : ℝ} (hε : 0 < ε) :
    Tendsto (fun t => P {ω | ε ≤ |S.c a t ω * S.Z a t ω|}) atTop (𝓝 0) := by
  rw [ENNReal.tendsto_nhds_zero]
  intro η hη
  rcases eq_or_ne η ⊤ with hηtop | hηtop
  · exact Eventually.of_forall fun _ => hηtop ▸ le_top
  -- split the budget: half for the scale, half for the draw
  have hη2 : 0 < η / 2 := ENNReal.half_pos hη.ne'
  have hη2top : η / 2 ≠ ⊤ := ENNReal.div_ne_top hηtop two_ne_zero
  set e : ℝ := (η / 2).toReal with he
  have he_pos : 0 < e := ENNReal.toReal_pos hη2.ne' hη2top
  set s : ℝ := Real.sqrt e with hs
  have hs_pos : 0 < s := Real.sqrt_pos.2 he_pos
  -- thresholds: `γ` for the scale, `κ` for the draw, with `γ κ = ε`
  set γ : ℝ := ε * s with hγ
  set κ : ℝ := 1 / s with hκ
  have hγ_pos : 0 < γ := mul_pos hε hs_pos
  have hκ_pos : 0 < κ := by positivity
  have hγκ : γ * κ = ε := by rw [hγ, hκ]; field_simp
  -- Chebyshev: the draw exceeds `κ` with probability at most `η / 2`
  have hZ : ∀ t, P {ω | κ ≤ |S.Z a t ω|} ≤ η / 2 := by
    intro t
    have h := meas_ge_le_variance_div_sq (S.Z_memLp a t) hκ_pos
    simp only [S.Z_mean a t, sub_zero] at h
    refine h.trans ?_
    have h1 : variance (S.Z a t) P / κ ^ 2 ≤ e := by
      calc variance (S.Z a t) P / κ ^ 2 ≤ 1 / κ ^ 2 :=
            div_le_div_of_nonneg_right (S.Z_var a t) (by positivity)
        _ = e := by rw [hκ]; field_simp; rw [hs, Real.sq_sqrt he_pos.le]
    calc ENNReal.ofReal (variance (S.Z a t) P / κ ^ 2) ≤ ENNReal.ofReal e :=
          ENNReal.ofReal_le_ofReal h1
      _ = η / 2 := ENNReal.ofReal_toReal hη2top
  -- the scale exceeds `γ` with probability at most `η / 2`, eventually
  have hc : TendstoInMeasure P (fun t => S.c a t) atTop (fun _ => 0) :=
    tendstoInMeasure_of_tendsto_ae (fun t => (S.c_meas a t).aestronglyMeasurable) (S.c_tendsto a)
  have hc' := tendsto_measure_abs_of_tendstoInMeasure hc hγ_pos
  simp only [sub_zero] at hc'
  filter_upwards [ENNReal.tendsto_nhds_zero.1 hc' (η / 2) hη2] with t ht
  -- `ε ≤ |c Z|` forces `γ ≤ |c|` or `κ ≤ |Z|`
  have hsub : {ω | ε ≤ |S.c a t ω * S.Z a t ω|}
      ⊆ {ω | γ ≤ |S.c a t ω|} ∪ {ω | κ ≤ |S.Z a t ω|} := by
    intro ω hω
    by_contra hcon
    simp only [Set.mem_union, Set.mem_ofPred_eq, not_or, not_le] at hcon
    have : |S.c a t ω * S.Z a t ω| < ε := by
      rw [abs_mul, ← hγκ]
      exact mul_lt_mul'' hcon.1 hcon.2 (abs_nonneg _) (abs_nonneg _)
    exact absurd hω (not_le.2 this)
  calc P {ω | ε ≤ |S.c a t ω * S.Z a t ω|}
      ≤ P ({ω | γ ≤ |S.c a t ω|} ∪ {ω | κ ≤ |S.Z a t ω|}) := measure_mono hsub
    _ ≤ P {ω | γ ≤ |S.c a t ω|} + P {ω | κ ≤ |S.Z a t ω|} := measure_union_le _ _
    _ ≤ η / 2 + η / 2 := add_le_add ht (hZ t)
    _ = η := ENNReal.add_halves η

/-- The probability that a posterior sample assigns a device a suboptimal
schedule tends to zero. -/
theorem sample_suboptimal_tendsto_zero (hR : R.Stationary P) (S : Sampling R P) {a₀ : A}
    (hgap : ∀ a, a ≠ a₀ → R.mean P a < R.mean P a₀) :
    Tendsto (fun t => P {ω | ∃ a, a ≠ a₀ ∧ S.score a₀ t ω ≤ S.score a t ω}) atTop (𝓝 0) := by
  obtain ⟨Δ, hΔ, hgapΔ⟩ := Stability.exists_gap hgap
  have hΔ4 : 0 < Δ / 4 := by positivity
  -- the two ways a sample can be off by `Δ / 2`: the posterior mean or the noise
  set E₁ : A → ℕ → Set Ω := fun a t => {ω | Δ / 4 ≤ |R.post a t ω - R.mean P a|} with hE₁
  set E₂ : A → ℕ → Set Ω := fun a t => {ω | Δ / 4 ≤ |S.c a t ω * S.Z a t ω|} with hE₂
  have hsub : ∀ t, {ω | ∃ a, a ≠ a₀ ∧ S.score a₀ t ω ≤ S.score a t ω}
      ⊆ (⋃ a, E₁ a t) ∪ (⋃ a, E₂ a t) := by
    intro t ω hω
    by_contra hcon
    simp only [Set.mem_union, Set.mem_iUnion, hE₁, hE₂, Set.mem_ofPred_eq, not_or, not_exists,
      not_le] at hcon
    have hclose : ∀ a, |S.score a t ω - R.mean P a| < Δ / 2 := by
      intro a
      have h1 := hcon.1 a
      have h2 := hcon.2 a
      calc |S.score a t ω - R.mean P a|
          = |(R.post a t ω - R.mean P a) + S.c a t ω * S.Z a t ω| := by
            unfold Sampling.score; ring_nf
        _ ≤ |R.post a t ω - R.mean P a| + |S.c a t ω * S.Z a t ω| := abs_add_le _ _
        _ < Δ / 4 + Δ / 4 := add_lt_add h1 h2
        _ = Δ / 2 := by ring
    obtain ⟨a, ha, hle⟩ := hω
    exact absurd hle (not_le.2 (Stability.lt_of_close hgapΔ hclose a ha))
  have hbound : ∀ t, P {ω | ∃ a, a ≠ a₀ ∧ S.score a₀ t ω ≤ S.score a t ω}
      ≤ ∑ a, P (E₁ a t) + ∑ a, P (E₂ a t) := by
    intro t
    calc P {ω | ∃ a, a ≠ a₀ ∧ S.score a₀ t ω ≤ S.score a t ω}
        ≤ P ((⋃ a, E₁ a t) ∪ (⋃ a, E₂ a t)) := measure_mono (hsub t)
      _ ≤ P (⋃ a, E₁ a t) + P (⋃ a, E₂ a t) := measure_union_le _ _
      _ ≤ ∑ a, P (E₁ a t) + ∑ a, P (E₂ a t) :=
          add_le_add (measure_iUnion_fintype_le _ _) (measure_iUnion_fintype_le _ _)
  have h₁ : ∀ a, Tendsto (fun t => P (E₁ a t)) atTop (𝓝 0) := fun a =>
    tendsto_measure_abs_of_tendstoInMeasure (post_tendstoInMeasure hR a) hΔ4
  have h₂ : ∀ a, Tendsto (fun t => P (E₂ a t)) atTop (𝓝 0) := fun a =>
    noise_tendsto_zero S a hΔ4
  have hsum : Tendsto (fun t => ∑ a, P (E₁ a t) + ∑ a, P (E₂ a t)) atTop (𝓝 0) := by
    have := (tendsto_finsetSum univ fun a _ => h₁ a).add (tendsto_finsetSum univ fun a _ => h₂ a)
    simpa using this
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hsum
    (fun _ => zero_le) hbound

end Consistency
