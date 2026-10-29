import Mathlib

/-! # The AoII chain of one device under a retransmission schedule

Section III-A and III-B of the paper: a device leaves the correct state when a
move (probability `pE` per slot) fires a trigger that is lost (probability
`pℓ`), and returns from the erroneous state at a recovery rate `r` that
collects every event refreshing the base-station estimate. The sojourn in
the erroneous state is geometric of mean `1/r`, the AoII grows by the mean
mismatch `δ` in every slot of it, and the expected per-slot AoII is
eq. (aoii-model): `δ e / (r (e + r))` with `e = pE pℓ`.

The modelling step is the renewal-reward identity: the per-slot mean AoII
is the rate of sojourns times the expected AoII area of one sojourn. This
file proves everything after that step: the expected area, the stationary
law of the chain, the closed form, its monotonicity in `r`, the reactive and
background special cases, and the recovery rate eq. (repair) of a schedule.
-/

open Filter Finset
open scoped Topology

namespace Model

noncomputable section

/-- Per-slot traffic statistics of one device, held fixed over the horizon. -/
structure Traffic where
  /-- probability per slot that the source moves and fires a trigger -/
  pE : ℝ
  /-- probability that a packet is lost -/
  pℓ : ℝ
  /-- mean mismatch while the base station is in error -/
  δ : ℝ
  pE_pos : 0 < pE
  pE_le : pE ≤ 1
  pℓ_nonneg : 0 ≤ pℓ
  pℓ_lt : pℓ < 1
  δ_pos : 0 < δ

/-- Probability per slot of entering the erroneous state: a move whose trigger
is lost. -/
def Traffic.e (T : Traffic) : ℝ := T.pE * T.pℓ

lemma Traffic.e_nonneg (T : Traffic) : 0 ≤ T.e := mul_nonneg T.pE_pos.le T.pℓ_nonneg

/-- The retransmission schedule `θ = (q, n_r, x_bg)` of eq. (schedule). -/
structure Schedule where
  q : ℝ
  nr : ℕ
  bg : ℝ
  q_nonneg : 0 ≤ q
  q_le : q ≤ 1
  bg_nonneg : 0 ≤ bg
  bg_le : bg ≤ 1

/-- Transmission probability `π_θ(m)` after `m` re-sends since the latest
trigger: `q` while the repeat budget lasts, `x_bg` after it. -/
def Schedule.pi (θ : Schedule) (m : ℕ) : ℝ := if m < θ.nr then θ.q else θ.bg

lemma Schedule.pi_nonneg (θ : Schedule) (m : ℕ) : 0 ≤ θ.pi m := by
  unfold Schedule.pi; split_ifs <;> [exact θ.q_nonneg; exact θ.bg_nonneg]

lemma Schedule.pi_le_one (θ : Schedule) (m : ℕ) : θ.pi m ≤ 1 := by
  unfold Schedule.pi; split_ifs <;> [exact θ.q_le; exact θ.bg_le]

/-- Recovery rate eq. (repair) when the device transmits with probability `x`:
the base station is refreshed by a decoded re-send or by the decoded trigger
of the next move. -/
def recovery (T : Traffic) (x : ℝ) : ℝ :=
  1 - (1 - x * (1 - T.pℓ)) * (1 - T.pE * (1 - T.pℓ))

/-- Mean per-slot AoII of eq. (aoii-model). -/
def meanAoII (δ e r : ℝ) : ℝ := δ * e / (r * (e + r))

/-! ## The expected AoII area of one sojourn -/

/-- A sojourn of `w + 1` slots accumulates `(w+1)(w+2)/2` units of AoII per
unit of mismatch; under a geometric sojourn of rate `r` its expectation is
`1 / r²`. -/
theorem expected_area {r : ℝ} (hr : 0 < r) (hr1 : r ≤ 1) :
    ∑' w : ℕ, r * (1 - r) ^ w * (((w : ℝ) + 1) * ((w : ℝ) + 2) / 2) = 1 / r ^ 2 := by
  have hx : ‖(1 - r)‖ < 1 := by
    rw [Real.norm_eq_abs, abs_lt]; constructor <;> linarith
  have h := tsum_choose_mul_geometric_of_norm_lt_one 2 hx
  have hc : ∀ w : ℕ, (((w + 2).choose 2 : ℕ) : ℝ) = ((w : ℝ) + 1) * ((w : ℝ) + 2) / 2 := by
    intro w
    rw [Nat.cast_choose_two]; push_cast; ring
  have hsum : ∑' w : ℕ, r * (1 - r) ^ w * (((w : ℝ) + 1) * ((w : ℝ) + 2) / 2)
      = r * ∑' w : ℕ, (((w + 2).choose 2 : ℕ) : ℝ) * (1 - r) ^ w := by
    rw [← tsum_mul_left]
    congr 1; ext w; rw [hc]; ring
  rw [hsum, h]
  have : (1 - (1 - r)) = r := by ring
  rw [this]; field_simp

/-! ## The stationary law of the two-state chain -/

/-- Stationary probability of the erroneous state. -/
def piW (e r : ℝ) : ℝ := e / (e + r)

/-- `(1 - πW, πW)` is invariant under the chain with entry rate `e` and
recovery rate `r`. -/
theorem piW_stationary {e r : ℝ} (he : 0 ≤ e) (hr : 0 < r) :
    (1 - piW e r) * e + piW e r * (1 - r) = piW e r := by
  unfold piW
  have : e + r ≠ 0 := by linarith
  field_simp; ring

/-- The rate of sojourns, `(1 - πW) e`, equals `e r / (e + r)`. -/
theorem sojourn_rate {e r : ℝ} (he : 0 ≤ e) (hr : 0 < r) :
    (1 - piW e r) * e = e * r / (e + r) := by
  unfold piW
  have : e + r ≠ 0 := by linarith
  field_simp; ring

/-- Eq. (aoii-model): the renewal-reward product of the sojourn rate, the
mean mismatch and the expected area is the closed form. -/
theorem meanAoII_eq {δ e r : ℝ} (he : 0 ≤ e) (hr : 0 < r) :
    δ * ((1 - piW e r) * e) * (1 / r ^ 2) = meanAoII δ e r := by
  rw [sojourn_rate he hr]
  unfold meanAoII
  have : e + r ≠ 0 := by linarith
  field_simp

/-! ## Properties of the closed form -/

/-- The mean AoII decreases strictly in the recovery rate. -/
theorem meanAoII_strictAnti {δ e r₁ r₂ : ℝ} (hδ : 0 < δ) (he : 0 < e)
    (hr₁ : 0 < r₁) (h : r₁ < r₂) : meanAoII δ e r₂ < meanAoII δ e r₁ := by
  unfold meanAoII
  have hr₂ : 0 < r₂ := hr₁.trans h
  apply div_lt_div_of_pos_left (by positivity)
  · positivity
  · nlinarith

/-- The mean AoII is at most `δ e / r²`. -/
theorem meanAoII_le {δ e r : ℝ} (hδ : 0 < δ) (he : 0 ≤ e) (hr : 0 < r) :
    meanAoII δ e r ≤ δ * e / r ^ 2 := by
  unfold meanAoII
  apply div_le_div_of_nonneg_left (by positivity) (by positivity)
  nlinarith

/-! ## Recovery under a schedule -/

lemma recovery_eq (T : Traffic) (x : ℝ) :
    recovery T x = x * (1 - T.pℓ) + T.pE * (1 - T.pℓ) - x * (1 - T.pℓ) * (T.pE * (1 - T.pℓ)) := by
  unfold recovery; ring

/-- A purely reactive device, `x = 0`, recovers only at the next move. -/
theorem recovery_reactive (T : Traffic) : recovery T 0 = T.pE * (1 - T.pℓ) := by
  unfold recovery; ring

/-- The recovery rate is at least the rate of decoded re-sends. -/
theorem recovery_ge_send (T : Traffic) {x : ℝ} (_hx : 0 ≤ x) (hx1 : x ≤ 1) :
    x * (1 - T.pℓ) ≤ recovery T x := by
  rw [recovery_eq]
  have h1 : 0 ≤ 1 - T.pℓ := by linarith [T.pℓ_lt]
  have h1' : 1 - T.pℓ ≤ 1 := by linarith [T.pℓ_nonneg]
  have h2 : 0 ≤ T.pE * (1 - T.pℓ) := mul_nonneg T.pE_pos.le h1
  have h4 : x * (1 - T.pℓ) ≤ 1 := by
    have := mul_le_mul hx1 h1' h1 zero_le_one; linarith
  nlinarith [mul_nonneg h2 (sub_nonneg.2 h4)]

/-- The recovery rate is at least the reactive one. -/
theorem recovery_ge_reactive (T : Traffic) {x : ℝ} (hx : 0 ≤ x) (_hx1 : x ≤ 1) :
    T.pE * (1 - T.pℓ) ≤ recovery T x := by
  rw [recovery_eq]
  have h1 : 0 ≤ 1 - T.pℓ := by linarith [T.pℓ_lt]
  have h3 : T.pE * (1 - T.pℓ) ≤ 1 := by nlinarith [T.pE_le, T.pℓ_nonneg]
  nlinarith [mul_nonneg (mul_nonneg hx h1) (sub_nonneg.2 h3)]

theorem recovery_pos (T : Traffic) {x : ℝ} (hx : 0 ≤ x) (hx1 : x ≤ 1) : 0 < recovery T x :=
  lt_of_lt_of_le (mul_pos T.pE_pos (by linarith [T.pℓ_lt])) (recovery_ge_reactive T hx hx1)

/-- The recovery rate grows with the transmission probability. -/
theorem recovery_mono (T : Traffic) {x y : ℝ} (hxy : x ≤ y) : recovery T x ≤ recovery T y := by
  rw [recovery_eq, recovery_eq]
  have h1 : 0 ≤ 1 - T.pℓ := by linarith [T.pℓ_lt]
  have h3 : T.pE * (1 - T.pℓ) ≤ 1 := by nlinarith [T.pE_le, T.pℓ_nonneg]
  have h4 : 0 ≤ T.pE * (1 - T.pℓ) := mul_nonneg T.pE_pos.le h1
  nlinarith [mul_le_mul_of_nonneg_right hxy h1]

/-! ## Reactive against background: the headline of Section III-A -/

/-- Closed form of the reactive mean AoII, the result of Munari (2024). -/
theorem meanAoII_reactive (T : Traffic) :
    meanAoII T.δ T.e (recovery T 0) = T.δ * T.pℓ / ((1 - T.pℓ) * T.pE) := by
  rw [recovery_reactive]
  unfold meanAoII Traffic.e
  have h1 : 0 < 1 - T.pℓ := by linarith [T.pℓ_lt]
  have h2 : T.pE ≠ 0 := T.pE_pos.ne'
  field_simp
  ring

/-- With a background probability `x > 0` the mean AoII is bounded by
`δ pE pℓ / (x (1 - pℓ))²`, whatever the repeat budget. -/
theorem meanAoII_background_le (T : Traffic) {x : ℝ} (hx : 0 < x) (hx1 : x ≤ 1) :
    meanAoII T.δ T.e (recovery T x) ≤ T.δ * T.pE * T.pℓ / (x * (1 - T.pℓ)) ^ 2 := by
  have h1 : 0 < 1 - T.pℓ := by linarith [T.pℓ_lt]
  have hs : 0 < x * (1 - T.pℓ) := mul_pos hx h1
  have hr := recovery_pos T hx.le hx1
  calc meanAoII T.δ T.e (recovery T x)
      ≤ T.δ * T.e / (recovery T x) ^ 2 := meanAoII_le T.δ_pos T.e_nonneg hr
    _ ≤ T.δ * T.e / (x * (1 - T.pℓ)) ^ 2 := by
        apply div_le_div_of_nonneg_left (mul_nonneg T.δ_pos.le T.e_nonneg) (by positivity)
        exact pow_le_pow_left₀ hs.le (recovery_ge_send T hx.le hx1) 2
    _ = T.δ * T.pE * T.pℓ / (x * (1 - T.pℓ)) ^ 2 := by unfold Traffic.e; ring

/-- The reactive mean AoII exceeds the background bound by the factor
`x² (1 - pℓ) / pE²`: the cost of a lost trigger grows as `1 / pE²` while a
background re-send costs a constant. -/
theorem reactive_ge_background (T : Traffic) {x : ℝ} (hx : 0 < x) (hx1 : x ≤ 1) :
    x ^ 2 * (1 - T.pℓ) / T.pE ^ 2 * meanAoII T.δ T.e (recovery T x)
      ≤ meanAoII T.δ T.e (recovery T 0) := by
  have h1 : 0 < 1 - T.pℓ := by linarith [T.pℓ_lt]
  have hb := meanAoII_background_le T hx hx1
  have hfac : 0 ≤ x ^ 2 * (1 - T.pℓ) / T.pE ^ 2 := by positivity
  rw [meanAoII_reactive]
  calc x ^ 2 * (1 - T.pℓ) / T.pE ^ 2 * meanAoII T.δ T.e (recovery T x)
      ≤ x ^ 2 * (1 - T.pℓ) / T.pE ^ 2 * (T.δ * T.pE * T.pℓ / (x * (1 - T.pℓ)) ^ 2) :=
        mul_le_mul_of_nonneg_left hb hfac
    _ = T.δ * T.pℓ / ((1 - T.pℓ) * T.pE) := by
        have hx' : x ≠ 0 := hx.ne'
        have h1' : (1 - T.pℓ) ≠ 0 := h1.ne'
        have hpE : T.pE ≠ 0 := T.pE_pos.ne'
        field_simp

end

end Model
