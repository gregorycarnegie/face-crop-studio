import TensorProofs
import Mathlib.Tactic.IntervalCases

open Aeneas Aeneas.Std Result
open crop_region

set_option maxHeartbeats 1000000

attribute [step] checked_elements_spec

@[step] theorem checked_nchw_spec (n c h w : Usize) :
    layout.checked_nchw n c h w ⦃ result =>
      match result with
      | none => True
      | some (plane, elements) =>
        plane.val = h.val * w.val ∧ elements.val = n.val * c.val * plane.val ⦄ := by
  unfold layout.checked_nchw
  step*
  all_goals simp_all [countCorrect, prefixProduct, Array.to_slice, Array.make, Nat.mul_assoc]

theorem checked_nchw_valid (n c h w : Usize)
    (hp : h.val * w.val ≤ Usize.max) (hc : n.val * c.val ≤ Usize.max)
    (he : n.val * c.val * (h.val * w.val) ≤ Usize.max) :
    layout.checked_nchw n c h w ⦃ result =>
      match result with
      | none => False
      | some (plane, elements) =>
        plane.val = h.val * w.val ∧ elements.val = n.val * c.val * plane.val ⦄ := by
  unfold layout.checked_nchw
  step*
  all_goals simp_all [countCorrect, prefixProduct, Array.to_slice, Array.make, Nat.mul_assoc]
  all_goals
    obtain ⟨i, hi, bad⟩ : ∃ i : Nat, _ := by assumption
    interval_cases i <;> simp_all <;> scalar_tac

@[step] theorem plane_range_spec (b c n channels plane elements : Usize) :
    layout.plane_range b c n channels plane elements ⦃ result =>
      match result with
      | none => True
      | some (start, stop) =>
        b.val < n.val ∧ c.val < channels.val ∧
        start.val = (b.val * channels.val + c.val) * plane.val ∧
        stop.val = start.val + plane.val ∧ stop.val ≤ elements.val ⦄ := by
  unfold layout.plane_range
  step*

theorem plane_range_success (b c n channels plane elements : Usize)
    (hb : b.val < n.val) (hc : c.val < channels.val)
    (hi : b.val * channels.val + c.val ≤ Usize.max)
    (he : (b.val * channels.val + c.val) * plane.val + plane.val ≤ elements.val) :
    layout.plane_range b c n channels plane elements ⦃ result =>
      match result with
      | none => False
      | some (start, stop) =>
        start.val = (b.val * channels.val + c.val) * plane.val ∧
        stop.val = start.val + plane.val ∧ stop.val ≤ elements.val ⦄ := by
  unfold layout.plane_range
  step*

/-- Distinct valid planes are disjoint, including zero-length planes. -/
theorem planes_disjoint (b₁ c₁ b₂ c₂ channels plane : Nat)
    (hc₁ : c₁ < channels) (hc₂ : c₂ < channels)
    (different : b₁ ≠ b₂ ∨ c₁ ≠ c₂) :
    (b₁ * channels + c₁ + 1) * plane ≤ (b₂ * channels + c₂) * plane ∨
    (b₂ * channels + c₂ + 1) * plane ≤ (b₁ * channels + c₁) * plane := by
  have distinct : b₁ * channels + c₁ ≠ b₂ * channels + c₂ := by
    intro eq
    rcases lt_trichotomy b₁ b₂ with hb | hb | hb
    · have h := Nat.mul_le_mul_right channels (Nat.succ_le_of_lt hb)
      nlinarith
    · subst b₂
      rcases different with hb | hc
      · exact hb rfl
      · exact hc (by omega)
    · have h := Nat.mul_le_mul_right channels (Nat.succ_le_of_lt hb)
      nlinarith
  rcases lt_or_gt_of_ne distinct with h | h
  · exact Or.inl (Nat.mul_le_mul_right plane (by omega))
  · exact Or.inr (Nat.mul_le_mul_right plane (by omega))

#print axioms checked_nchw_spec
#print axioms checked_nchw_valid
#print axioms plane_range_spec
#print axioms plane_range_success
#print axioms planes_disjoint
