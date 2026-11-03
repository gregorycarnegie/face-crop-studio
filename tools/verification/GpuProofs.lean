import GeometryProofs

open Aeneas Aeneas.Std Result
open crop_region

set_option maxHeartbeats 2000000

theorem gpu_output_dim_spec (size pad kernel stride : U32) :
    gpu_geometry.output_dim size pad kernel stride ⦃ result =>
      match result with
      | none => kernel.val = 0 ∨ stride.val = 0 ∨
          4294967295 < size.val + 2 * pad.val ∨ size.val + 2 * pad.val < kernel.val
      | some out => 0 < kernel.val ∧ 0 < stride.val ∧
          kernel.val ≤ size.val + 2 * pad.val ∧ size.val + 2 * pad.val ≤ 4294967295 ∧
          out.val = (size.val + 2 * pad.val - kernel.val) / stride.val + 1 ∧ 0 < out.val ⦄ := by
  unfold gpu_geometry.output_dim
  step*
  all_goals simp_all
  have hm := U32.checked_add_bv_spec i 1#u32
  have hd := Nat.div_le_self (size.val + pad.val * 2 - kernel.val) stride.val
  cases he : U32.checked_add i 1#u32 <;> simp_all [Nat.mul_comm] <;> scalar_tac

theorem gpu_ceil_div_spec (value divisor : U32) (hd : 0 < divisor.val) :
    gpu_geometry.ceil_div value divisor ⦃ count =>
      count.val = value.val / divisor.val + (if value.val % divisor.val = 0 then 0 else 1) ⦄ := by
  have hq := Nat.div_le_self value.val divisor.val
  have hr := Nat.mod_lt value.val hd
  have hsum := Nat.div_add_mod value.val divisor.val
  have hfit : value.val % divisor.val ≠ 0 → value.val / divisor.val + 1 ≤ value.val := by
    intro hn
    have hv : 0 < value.val := by omega
    have hs : 1 < divisor.val := by omega
    have h := Nat.div_lt_self hv hs
    omega
  unfold gpu_geometry.ceil_div
  step*
  split <;> step*
  all_goals simp_all

theorem dispatch_coverage (value divisor count : Nat) (hd : 0 < divisor)
    (hcount : count = value / divisor + (if value % divisor = 0 then 0 else 1)) :
    value ≤ count * divisor ∧
    (∀ i : Nat, i < value → i / divisor < count ∧ i % divisor < divisor ∧
      i / divisor * divisor + i % divisor = i) ∧
    (∀ g : Nat, g < count → g * divisor < value) := by
  have hbound := (ceil_le_iff value divisor count hd).mp (by omega)
  refine ⟨hbound, ?_, ?_⟩
  · intro i hi
    refine ⟨(Nat.div_lt_iff_lt_mul hd).mpr (by omega), Nat.mod_lt i hd, ?_⟩
    exact quotient_remainder i divisor
  · intro g hg
    have h := ceil_le_iff value divisor g hd
    omega

@[simp, scalar_tac_simps] theorem u32_max_usize :
    (UScalar.cast .Usize core.num.U32.MAX).val = 4294967295 := by
  have hv : core.num.U32.MAX.val = 4294967295 := by norm_num [core.num.U32.MAX, U32.rMax]
  have hb : core.num.U32.MAX.val ≤ UScalar.max .Usize := by
    rcases System.Platform.numBits_eq with h | h <;>
      norm_num [hv, UScalar.max, UScalarTy.numBits, h]
  have h := UScalar.cast_inBounds_spec .Usize core.num.U32.MAX hb
  simpa only [lift, WP.spec_ok, hv] using h

theorem shader_dim_spec (value : Usize) :
    gpu_geometry.shader_dim value ⦃ result =>
      match result with
      | none => 4294967295 < value.val
      | some out => out.val = value.val ∧ value.val ≤ 4294967295 ⦄ := by
  unfold gpu_geometry.shader_dim
  step*
  all_goals simp_all [U32.rMax]
  have h := UScalar.cast_inBounds_spec .U32 value (by scalar_tac)
  simp only [lift, WP.spec_ok] at h
  scalar_tac

theorem grouped_channels_spec (input output groups : U32) :
    gpu_geometry.grouped_channels input output groups ⦃ result =>
      match result with
      | none => groups.val = 0 ∨ input.val % groups.val ≠ 0 ∨ output.val % groups.val ≠ 0
      | some (ip, op) => 0 < groups.val ∧
        ip.val * groups.val = input.val ∧ op.val * groups.val = output.val ⦄ := by
  have hi := Nat.div_add_mod input.val groups.val
  have ho := Nat.div_add_mod output.val groups.val
  unfold gpu_geometry.grouped_channels
  step*
  all_goals simp_all [Nat.mul_comm]
  all_goals omega

#print axioms gpu_output_dim_spec
#print axioms gpu_ceil_div_spec
#print axioms dispatch_coverage
#print axioms shader_dim_spec
#print axioms grouped_channels_spec
