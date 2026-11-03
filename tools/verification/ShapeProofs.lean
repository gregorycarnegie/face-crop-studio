import TransformProofs

open Aeneas Aeneas.Std Result
open crop_region

set_option maxHeartbeats 2000000

@[simp, scalar_tac_simps] theorem usize_max_u64 :
    (UScalar.cast .U64 core.num.Usize.MAX).val = Usize.max := by
  have hv : core.num.Usize.MAX.val = Usize.max := by simp [core.num.Usize.MAX]
  have hb : core.num.Usize.MAX.val ≤ UScalar.max .U64 := by
    rcases System.Platform.numBits_eq with h | h <;>
      norm_num [hv, Usize.max, Usize.numBits, UScalar.max, U64.max, U64.numBits, UScalarTy.numBits, h]
  have h := UScalar.cast_inBounds_spec .U64 core.num.Usize.MAX hb
  simpa only [lift, WP.spec_ok, hv] using h

@[simp, scalar_tac_simps] theorem i64_max_u128 :
    (IScalar.hcast .U128 core.num.I64.MAX).val = 9223372036854775807 := by
  decide

@[simp, scalar_tac_simps] theorem usize_u128 (value : Usize) :
    (UScalar.cast .U128 value).val = value.val := by
  have hb : value.val ≤ UScalar.max .U128 := by
    have hv : value.val < 2 ^ System.Platform.numBits := value.hmax
    rcases System.Platform.numBits_eq with h | h
    all_goals rw [h] at hv
    all_goals norm_num [UScalar.max, U128.max, U128.numBits, UScalarTy.numBits] at *
    all_goals omega
  have h := UScalar.cast_inBounds_spec .U128 value hb
  simpa only [lift, WP.spec_ok] using h

@[simp, scalar_tac_simps] theorem i64_u64_val (value : I64) (hn : 0 ≤ value.val) :
    (IScalar.hcast .U64 value).val = value.val.toNat := by
  have hm := value.hmax
  have hb : value.val ≤ (UScalar.max .U64 : Int) := by
    norm_num [UScalar.max, U64.max, U64.numBits, IScalarTy.numBits] at *
    omega
  have h := IScalar.hcast_inBounds_spec .U64 value ⟨hn, hb⟩
  simp only [lift, WP.spec_ok] at h
  omega

theorem signed_dim_spec (value : I64) :
    size.signed_dim value ⦃ result =>
      match result with
      | none => value.val < 0 ∨ (Usize.max : Int) < value.val
      | some out => 0 ≤ value.val ∧ (out.val : Int) = value.val ⦄ := by
  unfold size.signed_dim
  step*
  all_goals simp_all
  have h := IScalar.hcast_inBounds_spec .Usize value (by scalar_tac)
  simp only [lift, WP.spec_ok] at h
  scalar_tac

theorem ort_dim_spec (value : Usize) :
    size.ort_dim value ⦃ result =>
      match result with
      | none => 9223372036854775807 < value.val
      | some out => out.val = value.val ⦄ := by
  unfold size.ort_dim
  step*
  all_goals simp_all
  have h := UScalar.hcast_inBounds_spec .I64 value (by scalar_tac)
  simp only [lift, WP.spec_ok] at h
  scalar_tac

theorem raw_f32_elements_spec (bytes : Usize) :
    size.raw_f32_elements bytes ⦃ result =>
      match result with
      | none => bytes.val % 4 ≠ 0
      | some count => count.val * 4 = bytes.val ⦄ := by
  have h := Nat.div_add_mod bytes.val 4
  unfold size.raw_f32_elements
  step*

#print axioms signed_dim_spec
#print axioms ort_dim_spec
#print axioms raw_f32_elements_spec
