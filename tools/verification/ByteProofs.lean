import GpuProofs
import RowsProofs

open Aeneas Aeneas.Std Result
open crop_region

set_option maxHeartbeats 2000000

theorem metadata_span_spec (cursor header payload trailer length : Usize) :
    metadata_ranges.span cursor header payload trailer length ⦃ result =>
      match result with
      | none => length.val < cursor.val + header.val + payload.val + trailer.val
      | some (start, stop, next) =>
        start.val = cursor.val + header.val ∧ stop.val = start.val + payload.val ∧
        next.val = stop.val + trailer.val ∧ next.val ≤ length.val ∧
        cursor.val ≤ start.val ∧ start.val ≤ stop.val ∧ stop.val ≤ next.val ∧
        (0 < header.val + payload.val + trailer.val → cursor.val < next.val) ⦄ := by
  unfold metadata_ranges.span
  step*
  all_goals scalar_tac

theorem cursor_steps_bounded (cursor : Nat → Nat) (steps length : Nat)
    (hstep : ∀ i, i < steps → cursor i < cursor (i + 1))
    (hbound : cursor steps ≤ length) : steps ≤ length := by
  have hgrow : ∀ i, i ≤ steps → cursor 0 + i ≤ cursor i := by
    intro i
    induction i with
    | zero => simp
    | succ i ih =>
      intro hi
      have hp := ih (by omega)
      have hn := hstep i (by omega)
      omega
  have h := hgrow steps (by omega)
  omega

theorem png_text_layout_spec (keyword value : Usize) :
    metadata_ranges.png_text_layout keyword value ⦃ result =>
      match result with
      | none => 4294967295 < keyword.val + value.val + 1 ∨
          (IScalar.hcast .Usize core.num.Isize.MAX).val < keyword.val + value.val + 13
      | some (data, total) => data.val = keyword.val + value.val + 1 ∧
          total.val = data.val + 12 ∧ data.val ≤ 4294967295 ∧
          total.val ≤ (IScalar.hcast .Usize core.num.Isize.MAX).val ⦄ := by
  have hlimit : (IScalar.hcast .Usize core.num.Isize.MAX).val ≤ Usize.max := by
    have h := (IScalar.hcast .Usize core.num.Isize.MAX).hmax
    scalar_tac
  unfold metadata_ranges.png_text_layout
  step*
  all_goals simp_all [U32.rMax]
  all_goals scalar_tac

@[simp, scalar_tac_simps] theorem u16_max_usize :
    (UScalar.cast .Usize core.num.U16.MAX).val = 65535 := by
  have hv : core.num.U16.MAX.val = 65535 := by norm_num [core.num.U16.MAX, U16.rMax]
  have hb : core.num.U16.MAX.val ≤ UScalar.max .Usize := by
    rcases System.Platform.numBits_eq with h | h <;>
      norm_num [hv, UScalar.max, UScalarTy.numBits, h]
  have h := UScalar.cast_inBounds_spec .Usize core.num.U16.MAX hb
  simpa only [lift, WP.spec_ok, hv] using h

theorem jpeg_length_spec (payload : Usize) :
    metadata_ranges.jpeg_length payload ⦃ result =>
      match result with
      | none => 65535 < payload.val + 2
      | some length => length.val = payload.val + 2 ∧ 2 ≤ length.val ∧ length.val ≤ 65535 ⦄ := by
  unfold metadata_ranges.jpeg_length
  step*
  all_goals simp_all [U16.rMax]

def hexDigit (byte : Nat) : Option Nat :=
  if 48 ≤ byte ∧ byte ≤ 57 then some (byte - 48)
  else if 65 ≤ byte ∧ byte ≤ 70 then some (byte - 55)
  else if 97 ≤ byte ∧ byte ≤ 102 then some (byte - 87)
  else none

@[step] theorem hex_digit_spec (byte : U8) :
    hex.digit byte ⦃ result =>
      match result with
      | none => hexDigit byte.val = none
      | some d => hexDigit byte.val = some d.val ∧ d.val < 16 ⦄ := by
  unfold hex.digit
  step*
  all_goals simp only [hexDigit]
  all_goals split_ifs <;> simp_all
  all_goals omega

def hexPair (first second : Nat) : Option Nat :=
  if first = 43 then hexDigit second
  else do
    let hi ← hexDigit first
    let lo ← hexDigit second
    some (hi * 16 + lo)

theorem hex_pair_spec (first second : U8) :
    hex.pair first second ⦃ result => result.map (fun b => b.val) = hexPair first.val second.val ⦄ := by
  unfold hex.pair
  step*
  all_goals simp_all [hexPair]
  cases result <;> simp_all

theorem hex_short_spec (byte : U8) :
    hex.short byte ⦃ result => result.map (fun b => b.val) = (hexDigit byte.val).map (fun d => d * 17) ⦄ := by
  unfold hex.short
  step*
  all_goals simp_all

#print axioms metadata_span_spec
#print axioms cursor_steps_bounded
#print axioms png_text_layout_spec
#print axioms jpeg_length_spec
#print axioms hex_digit_spec
#print axioms hex_pair_spec
#print axioms hex_short_spec
