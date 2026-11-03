import CropRegion

open Aeneas Aeneas.Std Result
open crop_region.region

set_option maxHeartbeats 1000000

@[simp, scalar_tac_simps] theorem saturating_sub_val (x y : U32) :
    (core.num.U32.saturating_sub x y).val = x.val - y.val := by
  simp only [core.num.U32.saturating_sub, UScalar.saturating_sub, UScalar.val,
    BitVec.toNat_ofNat, Nat.zero_le, max_eq_right]
  exact Nat.mod_eq_of_lt (by have := x.hmax; omega)

@[simp, scalar_tac_simps] theorem saturating_add_val (x y : U32) :
    (core.num.U32.saturating_add x y).val = min 4294967295 (x.val + y.val) := by
  simp only [core.num.U32.saturating_add, UScalar.saturating_add, UScalar.val,
    BitVec.toNat_ofNat, UScalar.max, UScalarTy.numBits]
  apply Nat.mod_eq_of_lt
  change min 4294967295 (x.val + y.val) < 4294967296
  omega

/-- Even manually constructed regions with inconsistent padding cannot return an
    empty rectangle or a rectangle extending beyond the supplied image. -/
theorem in_bounds_rect_safe (r : CropRegion) (img_w img_h : U32) :
    r.in_bounds_rect img_w img_h ⦃ result =>
      match result with
      | none => True
      | some (x, y, w, h) =>
        0 < w.val ∧ 0 < h.val ∧
        x.val + w.val ≤ img_w.val ∧ y.val + h.val ≤ img_h.val ⦄ := by
  unfold CropRegion.in_bounds_rect CropRegion.in_bounds_width CropRegion.in_bounds_height
  simp only [lift, bind_ok]
  split <;> simp_all [WP.spec_ok]
  split <;> simp_all [WP.spec_ok]
  scalar_tac

#print axioms in_bounds_rect_safe

@[simp, scalar_tac_simps] theorem cast_i32_i64_val (x : I32) :
    (IScalar.cast .I64 x).val = x.val := by
  have h := IScalar.cast_inBounds_spec .I64 x (by scalar_tac)
  simpa only [lift, WP.spec_ok] using h

@[simp, scalar_tac_simps] theorem cast_u32_i64_val (x : U32) :
    (UScalar.hcast .I64 x).val = x.val := by
  have h := UScalar.hcast_inBounds_spec .I64 x (by scalar_tac)
  simpa only [lift, WP.spec_ok] using h

@[simp, scalar_tac_simps] theorem cast_i64_u32_val (x : I64) :
    (IScalar.hcast .U32 x).val = (x.val % 4294967296).toNat := by
  simp only [IScalar.hcast, UScalar.val, BitVec.signExtend, BitVec.toNat_ofInt,
    IScalar.val, IScalarTy.numBits, UScalarTy.numBits]
  norm_num

@[step] theorem new_spec (x y : I32) (w h iw ih : U32) :
    CropRegion.new x y w h iw ih ⦃ r =>
      r.x = x ∧ r.y = y ∧ r.width = w ∧ r.height = h ∧
      r.pad_left.val = (max (-x.val) 0).toNat ∧
      r.pad_top.val = (max (-y.val) 0).toNat ∧
      r.pad_right.val = ((max (x.val + w.val - iw.val) 0) % 4294967296).toNat ∧
      r.pad_bottom.val = ((max (y.val + h.val - ih.val) 0) % 4294967296).toNat ⦄ := by
  unfold CropRegion.new
  step*
  all_goals simp_all
  all_goals scalar_tac

#print axioms new_spec

/-- Length of the overlap of half-open intervals [x, x+w) and [0, image). -/
def overlap (x : I32) (w image : U32) : Nat :=
  (min (x.val + w.val) image.val - max x.val 0).toNat

theorem axis_intersection (x : I32) (w image : U32) :
    min (w.val - min 4294967295
      ((max (-x.val) 0).toNat +
        ((max (x.val + w.val - image.val) 0) % 4294967296).toNat))
      (image.val - (max x.val 0).toNat) = overlap x w image := by
  have := x.hmin
  have := x.hmax
  have := w.hmax
  have := image.hmax
  unfold overlap
  norm_num [IScalarTy.numBits, UScalarTy.numBits] at *
  omega

#print axioms axis_intersection

@[simp, scalar_tac_simps] theorem start_val (x : I32) :
    (IScalar.hcast .U32 (core.cmp.impls.OrdI32.max x 0#i32)).val =
      (max x.val 0).toNat := by
  have h := IScalar.hcast_inBounds_spec .U32 (core.cmp.impls.OrdI32.max x 0#i32)
    (by scalar_tac)
  simp only [lift, WP.spec_ok] at h
  scalar_tac

/-- Constructing a region and intersecting with the same image returns exactly
    the geometric overlap, or None exactly when either overlap length is zero. -/
theorem new_intersection_correct (x y : I32) (w h iw ih : U32) :
    (do
      let r ← CropRegion.new x y w h iw ih
      r.in_bounds_rect iw ih) ⦃ result =>
      match result with
      | none => overlap x w iw = 0 ∨ overlap y h ih = 0
      | some (sx, sy, cw, ch) =>
        sx.val = (max x.val 0).toNat ∧ sy.val = (max y.val 0).toNat ∧
        cw.val = overlap x w iw ∧ ch.val = overlap y h ih ∧
        0 < cw.val ∧ 0 < ch.val ⦄ := by
  step as ⟨r, hx, hy, hw, hh, hl, ht, hr, hb⟩
  have ax := axis_intersection x w iw
  have ay := axis_intersection y h ih
  unfold CropRegion.in_bounds_rect CropRegion.in_bounds_width CropRegion.in_bounds_height
  simp only [lift, bind_ok]
  split <;> simp_all [WP.spec_ok]
  all_goals try split
  all_goals try simp_all [WP.spec_ok]
  all_goals scalar_tac

#print axioms new_intersection_correct
