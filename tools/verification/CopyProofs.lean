import Proofs
import Mathlib.Tactic.IntervalCases

open Aeneas Aeneas.Std Result
open crop_region

set_option maxHeartbeats 2000000

theorem clip_extent_spec (requested offset canvas : U32) :
    pixel_copy.clip_extent requested offset canvas ⦃ count =>
      count.val = min requested.val (canvas.val - offset.val) ∧
      count.val ≤ requested.val ∧
      (offset.val ≤ canvas.val → offset.val + count.val ≤ canvas.val) ∧
      (canvas.val < offset.val → count.val = 0) ⦄ := by
  unfold pixel_copy.clip_extent
  simp only [lift, bind_ok]
  step*
  all_goals simp_all
  all_goals scalar_tac

theorem pixel_row_spec (row column width count lanes : Usize) (length : Nat)
    (hc : column.val + count.val ≤ width.val) (hl : 0 < lanes.val)
    (hr : (row.val + 1) * width.val * lanes.val ≤ length) (hlen : length ≤ Usize.max) :
    pixel_copy.pixel_row row column width count lanes ⦃ start stop =>
      start.val = (row.val * width.val + column.val) * lanes.val ∧
      stop.val = start.val + count.val * lanes.val ∧ stop.val ≤ length ⦄ := by
  have h₀ := Nat.mul_le_mul_right lanes.val (show row.val * width.val + column.val + count.val ≤ (row.val + 1) * width.val by nlinarith)
  have h₁ : row.val * width.val + column.val ≤ (row.val * width.val + column.val) * lanes.val := by
    simpa only [Nat.mul_one] using Nat.mul_le_mul_left (row.val * width.val + column.val) (show 1 ≤ lanes.val by omega)
  unfold pixel_copy.pixel_row
  step*

theorem opaque_rgba_spec (rgb : Array U8 3#usize) :
    pixel_copy.opaque_rgba rgb ⦃ rgba => rgba.val = rgb.val ++ [255#u8] ⦄ := by
  unfold pixel_copy.opaque_rgba
  step*
  all_goals simp_all [Array.make]
  have hlen : rgb.val.length = 3 := by scalar_tac
  have he : rgb.val = [rgb.val[0], rgb.val[1], rgb.val[2]] := by
    apply List.ext_getElem
    · simp_all
    · intro i hi hj
      have hb : i < 3 := by omega
      interval_cases i <;> simp
  exact congrArg (fun xs => xs ++ [255#u8]) he.symm

#print axioms clip_extent_spec
#print axioms pixel_row_spec
#print axioms opaque_rgba_spec

theorem clipped_copy_spec (srcX srcY width height imageW imageH offsetX offsetY canvasW canvasH : U32)
    (hsrcX : srcX.val + width.val ≤ imageW.val)
    (hsrcY : srcY.val + height.val ≤ imageH.val) :
    (do
      let w ← pixel_copy.clip_extent width offsetX canvasW
      let h ← pixel_copy.clip_extent height offsetY canvasH
      ok (w, h)) ⦃ w h =>
        ∀ x y : Nat, x < w.val → y < h.val →
          srcX.val + x < imageW.val ∧ srcY.val + y < imageH.val ∧
          offsetX.val + x < canvasW.val ∧ offsetY.val + y < canvasH.val ⦄ := by
  apply WP.spec_bind (clip_extent_spec width offsetX canvasW)
  intro w hw
  apply WP.spec_bind (clip_extent_spec height offsetY canvasH)
  intro h hh
  simp only [WP.spec_ok]
  intro x y hx hy
  have hxbound : offsetX.val ≤ canvasW.val := by
    by_contra hn
    have hz := hw.2.2.2 (by omega)
    omega
  have hybound : offsetY.val ≤ canvasH.val := by
    by_contra hn
    have hz := hh.2.2.2 (by omega)
    omega
  have hcx := hw.2.2.1 hxbound
  have hcy := hh.2.2.1 hybound
  rcases hw with ⟨_, hw, _⟩
  rcases hh with ⟨_, hh, _⟩
  omega

#print axioms clipped_copy_spec
