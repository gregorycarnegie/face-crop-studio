import CropRegion

open Aeneas Aeneas.Std Result ControlFlow
open crop_region

set_option maxHeartbeats 1000000

@[simp, scalar_tac_simps] theorem isize_max_as_usize :
    (IScalar.hcast .Usize core.num.Isize.MAX).val = Isize.max.toNat := by
  have hv : core.num.Isize.MAX.val = Isize.max := by simp [core.num.Isize.MAX]
  have bounds : 0 ≤ Isize.max ∧ Isize.max ≤ (UScalar.max .Usize : Int) := by
    rcases System.Platform.numBits_eq with h | h <;>
      norm_num [Isize.max, Isize.numBits, UScalar.max, UScalarTy.numBits, IScalarTy.numBits, h]
  have h := IScalar.hcast_inBounds_spec .Usize core.num.Isize.MAX (by simpa only [hv] using bounds)
  simp only [lift, WP.spec_ok] at h
  rw [hv] at h
  omega

@[step] theorem row_range_spec (row bytes stride length : Usize) :
    rows.row_range row bytes stride length ⦃ result =>
      match result with
      | none => True
      | some (start, stop) =>
        start.val = row.val * stride.val ∧ stop.val = start.val + bytes.val ∧
        start.val ≤ stop.val ∧ stop.val ≤ length.val ⦄ := by
  unfold rows.row_range
  step*

/-- The exact concatenation of the requested pixel rows, excluding stride padding. -/
def packedPrefix (data : List U8) (bytes stride : Nat) : Nat → List U8
  | 0 => []
  | count + 1 => packedPrefix data bytes stride count ++ (data.drop (count * stride)).take bytes

@[simp] theorem packedPrefix_zero (data : List U8) (stride count : Nat) :
    packedPrefix data 0 stride count = [] := by
  induction count with
  | zero => rfl
  | succ count ih => simp [packedPrefix, ih]

@[step] theorem extend_bytes_spec (v : alloc.vec.Vec U8) (s : Slice U8)
    (fit : v.length + s.length ≤ Usize.max) :
    alloc.vec.Vec.extend_from_slice core.clone.CloneU8 v s ⦃ result =>
      result.val = v.val ++ s.val ∧ result.length = v.length + s.length ⦄ := by
  unfold alloc.vec.Vec.extend_from_slice
  rw [dite_eq_left fit]
  apply WP.spec_bind (@List.clone_spec _ core.clone.CloneU8.clone s.val (by intros; simp_all))
  intro copied hcopied
  simp [hcopied, WP.spec_ok]

theorem row_range_success (row bytes stride length : Usize)
    (fit : row.val * stride.val + bytes.val ≤ length.val) :
    rows.row_range row bytes stride length ⦃ result =>
      match result with
      | none => False
      | some (start, stop) =>
        start.val = row.val * stride.val ∧ stop.val = start.val + bytes.val ∧
        start.val ≤ stop.val ∧ stop.val ≤ length.val ⦄ := by
  unfold rows.row_range
  step*

theorem pack_rows_loop_spec
    (data : Slice U8) (height stride bytes : Usize) (out : alloc.vec.Vec U8) (row : Usize)
    (hrow : row.val ≤ height.val)
    (hcopy : out.val = packedPrefix data.val bytes.val stride.val row.val)
    (hlen : out.length = row.val * bytes.val)
    (hsize : height.val * bytes.val ≤ Usize.max)
    (hranges : ∀ r, r < height.val → r * stride.val + bytes.val ≤ data.length) :
    rows.pack_rows_loop data height stride bytes out row ⦃ result =>
      ∃ output, result = some output ∧
        output.val = packedPrefix data.val bytes.val stride.val height.val ∧
        output.length = height.val * bytes.val ⦄ := by
  unfold rows.pack_rows_loop
  apply loop.spec_decr_nat
    (fun (_, r) => height.val - r.val)
    (fun (v, r) => r.val ≤ height.val ∧
      v.val = packedPrefix data.val bytes.val stride.val r.val ∧ v.length = r.val * bytes.val)
  · rintro ⟨v, r⟩ ⟨hi, hv, hl⟩
    unfold rows.pack_rows_loop.body
    dsimp only
    split
    · apply WP.spec_bind (row_range_success r bytes stride (Slice.len data)
        (by simpa using hranges r.val (by scalar_tac)))
      intro range hr
      cases range with
      | none => exact False.elim hr
      | some endpoints =>
        rcases endpoints with ⟨start, stop⟩
        rcases hr with ⟨hs, he, hle, hbound⟩
        step as ⟨pixels, hp, hpLen⟩
        have hmore := Nat.mul_le_mul_right bytes.val (show r.val + 1 ≤ height.val by scalar_tac)
        step as ⟨nextOutput, hn, hnLen⟩
        step as ⟨nextRow, hnext⟩
        refine ⟨by scalar_tac, ?_, ?_, by scalar_tac⟩
        · simp_all [packedPrefix, List.slice]
        · simp_all [Nat.add_mul]
    · have eq : r.val = height.val := by scalar_tac
      simp_all [WP.spec_ok]
  · exact ⟨hrow, hcopy, hlen⟩

theorem pack_rows_spec (data : Slice U8) (width height stride : Usize) :
    rows.pack_rows data width height stride ⦃ result =>
      match result with
      | none => True
      | some output =>
        output.val = packedPrefix data.val (width.val * 3) stride.val height.val ∧
        output.length = height.val * (width.val * 3) ⦄ := by
  unfold rows.pack_rows
  step*
  all_goals simp_all [packedPrefix_zero, packedPrefix]
  have hranges : ∀ r, r < height.val → r * stride.val + row_bytes.val ≤ data.length := by
    intro r hr
    have hmono := Nat.mul_le_mul_right stride.val (show r ≤ height.val - 1 by omega)
    scalar_tac
  apply WP.spec_mono (pack_rows_loop_spec data height stride row_bytes
    (alloc.vec.Vec.with_capacity U8 length) 0#usize
    (by scalar_tac)
    (by simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.new, packedPrefix])
    (by simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.new])
    (by simpa only [o_post.2.1, Nat.mul_comm] using o1_post.1) hranges)
  rintro result ⟨output, eq, hval, hlen⟩
  simp_all

attribute [local step] row_range_success

/-- Valid nonempty decoder buffers are accepted, and the pixels are copied exactly. -/
theorem pack_rows_valid_nonempty (data : Slice U8) (width height stride : Usize)
    (hw : 0 < width.val) (hh : 0 < height.val)
    (hs : width.val * 3 ≤ stride.val)
    (hcapacity : width.val * 3 * height.val ≤ (IScalar.hcast .Usize core.num.Isize.MAX).val)
    (hlast : (height.val - 1) * stride.val + width.val * 3 ≤ data.length) :
    rows.pack_rows data width height stride ⦃ result =>
      match result with
      | none => False
      | some output =>
        output.val = packedPrefix data.val (width.val * 3) stride.val height.val ∧
        output.length = height.val * (width.val * 3) ⦄ := by
  have hcapmax : (IScalar.hcast .Usize core.num.Isize.MAX).val ≤ Usize.max := by
    have h := (IScalar.hcast .Usize core.num.Isize.MAX).hmax
    scalar_tac
  have hproduct : width.val * 3 * height.val ≤ Usize.max := hcapacity.trans hcapmax
  have hmax0 : 0 ≤ Isize.max := by
    rcases System.Platform.numBits_eq with h | h <;>
      norm_num [Isize.max, Isize.numBits, IScalarTy.numBits, h]
  have hcapacityNat : width.val * 3 * height.val ≤ Isize.max.toNat := by
    simpa only [isize_max_as_usize] using hcapacity
  have hcapacityInt : (width.val : Int) * 3 * height.val ≤ Isize.max := by
    have h := Int.ofNat_le.mpr hcapacityNat
    simpa only [Nat.cast_mul, Nat.cast_ofNat, Int.toNat_of_nonneg hmax0] using h
  unfold rows.pack_rows
  step*
  all_goals simp_all
  all_goals try omega
  have hranges : ∀ r, r < height.val → r * stride.val + row_bytes.val ≤ data.length := by
    intro r hr
    have hmono := Nat.mul_le_mul_right stride.val (show r ≤ height.val - 1 by omega)
    scalar_tac
  apply WP.spec_mono (pack_rows_loop_spec data height stride row_bytes
    (alloc.vec.Vec.with_capacity U8 length) 0#usize
    (by scalar_tac)
    (by simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.new, packedPrefix])
    (by simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.new])
    (by simpa only [o_post.2.1, Nat.mul_comm] using hproduct) hranges)
  rintro result ⟨output, eq, hval, hlen⟩
  simp_all

#print axioms row_range_spec
#print axioms pack_rows_loop_spec
#print axioms pack_rows_spec
#print axioms pack_rows_valid_nonempty
