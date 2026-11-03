import PartitionProofs

open Aeneas Aeneas.Std Result
open crop_region

set_option maxHeartbeats 2000000

@[simp, scalar_tac_simps] theorem saturating_usize_sub_val (x y : Usize) :
    (core.num.Usize.saturating_sub x y).val = x.val - y.val := by
  simp only [core.num.Usize.saturating_sub, UScalar.saturating_sub, UScalar.val,
    BitVec.toNat_ofNat, Nat.zero_le, max_eq_right]
  exact Nat.mod_eq_of_lt (by have := x.hmax; omega)

@[step] theorem saturating_usize_sub_spec (x y : Usize) :
    lift (core.num.Usize.saturating_sub x y) ⦃ r => r.val = x.val - y.val ⦄ := by
  simp [lift, WP.spec_ok]

theorem output_dim_spec (input kernel stride pad : Usize) :
    geometry.output_dim input kernel stride pad ⦃ result =>
      match result with
      | none => kernel.val = 0 ∨ stride.val = 0 ∨ Usize.max < input.val + 2 * pad.val
      | some out =>
        0 < kernel.val ∧ 0 < stride.val ∧ input.val + 2 * pad.val ≤ Usize.max ∧
        out.val = (input.val + 2 * pad.val - kernel.val) / stride.val + 1 ∧ 0 < out.val ⦄ := by
  unfold geometry.output_dim
  step*
  all_goals simp_all
  have hm := Usize.checked_add_bv_spec i1 1#usize
  have hd := Nat.div_le_self (input.val + pad.val * 2 - kernel.val) stride.val
  have hm1 := (1#usize).hmax
  cases he : Usize.checked_add i1 1#usize <;> simp_all [Nat.mul_comm] <;> scalar_tac

theorem tap_rows_spec (origin height kernel pad : Usize)
    (hsize : height.val + pad.val ≤ Usize.max) :
    geometry.tap_rows origin height kernel pad ⦃ lo hi first =>
      lo.val ≤ hi.val ∧ hi.val ≤ kernel.val ∧ first.val + (hi.val - lo.val) ≤ height.val ∧
      (∀ k : Nat, k < kernel.val →
        (lo.val ≤ k ∧ k < hi.val ↔ pad.val ≤ origin.val + k ∧ origin.val + k < height.val + pad.val)) ∧
      (∀ k : Nat, lo.val ≤ k → k < hi.val →
        first.val + (k - lo.val) + pad.val = origin.val + k) ⦄ := by
  unfold geometry.tap_rows
  step*
  all_goals simp_all
  refine ⟨by omega, ?_, ?_⟩
  · intro k hk
    omega
  · intro k hlo hhi
    omega

theorem ceil_le_iff (a stride x : Nat) (hs : 0 < stride) :
    a / stride + (if a % stride = 0 then 0 else 1) ≤ x ↔ a ≤ x * stride := by
  by_cases hm : a % stride = 0
  · have he : a = a / stride * stride := by
      have h := Nat.div_add_mod a stride
      simp only [hm, Nat.add_zero] at h
      nlinarith
    simp only [hm, ↓reduceIte, Nat.add_zero]
    constructor
    · intro h
      have hh := Nat.mul_le_mul_right stride h
      omega
    · intro h
      by_contra hn
      have hh := Nat.mul_le_mul_right stride (show x + 1 ≤ a / stride by omega)
      nlinarith
  · simp only [hm, ↓reduceIte, Nat.add_one_le_iff, Nat.div_lt_iff_lt_mul hs]
    constructor
    · omega
    · intro h
      have he : a ≠ x * stride := by
        intro he
        apply hm
        rw [he]
        simp
      omega

theorem interior_spec (width kernel stride pad out : Usize)
    (hk : 0 < kernel.val) (hs : 0 < stride.val)
    (hsize : width.val + pad.val ≤ Usize.max) :
    geometry.interior width kernel stride pad out ⦃ lo hi =>
      lo.val ≤ hi.val ∧ hi.val ≤ out.val ∧
      (∀ x : Nat, x < out.val →
        (lo.val ≤ x ∧ x < hi.val ↔
          pad.val ≤ x * stride.val ∧ x * stride.val + kernel.val ≤ width.val + pad.val)) ⦄ := by
  have hquot := Nat.div_le_self (width.val + pad.val - kernel.val) stride.val
  have hceil := Nat.div_le_self pad.val stride.val
  have hrem := Nat.div_add_mod pad.val stride.val
  have hmod := Nat.mod_lt pad.val hs
  have hmax := pad.hmax
  have hceilfit : pad.val % stride.val ≠ 0 → pad.val / stride.val + 1 ≤ pad.val := by
    intro hne
    have hp : 0 < pad.val := by omega
    have hs₂ : 1 < stride.val := by
      omega
    have hh := Nat.div_lt_self hp hs₂
    omega
  unfold geometry.interior
  step*
  split <;> step*
  all_goals split <;> step*
  all_goals simp_all
  all_goals intro x hx
  all_goals have hl := ceil_le_iff pad.val stride.val x hs
  all_goals have hu := Nat.le_div_iff_mul_le (x := x) (y := width.val + pad.val - kernel.val) hs
  all_goals simp_all
  all_goals omega

/-- Every tap in a supported output window has a representable padded coordinate. -/
theorem window_tap_bounded (input kernel stride pad out x tap : Nat)
    (_hk : 0 < kernel) (hs : 0 < stride) (ht : tap < kernel)
    (hout : out = (input + 2 * pad - kernel) / stride + 1) (hx : x < out) :
    x * stride + tap < max (input + 2 * pad) kernel := by
  have hq : x ≤ (input + 2 * pad - kernel) / stride := by omega
  have hm := (Nat.le_div_iff_mul_le hs).mp hq
  by_cases hp : kernel ≤ input + 2 * pad
  · omega
  · have hz : x = 0 := by
      have hz : input + 2 * pad - kernel = 0 := by omega
      simp only [hz, Nat.zero_div, Nat.zero_add] at hout
      omega
    simp only [hz, Nat.zero_mul, Nat.zero_add]
    omega

/-- All pixels of an interior tile may omit horizontal bounds checks. -/
theorem interior_tile_safe (width kernel stride pad out : Usize)
    (hk : 0 < kernel.val) (hs : 0 < stride.val)
    (hsize : width.val + pad.val ≤ Usize.max) :
    geometry.interior width kernel stride pad out ⦃ lo hi =>
      ∀ ox p kx : Nat, lo.val ≤ ox → ox + p < hi.val → kx < kernel.val →
        pad.val ≤ (ox + p) * stride.val + kx ∧
        (ox + p) * stride.val + kx < width.val + pad.val ⦄ := by
  apply WP.spec_mono (interior_spec width kernel stride pad out hk hs hsize)
  rintro ⟨lo, hi⟩ h ox p kx hlo hhi hkx
  rcases h with ⟨_, hbound, hspec⟩
  have hpoint := (hspec (ox + p) (by omega)).mp ⟨by omega, hhi⟩
  rcases hpoint with ⟨hleft, hright⟩
  omega

theorem column_inside_spec (padded pad width : Usize) :
    geometry.column_inside padded pad width ⦃ inside =>
      inside = true ↔ pad.val ≤ padded.val ∧ padded.val < pad.val + width.val ⦄ := by
  unfold geometry.column_inside
  step*

theorem column_index_spec (origin offset pad : Usize) (width : Nat)
    (hfit : origin.val + offset.val ≤ Usize.max)
    (hlo : pad.val ≤ origin.val + offset.val)
    (hhi : origin.val + offset.val < pad.val + width) :
    geometry.column_index origin offset pad ⦃ ix =>
      ix.val + pad.val = origin.val + offset.val ∧ ix.val < width ⦄ := by
  unfold geometry.column_index
  step*
  all_goals scalar_tac

theorem input_index_spec (row column block lane : Usize) (width length : Nat)
    (hx : column.val < width) (hj : lane.val < block.val)
    (hrow : (row.val + width) * block.val ≤ length) (hlen : length ≤ Usize.max) :
    geometry.input_index row column block lane ⦃ index =>
      index.val = (row.val + column.val) * block.val + lane.val ∧ index.val < length ⦄ := by
  have h := Nat.mul_le_mul_right block.val (show row.val + column.val + 1 ≤ row.val + width by omega)
  have hb : row.val + column.val ≤ (row.val + column.val) * block.val := by
    simpa only [Nat.mul_one] using Nat.mul_le_mul_left (row.val + column.val) (show 1 ≤ block.val by omega)
  unfold geometry.input_index
  step*

theorem tap_offset_spec (ib ky kernel block : Usize) (blocks : Nat)
    (hib : ib.val < blocks) (hky : ky.val < kernel.val) (hb : 0 < block.val)
    (hfit : blocks * kernel.val * kernel.val * block.val ≤ Usize.max) :
    geometry.tap_offset ib ky kernel block ⦃ tap =>
      tap.val = (ib.val * kernel.val + ky.val) * kernel.val * block.val ∧
      tap.val + kernel.val * block.val ≤ blocks * kernel.val * kernel.val * block.val ⦄ := by
  have h₀ := Nat.mul_le_mul_right kernel.val (show ib.val + 1 ≤ blocks by omega)
  have h₁ : ib.val * kernel.val + ky.val + 1 ≤ blocks * kernel.val := by nlinarith
  have h₂ := Nat.mul_le_mul_right (kernel.val * block.val) h₁
  have hk : 0 < kernel.val := by omega
  have h₃ : ib.val * kernel.val + ky.val ≤ (ib.val * kernel.val + ky.val) * kernel.val := by
    simpa only [Nat.mul_one] using Nat.mul_le_mul_left (ib.val * kernel.val + ky.val) (show 1 ≤ kernel.val by omega)
  have h₄ : (ib.val * kernel.val + ky.val) * kernel.val ≤ (ib.val * kernel.val + ky.val) * kernel.val * block.val := by
    simpa only [Nat.mul_one] using Nat.mul_le_mul_left ((ib.val * kernel.val + ky.val) * kernel.val) (show 1 ≤ block.val by omega)
  unfold geometry.tap_offset
  step*
  all_goals simp_all [Nat.mul_assoc, Nat.add_mul]
  all_goals scalar_tac

theorem weight_index_spec (filter per_block tap kx block lane : Usize) (filters kernel length : Nat)
    (hf : filter.val < filters) (hx : kx.val < kernel) (hj : lane.val < block.val)
    (ht : tap.val + kernel * block.val ≤ per_block.val)
    (hsize : filters * per_block.val ≤ length) (hlen : length ≤ Usize.max) :
    geometry.weight_index filter per_block tap kx block lane ⦃ index =>
      index.val = filter.val * per_block.val + tap.val + kx.val * block.val + lane.val ∧
      index.val < length ⦄ := by
  have h₀ := Nat.mul_le_mul_right block.val (show kx.val + 1 ≤ kernel by omega)
  have h₁ := Nat.mul_le_mul_right per_block.val (show filter.val + 1 ≤ filters by omega)
  unfold geometry.weight_index
  step*
  all_goals simp_all [Nat.add_mul]
  all_goals scalar_tac

/-- Compose the production row and lane calculations, for NCHW and blocked inputs alike. -/
theorem input_read_spec (n ib y x block lane batches blocks height width : Usize) (length : Nat)
    (hn : n.val < batches.val) (hi : ib.val < blocks.val) (hy : y.val < height.val)
    (hx : x.val < width.val) (hj : lane.val < block.val)
    (hsize : batches.val * blocks.val * height.val * width.val * block.val ≤ length)
    (hlen : length ≤ Usize.max) :
    (do
      let row ← partition.row_base n ib y blocks height width
      geometry.input_index row x block lane) ⦃ index =>
        index.val = (((n.val * blocks.val + ib.val) * height.val + y.val) * width.val + x.val) * block.val + lane.val ∧
        index.val < length ⦄ := by
  have hscale : batches.val * blocks.val * height.val * width.val ≤
      batches.val * blocks.val * height.val * width.val * block.val := by
    simpa only [Nat.mul_one] using Nat.mul_le_mul_left
      (batches.val * blocks.val * height.val * width.val) (show 1 ≤ block.val by omega)
  apply WP.spec_bind (row_base_spec n ib y batches blocks height width hn hi hy (by omega)
    (hscale.trans (hsize.trans hlen)))
  intro row hrow
  have hrange := (Nat.mul_le_mul_right block.val hrow.2).trans hsize
  apply WP.spec_mono (input_index_spec row x block lane width.val length hx hj hrange hlen)
  intro index hindex
  exact ⟨by rw [hindex.1, hrow.1], hindex.2⟩

/-- Compose the production tap and filter calculations, including padded channel lanes. -/
theorem weight_read_spec (filter per_block ib ky kx kernel block lane : Usize)
    (filters blocks length : Nat)
    (hf : filter.val < filters) (hi : ib.val < blocks)
    (hy : ky.val < kernel.val) (hx : kx.val < kernel.val) (hj : lane.val < block.val)
    (hper : per_block.val = blocks * kernel.val * kernel.val * block.val)
    (hsize : filters * per_block.val ≤ length) (hlen : length ≤ Usize.max) :
    (do
      let tap ← geometry.tap_offset ib ky kernel block
      geometry.weight_index filter per_block tap kx block lane) ⦃ index =>
        index.val = filter.val * per_block.val +
          (ib.val * kernel.val + ky.val) * kernel.val * block.val + kx.val * block.val + lane.val ∧
        index.val < length ⦄ := by
  have hfit : blocks * kernel.val * kernel.val * block.val ≤ Usize.max := by
    rw [← hper]
    scalar_tac
  apply WP.spec_bind (tap_offset_spec ib ky kernel block blocks hi hy (by omega) hfit)
  intro tap ht
  apply WP.spec_mono (weight_index_spec filter per_block tap kx block lane filters kernel.val length
    hf hx hj (by omega) hsize hlen)
  intro index hindex
  exact ⟨by rw [hindex.1, ht.1], hindex.2⟩

theorem tiled_taps_fit (input kernel stride pad out ox pixels p kx : Nat)
    (hk : 0 < kernel) (hs : 0 < stride) (hx : ox + pixels ≤ out)
    (hp : p < pixels) (ht : kx < kernel)
    (hout : out = (input + 2 * pad - kernel) / stride + 1)
    (hsize : input + 2 * pad ≤ Usize.max) (hkernel : kernel ≤ Usize.max) :
    ox * stride + (p * stride + kx) ≤ Usize.max := by
  have hb := window_tap_bounded input kernel stride pad out (ox + p) kx hk hs ht hout (by omega)
  have hm : max (input + 2 * pad) kernel ≤ Usize.max := by omega
  simp only [Nat.add_mul] at hb
  omega

#print axioms output_dim_spec
#print axioms tap_rows_spec
#print axioms interior_spec
#print axioms interior_tile_safe
#print axioms window_tap_bounded
#print axioms column_inside_spec
#print axioms column_index_spec
#print axioms input_index_spec
#print axioms tap_offset_spec
#print axioms weight_index_spec
#print axioms input_read_spec
#print axioms weight_read_spec
#print axioms tiled_taps_fit
