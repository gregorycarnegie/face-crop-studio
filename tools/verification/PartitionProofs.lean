import CropRegion

open Aeneas Aeneas.Std Result
open crop_region

set_option maxHeartbeats 1000000

theorem quotient_remainder (a b : Nat) : a / b * b + a % b = a := by
  simpa only [Nat.mul_comm] using Nat.div_add_mod a b

theorem decode_job_spec (job groups height chunks : Usize)
    (hg : 0 < groups.val) (hh : 0 < height.val) (hc : 0 < chunks.val) :
    partition.decode_job job groups height chunks ⦃ n g y c =>
      g.val < groups.val ∧ y.val < height.val ∧ c.val < chunks.val ∧
      ((n.val * groups.val + g.val) * height.val + y.val) * chunks.val + c.val = job.val ⦄ := by
  unfold partition.decode_job
  step*
  all_goals simp_all only [quotient_remainder]
  all_goals exact ⟨Nat.mod_lt _ hg, Nat.mod_lt _ hh, Nat.mod_lt _ hc, True.intro⟩

theorem chunk_span_spec (index chunk length : Usize)
    (hc : 0 < chunk.val) (hi : index.val * chunk.val < length.val) :
    partition.chunk_span index chunk length ⦃ start stop =>
      start.val = index.val * chunk.val ∧
      stop.val = min ((index.val + 1) * chunk.val) length.val ∧
      start.val < stop.val ∧ stop.val ≤ length.val ⦄ := by
  unfold partition.chunk_span
  step*
  all_goals simp_all [Nat.add_mul]
  all_goals omega

theorem tile_end_spec (start stop tile : Usize)
    (hs : start.val ≤ stop.val) (ht : 0 < tile.val) :
    partition.tile_end start stop tile ⦃ full =>
      full.val = start.val + (stop.val - start.val) / tile.val * tile.val ∧
      start.val ≤ full.val ∧ full.val ≤ stop.val ∧
      stop.val - full.val < tile.val ⦄ := by
  have hb := Nat.div_mul_le_self (stop.val - start.val) tile.val
  have hr := Nat.mod_lt (stop.val - start.val) ht
  have he := Nat.mod_add_div (stop.val - start.val) tile.val
  unfold partition.tile_end
  step*
  all_goals simp_all [Nat.mul_comm]
  all_goals omega

/-- Every column belongs to a chunk, including the last partial chunk. -/
theorem chunks_cover (x chunk length : Nat) (hc : 0 < chunk) (hx : x < length) :
    x / chunk * chunk < length ∧ x / chunk * chunk ≤ x ∧
      x < min ((x / chunk + 1) * chunk) length := by
  have hb := Nat.div_mul_le_self x chunk
  have hr := Nat.mod_lt x hc
  have he := quotient_remainder x chunk
  simp only [Nat.add_mul, Nat.one_mul]
  omega

/-- Different chunks have disjoint half-open ranges. -/
theorem chunks_disjoint (i j chunk length : Nat) (h : i < j) :
    min ((i + 1) * chunk) length ≤ j * chunk :=
  Nat.le_trans (Nat.min_le_left _ _) (Nat.mul_le_mul_right chunk (by omega))

theorem digits_unique (a b a' b' radix : Nat) (hb : b < radix) (hb' : b' < radix)
    (eq : a * radix + b = a' * radix + b') : a = a' ∧ b = b' := by
  have ha : a = a' := by
    rcases lt_trichotomy a a' with h | h | h
    · have hm := Nat.mul_le_mul_right radix (Nat.succ_le_of_lt h)
      nlinarith
    · exact h
    · have hm := Nat.mul_le_mul_right radix (Nat.succ_le_of_lt h)
      nlinarith
  refine ⟨ha, ?_⟩
  rw [ha] at eq
  exact Nat.add_left_cancel eq

/-- NCHW coordinates cannot alias when channel, row and column indices are valid. -/
theorem output_indices_unique (n c y x n' c' y' x' channels height width : Nat)
    (hc : c < channels) (hc' : c' < channels)
    (hy : y < height) (hy' : y' < height)
    (hx : x < width) (hx' : x' < width)
    (eq : ((n * channels + c) * height + y) * width + x =
      ((n' * channels + c') * height + y') * width + x') :
    n = n' ∧ c = c' ∧ y = y' ∧ x = x' := by
  obtain ⟨hrow, hcol⟩ := digits_unique _ _ _ _ width hx hx' eq
  obtain ⟨hplane, hy⟩ := digits_unique _ _ _ _ height hy hy' hrow
  obtain ⟨hn, hc⟩ := digits_unique _ _ _ _ channels hc hc' hplane
  exact ⟨hn, hc, hy, hcol⟩

theorem decoded_batch_bound (job n g y c batches groups height chunks : Nat)
    (eq : ((n * groups + g) * height + y) * chunks + c = job)
    (hj : job < batches * groups * height * chunks) : n < batches := by
  by_contra h
  have h₀ : batches * groups ≤ n * groups + g :=
    Nat.le_trans (Nat.mul_le_mul_right groups (by omega)) (Nat.le_add_right _ _)
  have h₁ : batches * groups * height ≤ (n * groups + g) * height + y :=
    Nat.le_trans (Nat.mul_le_mul_right height h₀) (Nat.le_add_right _ _)
  have h₂ := Nat.le_trans (Nat.mul_le_mul_right chunks h₁) (Nat.le_add_right (((n * groups + g) * height + y) * chunks) c)
  omega

theorem row_base_spec (n c y batches channels height width : Usize)
    (hn : n.val < batches.val) (hc : c.val < channels.val) (hy : y.val < height.val)
    (hw : 0 < width.val)
    (hsize : batches.val * channels.val * height.val * width.val ≤ Usize.max) :
    partition.row_base n c y channels height width ⦃ base =>
      base.val = ((n.val * channels.val + c.val) * height.val + y.val) * width.val ∧
      base.val + width.val ≤ batches.val * channels.val * height.val * width.val ⦄ := by
  have h₀ := Nat.mul_le_mul_right channels.val (show n.val + 1 ≤ batches.val by omega)
  have h₁ : n.val * channels.val + c.val + 1 ≤ batches.val * channels.val := by nlinarith
  have h₂ := Nat.mul_le_mul_right height.val h₁
  have h₃ : (n.val * channels.val + c.val) * height.val + y.val + 1 ≤
      batches.val * channels.val * height.val := by nlinarith
  have h₄ := Nat.mul_le_mul_right width.val h₃
  have h₅ : n.val * channels.val + c.val ≤ (n.val * channels.val + c.val) * height.val := by
    simpa only [Nat.mul_one] using Nat.mul_le_mul_left (n.val * channels.val + c.val)
      (show 1 ≤ height.val by omega)
  have h₆ : (n.val * channels.val + c.val) * height.val + y.val ≤
      ((n.val * channels.val + c.val) * height.val + y.val) * width.val := by
    simpa only [Nat.mul_one] using Nat.mul_le_mul_left ((n.val * channels.val + c.val) * height.val + y.val)
      (show 1 ≤ width.val by omega)
  unfold partition.row_base
  step*

#print axioms decode_job_spec
#print axioms chunk_span_spec
#print axioms tile_end_spec
#print axioms chunks_cover
#print axioms chunks_disjoint
#print axioms output_indices_unique
#print axioms decoded_batch_bound
#print axioms row_base_spec
