import GeometryProofs
import RowsProofs

open Aeneas Aeneas.Std Result
open crop_region

set_option maxHeartbeats 2000000

theorem channel_end_spec (first count channels : Usize) :
    transform.channel_end first count channels ⦃ result =>
      match result with
      | none => channels.val < first.val + count.val
      | some stop => stop.val = first.val + count.val ∧ stop.val ≤ channels.val ⦄ := by
  unfold transform.channel_end
  step*

theorem double_dim_spec (value : Usize) :
    transform.double_dim value ⦃ result =>
      match result with
      | none => Usize.max < value.val * 2
      | some out => out.val = value.val * 2 ⦄ := by
  unfold transform.double_dim
  simp only [WP.spec_ok]
  have h := Usize.checked_mul_bv_spec value 2#usize
  cases he : Usize.checked_mul value 2#usize <;> simp_all

theorem channel_blocks_spec (channels : Usize) :
    transform.channel_blocks channels ⦃ blocks =>
      blocks.val = channels.val / 8 + (if channels.val % 8 = 0 then 0 else 1) ∧
      channels.val ≤ blocks.val * 8 ∧
      (∀ b : Nat, b < blocks.val → b * 8 < channels.val) ⦄ := by
  have h := Nat.div_add_mod channels.val 8
  have hr := Nat.mod_lt channels.val (by omega : 0 < 8)
  unfold transform.channel_blocks
  step*
  split <;> step*
  all_goals simp_all
  all_goals refine ⟨by omega, ?_⟩
  all_goals intro b hb
  all_goals omega

theorem channel_parts_spec (channel : Usize) :
    transform.channel_parts channel ⦃ block lane =>
      channel.val = block.val * 8 + lane.val ∧ lane.val < 8 ⦄ := by
  have h := Nat.div_add_mod channel.val 8
  unfold transform.channel_parts
  step*

theorem channel_parts_unique (b lane b' lane' : Nat) (hl : lane < 8) (hl' : lane' < 8)
    (h : b * 8 + lane = b' * 8 + lane') : b = b' ∧ lane = lane' :=
  digits_unique b lane b' lane' 8 hl hl' h

theorem real_channel_block (c channels blocks : Nat) (hc : c < channels)
    (hblocks : channels ≤ blocks * 8) : c / 8 < blocks := by
  have h := Nat.div_mul_le_self c 8
  omega

theorem active_lanes_spec (block channels : Usize) (hb : block.val * 8 < channels.val) :
    transform.active_lanes block channels ⦃ count =>
      count.val = min 8 (channels.val - block.val * 8) ∧ 0 < count.val ∧
      (∀ lane : Nat, lane < 8 → (lane < count.val ↔ block.val * 8 + lane < channels.val)) ⦄ := by
  unfold transform.active_lanes
  step*
  all_goals simp_all
  intro lane hl
  omega

theorem plane_base_spec (batch block blocks plane : Usize) (batches length : Nat)
    (hn : batch.val < batches) (hb : block.val < blocks.val) (hp : 0 < plane.val)
    (hsize : batches * blocks.val * plane.val ≤ length) (hlen : length ≤ Usize.max) :
    transform.plane_base batch block blocks plane ⦃ base =>
      base.val = (batch.val * blocks.val + block.val) * plane.val ∧
      base.val + plane.val ≤ length ⦄ := by
  have h₀ := Nat.mul_le_mul_right blocks.val (show batch.val + 1 ≤ batches by omega)
  have h₁ : batch.val * blocks.val + block.val + 1 ≤ batches * blocks.val := by nlinarith
  have h₂ := Nat.mul_le_mul_right plane.val h₁
  have h₃ : batch.val * blocks.val + block.val ≤ (batch.val * blocks.val + block.val) * plane.val := by
    simpa only [Nat.mul_one] using Nat.mul_le_mul_left (batch.val * blocks.val + block.val) (show 1 ≤ plane.val by omega)
  unfold transform.plane_base
  step*

theorem blocked_extraction_spec (batch channel blocks plane pixel : Usize)
    (batches channels length : Nat)
    (hn : batch.val < batches) (hc : channel.val < channels)
    (hb : channels ≤ blocks.val * 8) (hp : pixel.val < plane.val)
    (hsize : batches * blocks.val * plane.val * 8 ≤ length) (hlen : length ≤ Usize.max) :
    (do
      let (b, lane) ← transform.channel_parts channel
      let base ← transform.plane_base batch b blocks plane
      geometry.input_index base pixel 8#usize lane) ⦃ index =>
        index.val = ((batch.val * blocks.val + channel.val / 8) * plane.val + pixel.val) * 8 + channel.val % 8 ∧
        index.val < length ⦄ := by
  unfold transform.channel_parts
  step as ⟨b, hblock⟩
  step as ⟨lane, hlane⟩
  have hvalid : b.val < blocks.val := by
    have h := real_channel_block channel.val channels blocks.val hc hb
    scalar_tac
  have hsmall : batches * blocks.val * plane.val ≤ length := by nlinarith
  apply WP.spec_bind (plane_base_spec batch b blocks plane batches length hn hvalid (by omega) hsmall hlen)
  intro base hbase
  have hrange : (base.val + plane.val) * 8 ≤ length := by
    have hs := Nat.mul_le_mul_right 8 hbase.2
    have hs₂ := Nat.mul_le_mul_right plane.val (show batch.val * blocks.val + b.val + 1 ≤ batches * blocks.val by
      have h := Nat.mul_le_mul_right blocks.val (show batch.val + 1 ≤ batches by omega)
      nlinarith)
    rw [hbase.1] at *
    nlinarith
  apply WP.spec_mono (input_index_spec base pixel 8#usize lane plane.val length hp (by scalar_tac) hrange hlen)
  intro index hi
  exact ⟨by simp_all, hi.2⟩

theorem upsample_row_spec (row width : Usize) (rows length : Nat)
    (hr : row.val < 2 * rows) (hw : 0 < width.val)
    (hsize : rows * width.val ≤ length) (hlen : length ≤ Usize.max) :
    transform.upsample_row row width ⦃ start stop =>
      start.val = (row.val / 2) * width.val ∧ stop.val = start.val + width.val ∧ stop.val ≤ length ⦄ := by
  have h := Nat.mul_le_mul_right width.val (show row.val / 2 + 1 ≤ rows by omega)
  unfold transform.upsample_row
  step*

theorem nearest2x_source (y x height width : Nat) (hy : y < 2 * height) (hx : x < 2 * width) :
    y / 2 < height ∧ x / 2 < width := by omega

theorem nearest2x_replicates (y x dy dx : Nat) (hy : dy < 2) (hx : dx < 2) :
    (2 * y + dy) / 2 = y ∧ (2 * x + dx) / 2 = x := by omega

theorem nearest2x_preserves_plane (plane height row : Nat) :
    (plane * (2 * height) + row) / 2 = plane * height + row / 2 := by
  have he : plane * (2 * height) = 2 * (plane * height) := by ring
  rw [he]
  omega

theorem checked_allocation_spec (elements bytes : Usize) :
    size.checked_allocation elements bytes ⦃ result =>
      match result with
      | none => (IScalar.hcast .Usize core.num.Isize.MAX).val < elements.val * bytes.val
      | some size => size.val = elements.val * bytes.val ∧
        size.val ≤ (IScalar.hcast .Usize core.num.Isize.MAX).val ⦄ := by
  have hlimit : (IScalar.hcast .Usize core.num.Isize.MAX).val ≤ Usize.max := by
    have h := (IScalar.hcast .Usize core.num.Isize.MAX).hmax
    scalar_tac
  unfold size.checked_allocation
  step*

#print axioms channel_end_spec
#print axioms double_dim_spec
#print axioms channel_blocks_spec
#print axioms channel_parts_spec
#print axioms channel_parts_unique
#print axioms real_channel_block
#print axioms active_lanes_spec
#print axioms plane_base_spec
#print axioms blocked_extraction_spec
#print axioms upsample_row_spec
#print axioms nearest2x_source
#print axioms nearest2x_replicates
#print axioms nearest2x_preserves_plane
#print axioms checked_allocation_spec
