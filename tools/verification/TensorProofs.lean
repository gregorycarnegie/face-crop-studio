import CropRegion

open Aeneas Aeneas.Std Result ControlFlow
open crop_region

set_option maxHeartbeats 1000000

/-- Unbounded mathematical product of a dimension prefix. -/
def prefixProduct (dims : Slice Usize) (count : Nat) : Nat :=
  ((dims.val.take count).map (fun d => d.val)).prod

theorem prefixProduct_succ (dims : Slice Usize) (i : Nat) (hi : i < dims.length) :
    prefixProduct dims (i + 1) = prefixProduct dims i * (dims.val[i]).val := by
  unfold prefixProduct
  rw [List.take_succ_eq_append_getElem hi]
  simp only [List.map_append, List.map_cons, List.map_nil, List.prod_append,
    List.prod_cons, List.prod_nil, Nat.mul_one]

/-- The result is exact, or a dimension prefix really exceeds the machine limit. -/
def countCorrect (dims : Slice Usize) (result : Option Usize) : Prop :=
  match result with
  | some count => count.val = prefixProduct dims dims.length
  | none => ∃ i, i ≤ dims.length ∧ Usize.max < prefixProduct dims i

theorem checked_elements_loop_spec (dims : Slice Usize) (elements index : Usize)
    (hi : index.val ≤ dims.length) (hc : elements.val = prefixProduct dims index.val) :
    size.checked_elements_loop dims elements index ⦃ result => countCorrect dims result ⦄ := by
  unfold size.checked_elements_loop
  apply loop.spec_decr_nat
    (fun (_, i) => dims.length - i.val)
    (fun (count, i) => i.val ≤ dims.length ∧ count.val = prefixProduct dims i.val)
  · rintro ⟨count, i⟩ ⟨hindex, hcount⟩
    unfold size.checked_elements_loop.body
    dsimp only
    split
    · apply WP.spec_bind (Slice.index_usize_spec dims i (by scalar_tac))
      intro dim hdim
      have hm := Usize.checked_mul_bv_spec count dim
      simp only [lift, bind_ok]
      cases hmul : Usize.checked_mul count dim with
      | none =>
        simp only [hmul] at hm ⊢
        simp only [WP.spec_ok, countCorrect]
        refine ⟨i.val + 1, by scalar_tac, ?_⟩
        rw [prefixProduct_succ dims i.val (by scalar_tac), ← hcount, ← hdim]
        exact hm
      | some next =>
        simp only [hmul] at hm ⊢
        step as ⟨nextIndex, hadd⟩
        refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        rw [hadd, prefixProduct_succ dims i.val (by scalar_tac), ← hcount, ← hdim]
        exact hm.2.1
    · simp only [WP.spec_ok, countCorrect]
      have : i.val = dims.length := by scalar_tac
      simpa [this] using hcount
  · exact ⟨hi, hc⟩

theorem checked_elements_spec (dims : Slice Usize) :
    size.checked_elements dims ⦃ result => countCorrect dims result ⦄ := by
  unfold size.checked_elements
  apply checked_elements_loop_spec
  · scalar_tac
  · simp [prefixProduct]

theorem checked_f32_bytes_spec (elements : Usize) :
    size.checked_f32_bytes elements ⦃ result =>
      match result with
      | some bytes => bytes.val = elements.val * 4
      | none => Usize.max < elements.val * 4 ⦄ := by
  unfold size.checked_f32_bytes
  simp only [WP.spec_ok]
  have hm := Usize.checked_mul_bv_spec elements 4#usize
  cases hmul : Usize.checked_mul elements 4#usize <;> simp_all

#print axioms checked_elements_spec
#print axioms checked_f32_bytes_spec
