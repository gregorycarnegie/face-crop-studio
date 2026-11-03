import CropRegion

open Aeneas Aeneas.Std Result ControlFlow
open crop_region

set_option maxHeartbeats 3000000

def headSum (counts : List Usize) (n : Nat) : Nat := ((counts.take n).map (fun c => c.val)).sum

theorem headSum_succ (counts : List Usize) (i : Nat) (hi : i < counts.length) :
    headSum counts (i + 1) = headSum counts i + counts[i].val := by
  simp only [headSum, List.take_succ_eq_append_getElem hi, List.map_append,
    List.sum_append, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]

def headRanges (counts : List Usize) : Nat → List (Nat × Nat)
  | 0 => []
  | i + 1 => headRanges counts i ++ [(headSum counts i, (counts[i]?.getD 0#usize).val)]

def headsCorrect (counts : Slice Usize) (result : Option ((alloc.vec.Vec (Usize × Usize)) × Usize)) : Prop :=
  match result with
  | none => ∃ i, i ≤ counts.length ∧ Usize.max < headSum counts.val i
  | some (ranges, total) => total.val = headSum counts.val counts.length ∧
      ranges.val.map (fun p => (p.1.val, p.2.val)) = headRanges counts.val counts.length

theorem head_ranges_loop_spec (counts : Slice Usize) (ranges : alloc.vec.Vec (Usize × Usize)) (first i : Usize)
    (hi : i.val ≤ counts.length) (hlen : ranges.length = i.val)
    (hfirst : first.val = headSum counts.val i.val)
    (hmap : ranges.val.map (fun p => (p.1.val, p.2.val)) = headRanges counts.val i.val) :
    heads.ranges_loop counts ranges first i ⦃ result => headsCorrect counts result ⦄ := by
  unfold heads.ranges_loop
  apply loop.spec_decr_nat (fun (_, _, i) => counts.length - i.val)
    (fun (v, first, i) => i.val ≤ counts.length ∧ v.length = i.val ∧
      first.val = headSum counts.val i.val ∧
      v.val.map (fun p => (p.1.val, p.2.val)) = headRanges counts.val i.val)
  · rintro ⟨v, first, i⟩ ⟨hi, hv, hf, hm⟩
    unfold heads.ranges_loop.body
    dsimp only
    split
    · step as ⟨count, hc⟩
      step as ⟨o, ho⟩
      cases o with
      | none =>
        simp only [WP.spec_ok, headsCorrect]
        refine ⟨i.val + 1, by scalar_tac, ?_⟩
        rw [headSum_succ counts.val i.val (by scalar_tac), ← hf, ← hc]
        scalar_tac
      | some stop =>
        step*
        have hs := headSum_succ counts.val i.val (by scalar_tac)
        have hr : counts.val[i.val]?.getD 0#usize = count := by
          simp only [List.getElem?_eq_getElem (by scalar_tac : i.val < counts.length), Option.getD_some, ← hc]
        simp_all [headRanges]
        all_goals scalar_tac
    · have he : i.val = counts.length := by scalar_tac
      simp_all [WP.spec_ok, headsCorrect]
  · exact ⟨hi, hlen, hfirst, hmap⟩

theorem head_ranges_spec (counts : Slice Usize) :
    heads.ranges counts ⦃ result => headsCorrect counts result ⦄ := by
  unfold heads.ranges
  apply head_ranges_loop_spec
  · scalar_tac
  · simp [alloc.vec.Vec.new]
  · simp [headSum]
  · simp [alloc.vec.Vec.new, headRanges]

#print axioms head_ranges_spec

theorem head_sum_mono (counts : List Usize) (i j : Nat) (hij : i ≤ j) :
    headSum counts i ≤ headSum counts j := by
  have h := List.sum_take_add_sum_drop ((counts.take j).map (fun c => c.val)) i
  simp only [← List.map_take, List.take_take, Nat.min_eq_left hij] at h
  unfold headSum
  omega

theorem head_ranges_disjoint (counts : List Usize) (i j : Nat)
    (hi : i < counts.length) (hij : i < j) :
    headSum counts i + counts[i].val ≤ headSum counts j := by
  have h := head_sum_mono counts (i + 1) j (by omega)
  rwa [headSum_succ counts i hi] at h

theorem head_ranges_cover (counts : List Usize) (x : Nat)
    (hx : x < headSum counts counts.length) :
    ∃ i, ∃ hi : i < counts.length, headSum counts i ≤ x ∧ x < headSum counts i + counts[i].val := by
  induction counts generalizing x with
  | nil => simp [headSum] at hx
  | cons c tail ih =>
    by_cases hfirst : x < c.val
    · exact ⟨0, by simp, by simp [headSum], by simpa [headSum] using hfirst⟩
    · have hx' : x - c.val < headSum tail tail.length := by
        simp only [headSum, List.take_length, List.map_cons, List.sum_cons] at hx ⊢
        omega
      obtain ⟨i, hi, hlo, hhi⟩ := ih (x - c.val) hx'
      have he : headSum (c :: tail) (i + 1) = c.val + headSum tail i := by simp [headSum]
      refine ⟨i + 1, by simpa using hi, ?_, ?_⟩
      · rw [he]
        omega
      · rw [he]
        simp only [List.getElem_cons_succ]
        omega

#print axioms head_ranges_disjoint
#print axioms head_ranges_cover

theorem head_ranges_valid (counts : Slice Usize)
    (hfit : headSum counts.val counts.length ≤ Usize.max) :
    heads.ranges counts ⦃ result => result.isSome = true ∧ headsCorrect counts result ⦄ := by
  apply WP.spec_mono (head_ranges_spec counts)
  intro result h
  cases result with
  | none =>
    rcases h with ⟨i, hi, hbad⟩
    have hm := head_sum_mono counts.val i counts.length hi
    omega
  | some value => exact ⟨rfl, h⟩

#print axioms head_ranges_valid
