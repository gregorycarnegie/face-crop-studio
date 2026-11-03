import CropRegion

open Aeneas Aeneas.Std Result ControlFlow
open crop_region

set_option maxHeartbeats 3000000

/-! Greedy suppression over an arbitrary overlap relation `R`: the closure the production NMS
passes computes floating-point IoU, which these theorems treat as an opaque Boolean oracle. -/

abbrev flag (s : List Bool) (k : Nat) : Bool := s.getD k false

/-- Kept pairs whose earlier member has been visited do not overlap. -/
def Independent (R : Nat → Nat → Bool) (s : List Bool) (visited : Nat) : Prop :=
  ∀ a b, a < visited → a < b → b < s.length →
    flag s a = false → flag s b = false → R a b = false

/-- Each suppressed entry overlaps an earlier, visited, kept one. -/
def Covered (R : Nat → Nat → Bool) (s : List Bool) (visited : Nat) : Prop :=
  ∀ b, b < s.length → flag s b = true →
    ∃ a, a < b ∧ a < visited ∧ flag s a = false ∧ R a b = true

/-- What the inner loop has done to `s0` while kept entry `i` scanned up to `j`. -/
def Scanned (R : Nat → Nat → Bool) (s0 s : List Bool) (i j : Nat) : Prop :=
  s.length = s0.length ∧
  (∀ k, k ≤ i → flag s k = flag s0 k) ∧
  (∀ k, flag s0 k = true → flag s k = true) ∧
  (∀ k, flag s k = true → flag s0 k = true ∨ (i < k ∧ R i k = true)) ∧
  (∀ k, i < k → k < j → flag s k = false → R i k = false)

theorem flag_set (s : List Bool) (j k : Nat) (hj : j < s.length) :
    flag (s.set j true) k = if k = j then true else flag s k := by
  by_cases h : k = j
  · subst h
    simp [flag, List.getD_eq_getElem?_getD, hj]
  · simp [flag, List.getD_eq_getElem?_getD, h, Ne.symm h]

theorem inner_spec {F : Type} (inst : core.ops.function.Fn F (Usize × Usize) Bool)
    (overlaps : F) (R : Nat → Nat → Bool)
    (hcall : ∀ a b : Usize, inst.call overlaps (a, b) ⦃ r => r = R a.val b.val ⦄)
    (len i j : Usize) (s0 s : alloc.vec.Vec Bool)
    (hlen : s0.length = len.val) (hij : i.val < j.val) (hj : j.val ≤ len.val)
    (hscan : Scanned R s0.val s.val i.val j.val) :
    greedy.greedy_suppression_loop0_loop0 inst len overlaps s i j
      ⦃ result => Scanned R s0.val result.val i.val len.val ⦄ := by
  unfold greedy.greedy_suppression_loop0_loop0
  apply loop.spec_decr_nat (fun (_, k) => len.val - k.val)
    (fun (v, k) => i.val < k.val ∧ k.val ≤ len.val ∧ Scanned R s0.val v.val i.val k.val)
  · rintro ⟨v, k⟩ ⟨hik, hk, hlen', hkeep, hmono, hnew, hdone⟩
    unfold greedy.greedy_suppression_loop0_loop0.body
    dsimp only
    split
    · have hkv : k.val < v.length := by simp_all
      have hflag : flag v.val k.val = v.val[k.val] := by
        simp [flag, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hkv]
      step as ⟨b, hb⟩
      cases b
      · simp only [Bool.false_eq_true, ↓reduceIte]
        step as ⟨c, hc⟩
        cases c
        · -- Kept and apart from `i`: nothing changes, and `k` joins the scanned range.
          simp only [Bool.false_eq_true, ↓reduceIte]
          step*
          refine ⟨by scalar_tac, by scalar_tac, ⟨hlen', hkeep, hmono, hnew, ?_⟩, by scalar_tac⟩
          intro m hm1 hm2 hm3
          by_cases hmk : m = k.val
          · subst hmk
            exact hc.symm
          · exact hdone m hm1 (by omega) hm3
        · -- Kept and overlapping `i`: `k` is suppressed, with `i` as its witness.
          simp only [↓reduceIte]
          step*
          rcases x with ⟨x1, back⟩
          simp only at x_post1
          subst x_post1
          step*
          have hw : ∀ m, flag (v.set k true).val m = if m = k.val then true else flag v.val m :=
            fun m => by rw [alloc.vec.Vec.set_val_eq]; exact flag_set _ _ _ hkv
          refine ⟨by scalar_tac, by scalar_tac, ⟨?_, ?_, ?_, ?_, ?_⟩, by scalar_tac⟩
          · simp_all [alloc.vec.Vec.set_val_eq]
          · intro m hm
            rw [hw]
            split
            · omega
            · exact hkeep m hm
          · intro m hm
            rw [hw]
            split
            · rfl
            · exact hmono m hm
          · intro m hm
            rw [hw] at hm
            split at hm
            · subst_vars
              exact Or.inr ⟨hik, hc.symm⟩
            · exact hnew m hm
          · intro m hm1 hm2 hm3
            rw [hw] at hm3
            split at hm3
            · contradiction
            · exact hdone m hm1 (by omega) hm3
      · -- Already suppressed: it stays so and cannot be a kept counterexample.
        simp only [↓reduceIte]
        step*
        refine ⟨by scalar_tac, by scalar_tac, ⟨hlen', hkeep, hmono, hnew, ?_⟩, by scalar_tac⟩
        intro m hm1 hm2 hm3
        by_cases hmk : m = k.val
        · subst hmk
          rw [hflag, ← hb] at hm3
          contradiction
        · exact hdone m hm1 (by omega) hm3
    · have he : k.val = len.val := by scalar_tac
      simp_all [WP.spec_ok, Scanned]
  · exact ⟨hij, hj, hscan⟩

theorem outer_spec {F : Type} (inst : core.ops.function.Fn F (Usize × Usize) Bool)
    (overlaps : F) (R : Nat → Nat → Bool)
    (hcall : ∀ a b : Usize, inst.call overlaps (a, b) ⦃ r => r = R a.val b.val ⦄)
    (len i : Usize) (s : alloc.vec.Vec Bool)
    (hlen : s.length = len.val) (hi : i.val ≤ len.val)
    (hind : Independent R s.val i.val) (hcov : Covered R s.val i.val) :
    greedy.greedy_suppression_loop0 inst len overlaps s i
      ⦃ result => result.length = len.val ∧ Independent R result.val len.val ∧
        Covered R result.val len.val ⦄ := by
  unfold greedy.greedy_suppression_loop0
  apply loop.spec_decr_nat (fun (_, k) => len.val - k.val)
    (fun (v, k) => k.val ≤ len.val ∧ v.length = len.val ∧ Independent R v.val k.val ∧
      Covered R v.val k.val)
  · rintro ⟨v, k⟩ ⟨hk, hvlen, hvind, hvcov⟩
    unfold greedy.greedy_suppression_loop0.body
    dsimp only
    split
    · have hkv : k.val < v.length := by scalar_tac
      have hflag : flag v.val k.val = v.val[k.val] := by
        simp [flag, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hkv]
      step as ⟨b, hb⟩
      cases b
      · -- `k` is kept: it suppresses every later entry it overlaps.
        simp only [Bool.false_eq_true, ↓reduceIte]
        step as ⟨j, hj⟩
        apply WP.spec_bind (inner_spec inst overlaps R hcall len k j v v hvlen
          (by scalar_tac) (by scalar_tac)
          ⟨rfl, fun _ _ => rfl, fun _ h => h, fun _ h => Or.inl h,
            fun m h1 h2 _ => by scalar_tac⟩)
        rintro w ⟨hwlen, hwkeep, hwmono, hwnew, hwdone⟩
        step*
        have hfalse : ∀ m, flag w.val m = false → flag v.val m = false := by
          intro m hm
          cases h : flag v.val m
          · rfl
          · have := hwmono m h
            simp_all
        refine ⟨by scalar_tac, by scalar_tac, ?_, ?_, by scalar_tac⟩
        · intro a c ha hac hc hfa hfc
          by_cases hak : a = k.val
          · subst hak
            exact hwdone c hac (by scalar_tac) hfc
          · exact hvind a c (by scalar_tac) hac (by scalar_tac) (hfalse a hfa) (hfalse c hfc)
        · intro c hc hfc
          rcases hwnew c hfc with hold | ⟨hkc, hR⟩
          · obtain ⟨a, hac, hak, hfa, hRa⟩ := hvcov c (by scalar_tac) hold
            exact ⟨a, hac, by scalar_tac, by rw [hwkeep a (by omega)]; exact hfa, hRa⟩
          · refine ⟨k.val, hkc, by scalar_tac, ?_, hR⟩
            rw [hwkeep k.val (le_refl _), hflag, ← hb]
      · -- `k` was already suppressed: it is never a kept member of any pair.
        simp only [↓reduceIte]
        step*
        refine ⟨by scalar_tac, hvlen, ?_, ?_, by scalar_tac⟩
        · intro a c ha hac hc hfa hfc
          by_cases hak : a = k.val
          · subst hak
            rw [hflag, ← hb] at hfa
            contradiction
          · exact hvind a c (by scalar_tac) hac hc hfa hfc
        · intro c hc hfc
          obtain ⟨a, hac, hak, hfa, hRa⟩ := hvcov c hc hfc
          exact ⟨a, hac, by scalar_tac, hfa, hRa⟩
    · have he : k.val = len.val := by scalar_tac
      simp_all [WP.spec_ok]
  · exact ⟨hi, hlen, hind, hcov⟩

/-- For every overlap relation: one flag per detection, no two kept detections overlap, and
each suppressed one overlaps an earlier kept detection. -/
theorem greedy_suppression_spec {F : Type} (inst : core.ops.function.Fn F (Usize × Usize) Bool)
    (overlaps : F) (R : Nat → Nat → Bool)
    (hcall : ∀ a b : Usize, inst.call overlaps (a, b) ⦃ r => r = R a.val b.val ⦄)
    (len : Usize) :
    greedy.greedy_suppression inst len overlaps ⦃ s =>
      s.length = len.val ∧ Independent R s.val len.val ∧ Covered R s.val len.val ⦄ := by
  unfold greedy.greedy_suppression
  step as ⟨s, hs, hslen⟩
  apply outer_spec inst overlaps R hcall len 0#usize s hslen (by scalar_tac)
  · intro a _ ha
    simp at ha
  · intro c hc hfc
    simp [flag, hs, List.getD_eq_getElem?_getD] at hfc

#print axioms greedy_suppression_spec
