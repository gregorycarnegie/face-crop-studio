import CropRegion

open Aeneas Aeneas.Std Result ControlFlow
open crop_region

set_option maxHeartbeats 3000000

theorem retained_loop_spec (outputs : Slice Usize) (slot index : Usize)
    (hi : index.val ≤ outputs.length) (hseen : slot ∉ outputs.val.take index.val) :
    slots.retained_loop outputs slot index ⦃ result => result = true ↔ slot ∈ outputs.val ⦄ := by
  unfold slots.retained_loop
  apply loop.spec_decr_nat (fun i => outputs.length - i.val)
    (fun i => i.val ≤ outputs.length ∧ slot ∉ outputs.val.take i.val)
  · intro i h
    rcases h with ⟨hi, hseen⟩
    unfold slots.retained_loop.body
    step*
    · simp only [true_iff]
      apply List.mem_iff_getElem.mpr
      exact ⟨i.val, by scalar_tac, by simp_all⟩
    · refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      rw [i3_post, List.take_succ_eq_append_getElem (by scalar_tac : i.val < outputs.length)]
      simp only [List.mem_append, List.mem_singleton, not_or]
      refine ⟨hseen, ?_⟩
      intro heq
      scalar_tac
    · have he : i.val = outputs.length := by scalar_tac
      simp_all
  · exact ⟨hi, hseen⟩

@[step] theorem retained_spec (outputs : Slice Usize) (slot : Usize) :
    slots.retained outputs slot ⦃ result => result = true ↔ slot ∈ outputs.val ⦄ := by
  unfold slots.retained
  apply retained_loop_spec
  · scalar_tac
  · simp

#print axioms retained_spec

def readsAt (steps : List (Usize × Usize)) (index slot : Nat) : Bool :=
  match steps[index]? with
  | none => false
  | some (a, b) => decide (a.val = slot ∨ b.val = slot)

def latestReader (steps : List (Usize × Usize)) (slot : Nat) : Nat → Nat
  | 0 => 0
  | i + 1 => if readsAt steps i slot then i + 1 else latestReader steps slot i

@[step] theorem readers_loop_spec (steps : Slice (Usize × Usize)) (last : alloc.vec.Vec Usize) (index : Usize)
    (hi : index.val ≤ steps.length) (hlen : last.length = steps.length + 1)
    (hvalid : ∀ i (h : i < steps.length), (steps.val[i]).1.val < last.length ∧ (steps.val[i]).2.val < last.length)
    (hmap : ∀ j (h : j < last.length), (last.val[j]).val = latestReader steps.val j index.val) :
    slots.readers_loop steps last index ⦃ result =>
      result.length = steps.length + 1 ∧
      ∀ j (h : j < result.length), (result.val[j]).val = latestReader steps.val j steps.length ⦄ := by
  unfold slots.readers_loop
  apply loop.spec_decr_nat (fun (_, i) => steps.length - i.val)
    (fun (v, i) => i.val ≤ steps.length ∧ v.length = steps.length + 1 ∧
      ∀ j (h : j < v.length), (v.val[j]).val = latestReader steps.val j i.val)
  · rintro ⟨v, i⟩ ⟨hi, hv, hm⟩
    unfold slots.readers_loop.body
    dsimp only
    split
    · step as ⟨p, hp⟩
      rcases p with ⟨a, b⟩
      have hbounds := hvalid i.val (by scalar_tac)
      rw [← hp] at hbounds
      dsimp only at hbounds ⊢
      step*
      all_goals simp_all [alloc.vec.Vec.set]
      · scalar_tac
      · refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        intro j hj
        have hread : readsAt steps.val i.val j = decide (a.val = j ∨ b.val = j) := by
          simp only [readsAt, List.getElem?_eq_getElem (by scalar_tac : i.val < steps.length), ← hp]
        simp only [latestReader, hread]
        by_cases ha : j = a.val <;> by_cases hb : j = b.val <;>
          simp_all [List.getElem_set]
        all_goals omega
    · have he : i.val = steps.length := by scalar_tac
      simp_all [WP.spec_ok]
  · exact ⟨hi, hlen, hmap⟩

@[step] theorem readers_spec (steps : Slice (Usize × Usize))
    (hsize : steps.length < Usize.max)
    (hvalid : ∀ i (h : i < steps.length), (steps.val[i]).1.val ≤ steps.length ∧ (steps.val[i]).2.val ≤ steps.length) :
    slots.readers steps ⦃ last => last.length = steps.length + 1 ∧
      ∀ j (h : j < last.length), (last.val[j]).val = latestReader steps.val j steps.length ⦄ := by
  unfold slots.readers
  step*
  all_goals simp_all [latestReader]

theorem latest_reader_bound (steps : List (Usize × Usize)) (slot n : Nat) :
    latestReader steps slot n ≤ n := by
  induction n with
  | zero => simp [latestReader]
  | succ n ih => simp only [latestReader]; split <;> omega

theorem latest_reader_after_read (steps : List (Usize × Usize)) (slot n i : Nat)
    (hi : i < n) (hread : readsAt steps i slot = true) :
    i + 1 ≤ latestReader steps slot n := by
  induction n with
  | zero => omega
  | succ n ih =>
    simp only [latestReader]
    split
    · omega
    · have hne : i ≠ n := by intro he; subst i; contradiction
      have h := ih (by omega)
      omega

#print axioms readers_spec
#print axioms latest_reader_after_read

def scheduleSafe (last outputs : Slice Usize) (frees : alloc.vec.Vec (alloc.vec.Vec Usize)) : Prop :=
  ∀ row (hrow : row < frees.length) slot, slot ∈ frees.val[row].val →
    0 < slot.val ∧ slot.val < last.length ∧ slot ∉ outputs.val ∧
    (last.val[slot.val]?.map (fun v => v.val)) = some (row + 1)

theorem schedule_push_safe (last outputs : Slice Usize)
    (frees : alloc.vec.Vec (alloc.vec.Vec Usize)) (row slot : Usize) (newRow : alloc.vec.Vec Usize)
    (hr : row.val < frees.length) (hsafe : scheduleSafe last outputs frees)
    (hnew : newRow.val = frees.val[row.val].val ++ [slot])
    (hslot : 0 < slot.val ∧ slot.val < last.length ∧ slot ∉ outputs.val ∧
      (last.val[slot.val]?.map (fun v => v.val)) = some (row.val + 1)) :
    scheduleSafe last outputs (frees.set row newRow) := by
  intro i hi s hm
  by_cases he : i = row.val
  · subst i
    have hn : s ∈ newRow.val := by simpa [alloc.vec.Vec.set] using hm
    rw [hnew] at hn
    simp only [List.mem_append, List.mem_singleton] at hn
    rcases hn with hold | heq
    · exact hsafe row.val hr s hold
    · subst s
      exact hslot
  · have hio : i < frees.val.length := by simpa [alloc.vec.Vec.set] using hi
    have hold : s ∈ frees.val[i].val := by
      simpa [alloc.vec.Vec.set, List.getElem_set, Ne.symm he] using hm
    apply hsafe i (by simpa [alloc.vec.Vec.set] using hi) s hold

@[step] theorem release_loop_spec (last outputs : Slice Usize)
    (frees : alloc.vec.Vec (alloc.vec.Vec Usize)) (slot : Usize)
    (hslot : 1 ≤ slot.val ∧ slot.val ≤ last.length)
    (hvalues : ∀ j (h : j < last.length), (last.val[j]).val ≤ frees.length)
    (hsafe : scheduleSafe last outputs frees)
    (hrows : ∀ i (h : i < frees.length), (frees.val[i]).length < slot.val) :
    slots.release_schedule_loop last outputs frees slot ⦃ result =>
      result.length = frees.length ∧ scheduleSafe last outputs result ⦄ := by
  unfold slots.release_schedule_loop
  apply loop.spec_decr_nat (fun (_, i) => last.length - i.val)
    (fun (v, i) => 1 ≤ i.val ∧ i.val ≤ last.length ∧ v.length = frees.length ∧
      scheduleSafe last outputs v ∧ ∀ j (h : j < v.length), (v.val[j]).length < i.val)
  · rintro ⟨v, i⟩ ⟨hi0, hi, hv, hsafe, hrows⟩
    unfold slots.release_schedule_loop.body
    dsimp only
    split
    · step as ⟨atIndex, hat⟩
      have hvalue := hvalues i.val (by scalar_tac)
      split
      · step as ⟨keep, hkeep⟩
        split
        · step*
          refine ⟨by scalar_tac, by scalar_tac, hv, hsafe, ?_, by scalar_tac⟩
          intro j hj
          have h := hrows j hj
          scalar_tac
        · step as ⟨row, hrow⟩
          step as ⟨p, hp, hback⟩
          rcases p with ⟨oldRow, back⟩
          dsimp only at hp hback ⊢
          have hrange : row.val < v.length := by scalar_tac
          have hcap := hrows row.val hrange
          step*
          rw [hback]
          have hnew : x.val = v.val[row.val].val ++ [i] := by simpa only [hp] using x_post
          have hslotgood : 0 < i.val ∧ i.val < last.length ∧ i ∉ outputs.val ∧
              (last.val[i.val]?.map (fun x => x.val)) = some (row.val + 1) := by
            refine ⟨by scalar_tac, by scalar_tac, ?_, ?_⟩
            · intro hm
              have hh := hkeep.mpr hm
              contradiction
            · simp only [List.getElem?_eq_getElem (by scalar_tac : i.val < last.length), Option.map_some, Option.some.injEq]
              scalar_tac
          have hsafeNew := schedule_push_safe last outputs v row i x hrange hsafe hnew hslotgood
          have hlength : (v.set row x).length = v.length := by simp [alloc.vec.Vec.set]
          refine ⟨by scalar_tac, by scalar_tac, by simpa only [hlength] using hv, hsafeNew, ?_, by scalar_tac⟩
          intro j hj
          have hjo : j < v.val.length := by simpa [alloc.vec.Vec.set] using hj
          by_cases he : j = row.val
          · subst j
            have hlennew : x.length = oldRow.length + 1 := by simpa using congrArg List.length x_post
            have hcapold : oldRow.length < i.val := by simpa only [hp] using hcap
            have hbnd : x.length < slot1.val := by omega
            simpa [alloc.vec.Vec.set] using hbnd
          · have hold := hrows j (by simpa using hjo)
            have hbnd : (v.val[j]).length < slot1.val := by omega
            simpa [alloc.vec.Vec.set, List.getElem_set, Ne.symm he] using hbnd
      · step*
        refine ⟨by scalar_tac, by scalar_tac, hv, hsafe, ?_, by scalar_tac⟩
        intro j hj
        have h := hrows j hj
        scalar_tac
    · simp_all [WP.spec_ok]
  · exact ⟨hslot.1, hslot.2, rfl, hsafe, hrows⟩

theorem empty_schedule_loop_spec (steps : Usize) (frees : alloc.vec.Vec (alloc.vec.Vec Usize)) (index : Usize)
    (hi : index.val ≤ steps.val) (hmap : frees.val = List.replicate index.val (alloc.vec.Vec.new Usize)) :
    slots.empty_schedule_loop steps frees index ⦃ result => result.val = List.replicate steps.val (alloc.vec.Vec.new Usize) ⦄ := by
  unfold slots.empty_schedule_loop
  apply loop.spec_decr_nat (fun (_, i) => steps.val - i.val)
    (fun (v, i) => i.val ≤ steps.val ∧ v.val = List.replicate i.val (alloc.vec.Vec.new Usize))
  · rintro ⟨v, i⟩ ⟨hi, hm⟩
    have hv : v.length = i.val := by simpa using congrArg List.length hm
    unfold slots.empty_schedule_loop.body
    step*
    all_goals simp_all [List.replicate_add]
    all_goals scalar_tac
  · exact ⟨hi, hmap⟩

@[step] theorem empty_schedule_spec (steps : Usize) :
    slots.empty_schedule steps ⦃ result => result.val = List.replicate steps.val (alloc.vec.Vec.new Usize) ⦄ := by
  unfold slots.empty_schedule
  apply empty_schedule_loop_spec
  · scalar_tac
  · simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.new]

@[step] theorem release_schedule_spec (last outputs : Slice Usize) (steps : Usize)
    (hlen : last.length = steps.val + 1)
    (hvalues : ∀ j (h : j < last.length), (last.val[j]).val ≤ steps.val) :
    slots.release_schedule last outputs steps ⦃ frees => frees.length = steps.val ∧ scheduleSafe last outputs frees ⦄ := by
  unfold slots.release_schedule
  step*
  all_goals simp_all [scheduleSafe, alloc.vec.Vec.new]

theorem graph_release_safety (steps : Slice (Usize × Usize)) (outputs : Slice Usize)
    (hsize : steps.length < Usize.max)
    (hvalid : ∀ i (h : i < steps.length), (steps.val[i]).1.val ≤ steps.length ∧ (steps.val[i]).2.val ≤ steps.length) :
    slots.last_reads steps outputs ⦃ frees =>
      frees.length = steps.length ∧
      ∀ row (hrow : row < frees.length) slot, slot ∈ frees.val[row].val →
        0 < slot.val ∧ slot ∉ outputs.val ∧
        ∀ i : Nat, i < steps.length → readsAt steps.val i slot.val = true → i ≤ row ⦄ := by
  unfold slots.last_reads
  step as ⟨last, hlen, hmap⟩
  have hvalues : ∀ j (h : j < last.length), (last.val[j]).val ≤ steps.length := by
    intro j hj
    rw [hmap j hj]
    exact latest_reader_bound steps.val j steps.length
  apply WP.spec_mono (release_schedule_spec (alloc.vec.Vec.deref last) outputs (Slice.len steps)
    (by simpa [alloc.vec.Vec.deref, Slice.from, Slice.val, Data.ListN.ListN.from_to_inverse] using hlen)
    (by simpa [alloc.vec.Vec.deref, Slice.from, Slice.val, Data.ListN.ListN.from_to_inverse] using hvalues))
  intro frees hf
  rcases hf with ⟨hf, hs⟩
  refine ⟨by simpa using hf, ?_⟩
  intro row hr slot hmem
  obtain ⟨hpositive, hbound, hretained, hlast⟩ := hs row hr slot hmem
  refine ⟨hpositive, hretained, ?_⟩
  intro i hi hread
  have hbound' : slot.val < last.length := by
    simpa [alloc.vec.Vec.deref, Slice.from, Slice.val, Data.ListN.ListN.from_to_inverse] using hbound
  have hlastval := hmap slot.val hbound'
  simp only [alloc.vec.Vec.deref, Slice.from, Slice.val, Data.ListN.ListN.from_to_inverse] at hlast
  simp only [List.getElem?_eq_getElem hbound', Option.map_some, Option.some.injEq] at hlast
  have hafter := latest_reader_after_read steps.val slot.val steps.length i hi hread
  omega

#print axioms release_loop_spec
#print axioms release_schedule_spec
#print axioms graph_release_safety
