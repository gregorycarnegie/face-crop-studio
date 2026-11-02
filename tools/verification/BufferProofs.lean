import CropRegion

open Aeneas Aeneas.Std Result
open crop_region

set_option maxHeartbeats 2000000

theorem budget_fits_spec (current size limit : U64) :
    buffer_math.budget_fits current size limit ⦃ fits =>
      fits = true ↔ current.val + size.val ≤ limit.val ⦄ := by
  unfold buffer_math.budget_fits
  step*

def bestFit (seen : List (Usize × U64)) (requested : Nat) (best : Option (Usize × U64)) : Prop :=
  match best with
  | none => ∀ e ∈ seen, e.2.val < requested
  | some chosen => chosen ∈ seen ∧ requested ≤ chosen.2.val ∧
      ∀ e ∈ seen, requested ≤ e.2.val → chosen.2.val ≤ e.2.val

theorem choose_fit_preserves (seen : List (Usize × U64)) (best : Option (Usize × U64))
    (index : Usize) (size requested : U64) (hbest : bestFit seen requested.val best) :
    buffer_math.choose_fit best index size requested ⦃ result =>
      bestFit (seen ++ [(index, size)]) requested.val result ⦄ := by
  cases best with
  | none =>
    simp only [bestFit] at hbest
    unfold buffer_math.choose_fit
    step*
    all_goals simp_all [bestFit]
    · intro a b hmem
      rcases hmem with hold | ⟨rfl, rfl⟩
      · exact hbest a b hold
      · omega
    · intro a b hmem heligible
      rcases hmem with hold | ⟨rfl, rfl⟩
      · have h := hbest a b hold
        omega
      · omega
  | some old =>
    rcases old with ⟨old_index, old_size⟩
    simp only [bestFit] at hbest
    rcases hbest with ⟨hmem, heligible, hmin⟩
    unfold buffer_math.choose_fit
    step*
    all_goals simp_all [bestFit]
    · intro a b hmem hel
      rcases hmem with hold | ⟨rfl, rfl⟩
      · exact hmin a b hold hel
      · omega
    · intro a b hmem hel
      rcases hmem with hold | ⟨rfl, rfl⟩
      · exact hmin a b hold hel
      · omega
    · intro a b hmem hel
      rcases hmem with hold | ⟨rfl, rfl⟩
      · have h := hmin a b hold hel
        omega
      · omega

/-- Compositional loop model: apply the extracted Rust transition to eligible entries. -/
def scanFits (items : List (Usize × U64)) (requested : U64) (best : Option (Usize × U64)) :
    Result (Option (Usize × U64)) :=
  match items with
  | [] => ok best
  | (i, size) :: tail => do
    let next ← buffer_math.choose_fit best i size requested
    scanFits tail requested next

theorem scan_fits_spec (items seen : List (Usize × U64)) (requested : U64)
    (best : Option (Usize × U64)) (hbest : bestFit seen requested.val best) :
    scanFits items requested best ⦃ result => bestFit (seen ++ items) requested.val result ⦄ := by
  induction items generalizing seen best with
  | nil => simpa [scanFits, WP.spec_ok] using hbest
  | cons item tail ih =>
    rcases item with ⟨i, size⟩
    unfold scanFits
    apply WP.spec_bind (choose_fit_preserves seen best i size requested hbest)
    intro next hnext
    simpa [List.append_assoc] using ih (seen ++ [(i, size)]) next hnext

theorem best_fit_minimum (items : List (Usize × U64)) (requested : U64) :
    scanFits items requested none ⦃ result => bestFit items requested.val result ⦄ := by
  simpa using scan_fits_spec items [] requested none (by simp [bestFit])

theorem exact_fit_is_final (items : List (Usize × U64)) (chosen : Usize × U64) (requested : Nat)
    (hmem : chosen ∈ items) (hexact : chosen.2.val = requested) :
    bestFit items requested (some chosen) := by
  simp only [bestFit]
  refine ⟨hmem, by omega, ?_⟩
  intro e he hel
  omega

#print axioms budget_fits_spec
#print axioms choose_fit_preserves
#print axioms best_fit_minimum
#print axioms exact_fit_is_final
