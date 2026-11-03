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

@[simp, scalar_tac_simps] theorem saturating_u64_sub_val (x y : U64) :
    (core.num.U64.saturating_sub x y).val = x.val - y.val := by
  simp only [core.num.U64.saturating_sub, UScalar.saturating_sub, UScalar.val,
    BitVec.toNat_ofNat, Nat.zero_le, max_eq_right]
  exact Nat.mod_eq_of_lt (by have := x.hmax; omega)

theorem reserve_spec (current size limit : U64) :
    buffer_math.reserve current size limit ⦃ result =>
      match result with
      | none => limit.val < current.val + size.val
      | some next => next.val = current.val + size.val ∧ next.val ≤ limit.val ⦄ := by
  unfold buffer_math.reserve
  apply WP.spec_bind (budget_fits_spec current size limit)
  intro fits hfits
  cases fits
  · simp_all [WP.spec_ok]
  · have := limit.hmax
    step*

theorem release_spec (current freed : U64) :
    buffer_math.release current freed ⦃ result => result.val = current.val - freed.val ⦄ := by
  simp [buffer_math.release, WP.spec_ok]

/-- One atomic `fetch_update` on the allocated-bytes counter. A refused reservation leaves it. -/
inductive CounterOp where
  | reserve (size : U64)
  | release (freed : U64)

def applyOp (limit current : U64) : CounterOp → Result U64
  | .reserve size => do
    let next ← buffer_math.reserve current size limit
    ok (next.getD current)
  | .release freed => buffer_math.release current freed

/-- Updates to one atomic location are totally ordered, so every thread interleaving is a list. -/
def applyOps (limit current : U64) : List CounterOp → Result U64
  | [] => ok current
  | op :: rest => do
    let next ← applyOp limit current op
    applyOps limit next rest

theorem apply_op_within_budget (limit current : U64) (op : CounterOp)
    (h : current.val ≤ limit.val) :
    applyOp limit current op ⦃ next => next.val ≤ limit.val ⦄ := by
  cases op with
  | reserve size =>
    unfold applyOp
    apply WP.spec_bind (reserve_spec current size limit)
    intro result hresult
    cases result <;> simp_all [WP.spec_ok]
    omega
  | release freed =>
    unfold applyOp
    apply WP.spec_mono (release_spec current freed)
    intro next hnext
    omega

theorem counter_within_budget (limit current : U64) (ops : List CounterOp)
    (h : current.val ≤ limit.val) :
    applyOps limit current ops ⦃ final => final.val ≤ limit.val ⦄ := by
  induction ops generalizing current with
  | nil => simpa [applyOps, WP.spec_ok] using h
  | cons op rest ih =>
    unfold applyOps
    apply WP.spec_bind (apply_op_within_budget limit current op h)
    intro next hnext
    exact ih next hnext

#print axioms budget_fits_spec
#print axioms reserve_spec
#print axioms release_spec
#print axioms counter_within_budget
#print axioms choose_fit_preserves
#print axioms best_fit_minimum
#print axioms exact_fit_is_final
