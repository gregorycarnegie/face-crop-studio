import CropRegion

open Aeneas Aeneas.Std Result
open crop_region

/-! fcs-qt's export workers share one counter: each takes `ticket ← fetch_add(1)` and exports
the job `claim ticket jobs` returns, stopping at its first `none`. The atomic applies the
`fetch_add`s one at a time, so whichever workers make them, in whatever interleaving, the
tickets handed out are `0, 1, 2, ...` (a standard-library contract, not proved here). -/

theorem claim_spec (ticket jobs : Usize) :
    claim.claim ticket jobs ⦃ r =>
      r = if ticket.val < jobs.val then some ticket else none ⦄ := by
  unfold claim.claim
  split <;> simp_all [WP.spec_ok]

/-- The jobs claimed by the first `m` tickets, in ticket order, as `claim_spec` gives them. -/
def claimedJobs (jobs m : Nat) : List Nat :=
  (List.range m).filterMap (fun t => if t < jobs then some t else none)

/-- With `workers` workers each stopped at its first `none`, exactly `jobs + workers` tickets
were drawn, and every job was claimed by exactly one of them, whatever the interleaving. -/
theorem every_job_claimed_once (jobs workers : Nat) :
    claimedJobs jobs (jobs + workers) = List.range jobs ∧
      (claimedJobs jobs (jobs + workers)).Nodup := by
  have h : claimedJobs jobs (jobs + workers) = List.range jobs := by
    unfold claimedJobs
    rw [List.range_add, List.filterMap_append]
    have hlow : (List.range jobs).filterMap (fun t => if t < jobs then some t else none) =
        List.range jobs := by
      conv => rhs; rw [← List.filterMap_some (l := List.range jobs)]
      apply List.filterMap_congr
      intro t ht
      simp [List.mem_range.mp ht]
    have hhigh : ((List.range workers).map (jobs + ·)).filterMap
        (fun t => if t < jobs then some t else none) = [] := by
      simp [List.filterMap_map]
    rw [hlow, hhigh, List.append_nil]
  exact ⟨h, h ▸ List.nodup_range⟩

/-- A worker only stops when the jobs are gone: a ticket that claims nothing is past the last
job, so no later ticket claims one either. -/
theorem none_is_final (jobs ticket later : Nat) (hnone : ¬ ticket < jobs)
    (hlater : ticket ≤ later) : ¬ later < jobs := by
  omega

#print axioms claim_spec
#print axioms every_job_claimed_once
#print axioms none_is_final
