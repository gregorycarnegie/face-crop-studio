# Kani checks of production floating-point code

Lean's Aeneas pipeline cannot translate floating-point arithmetic (see
[LIMITS.md](../verification/LIMITS.md)). Kani can: it model-checks the compiled Rust
with bit-precise IEEE 754 `f32`, so each harness covers every input it allows, not
a sample. `lib.rs` includes `fcs-core/src/postprocess.rs`, `fcs-core/src/nms.rs` and
`fcs-core/src/scrfd/decode.rs` directly. A shim crate stands in for `fcs-utils`, so
there are no GPU or codec dependencies.

## Harnesses

- `iou_is_nonnegative_and_positive_only_on_overlap`: for every pair of boxes with
  fields within ±1e18, IoU computes no NaN at any step, is non-negative, and is
  positive only for overlapping boxes. It is **not** at most 1: rounding `x + width`
  can inflate the intersection (a random search found 1.23 for 0.001-pixel boxes
  near x = 3652). NMS only compares IoU against a threshold, so this is harmless.
- `subtracting_the_scene_origin_preserves_order`: for finite values, subtracting
  the grid origin keeps their order.
- `cell_of_is_monotone_and_in_range`: a later position never maps to an earlier
  grid cell, and every cell index is on the grid.
- `sigmoid_is_a_probability_split_at_zero`: the score activation is in [0, 1], is
  above 0.5 only for positive logits, and is at least 0.5 for every positive one.
  Logits in (0, ~1e-7) give exactly 0.5, so "above 0.5 exactly when positive" is
  false in `f32`. Kani does not model `expf`, so the harness replaces it with a stated
  contract: finite, non-negative, exactly 1 at zero, at most 1 below, at least 1
  above. That contract is trusted. A unit test in `decode.rs` covers `exp` overflowing.
- `decode_row_is_finite_and_ordered`: at each of the three head levels, with raw
  values within ±1e4 and letterbox scale in [1e-3, 1e3], the decoded box and both
  eye landmarks are finite, the box has non-negative size whenever its distances are
  non-negative, and the three untrained landmarks stay absent.

With IEEE division being monotone in its numerator for a positive divisor, the
order and cell lemmas give the spatial grid its premise: two overlapping boxes have
intersecting cell ranges, so the grid compares every pair the naive search could
suppress. The selection logic is proved separately in Lean (`GreedyProofs.lean`).

## Not checked

- Harnesses that pile up symbolic `f32` divisions did not finish. Over every finite
  input they ran 50 minutes with both the default solver and CaDiCaL, and with the
  ±1e18 bound they gave no result after more than 30 minutes. That covered IoU
  symmetry, `grid_cell_index` with its division, and the end-to-end grid harness. The
  grid's division step therefore rests on IEEE 754 correct rounding rather than a
  checked theorem.
- Without the ±1e18 bound, IoU's result still passed all three assertions. Kani's
  NaN check then flagged `union = inf - inf` once both areas overflow; `iou`
  returns 0 for it. It also flagged divisions by infinity, here and in the sigmoid,
  which IEEE defines as 0, not NaN; replaying the IoU case in plain Rust confirms
  `1.08e-19 / inf = 0`.

## Run

CI runs this in `.github/workflows/kani.yml` when the included sources change. Locally,
on Linux or WSL, after `cargo install --locked kani-verifier && cargo kani setup`
(about 4 minutes):

```powershell
wsl -d Ubuntu -- bash tools/kani/check.sh
```
