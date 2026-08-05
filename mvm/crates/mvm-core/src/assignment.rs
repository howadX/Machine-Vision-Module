//! Optimal bipartite assignment (phase 9.10) — **pure arithmetic, zero deps**.
//!
//! The tracker associates detections to tracks. Until now that match was
//! **greedy and order-dependent** (`tracker::associate_core`): each detection
//! independently grabbed the free track with the best `iou + W_OCM·momentum`.
//! Greedy is locally optimal but globally wrong on ambiguity — when two crossing
//! bodies share overlapping predictions, the first detection can steal the track
//! the second one needed, forcing a swap or a phantom id. A swap mis-routes
//! identity/emotion onto the wrong body, the most expensive error under the
//! track-gating of 9.6/9.7.
//!
//! This module owns the **cost-matrix → assignment** layer: a hand-rolled
//! Kuhn–Munkres (the Hungarian algorithm) that minimises the *total* cost across
//! all rows and columns at once. `O(n³)`, trivial for the handful of bodies (or,
//! in phase 10, overlapping objects) we ever track. Zero dependencies — it is
//! scalar arithmetic over a cost matrix the caller already built (invariant #1).
//!
//! ## Contract
//!
//! - Rows and columns mean whatever the caller wants (we use rows = tracks,
//!   cols = detections). The result maps **row → column**.
//! - Rectangular matrices are supported: with more rows than columns some rows
//!   stay `None`; with more columns than rows some columns are simply unused.
//! - A cell of `f32::INFINITY` marks a **forbidden** pair (e.g. below the IoU
//!   gate). A forbidden pair is never returned: the optimiser routes around it,
//!   and a row whose only options are all forbidden comes back `None`.
//! - The solver minimises cost. The caller converts its *score* (higher = better)
//!   to a cost (e.g. `-score`) before calling, and may re-gate the returned
//!   assignment by cost afterwards.
//! - Deterministic: equal-cost ties resolve to the **lowest column index**, so
//!   behaviour is reproducible (required for the 1–2 body identity tests).

/// Solve the assignment problem for `cost` (rows × columns), minimising the total
/// cost of the chosen row↔column pairs. Returns `assignment` where
/// `assignment[i] = Some(j)` if row `i` is matched to column `j`, else `None`.
///
/// `f32::INFINITY` cells are forbidden and never chosen. See the module docs for
/// the full contract.
pub fn solve(cost: &[Vec<f32>]) -> Vec<Option<usize>> {
    let rows = cost.len();
    let cols = if rows == 0 { 0 } else { cost[0].len() };
    if rows == 0 {
        return Vec::new();
    }
    if cols == 0 {
        return vec![None; rows];
    }

    // The compact Kuhn–Munkres below requires (logical) rows ≤ columns. Orient
    // the matrix so the smaller dimension indexes the rows we iterate over;
    // `transposed` remembers to invert the row↔column mapping at the end.
    let transposed = rows > cols;
    let (n, m) = if transposed {
        (cols, rows)
    } else {
        (rows, cols)
    };
    let at = |i: usize, j: usize| -> f32 { if transposed { cost[j][i] } else { cost[i][j] } };

    // Forbidden cells (`INFINITY`) are replaced by a finite `big` sentinel large
    // enough that no feasible assignment would ever prefer one — `big` exceeds
    // the sum of all finite magnitudes, so a single sentinel outweighs every real
    // pairing combined. After solving we drop any pair that landed on a sentinel.
    let finite_abs_sum: f64 = (0..n)
        .flat_map(|i| (0..m).map(move |j| (i, j)))
        .map(|(i, j)| at(i, j) as f64)
        .filter(|c| c.is_finite())
        .map(f64::abs)
        .sum();
    let big = finite_abs_sum + 1.0;
    let c = |i: usize, j: usize| -> f64 {
        let v = at(i, j) as f64;
        if v.is_finite() { v } else { big }
    };

    // Compact O(n³) Hungarian via dual potentials (Jonker–Volgenant skeleton),
    // 1-indexed. `p[j]` = row currently matched to column `j` (0 = none).
    const INF: f64 = f64::INFINITY;
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; m + 1];
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![INF; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = INF;
            let mut j1 = 0usize;
            // Increasing `j` with strict `<` keeps the lowest column index on a
            // tie → deterministic assignment.
            for j in 1..=m {
                if used[j] {
                    continue;
                }
                let cur = c(i0 - 1, j - 1) - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        // Augment along the alternating path back to the free column.
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    // `p[j]` (j = 1..=m) → row assigned to column j. Build the row→column map in
    // the ORIGINAL orientation, dropping any pair that fell on a forbidden
    // sentinel (its original cost was non-finite).
    let mut result = vec![None; rows];
    for (j, &i) in p.iter().enumerate().take(m + 1).skip(1) {
        if i == 0 {
            continue; // column unused (n < m)
        }
        // (i, j) are 1-indexed in the oriented matrix; (i-1, j-1) are 0-indexed.
        if !at(i - 1, j - 1).is_finite() {
            continue; // forbidden pair — leave both endpoints unmatched
        }
        let (orig_row, orig_col) = if transposed {
            (j - 1, i - 1)
        } else {
            (i - 1, j - 1)
        };
        result[orig_row] = Some(orig_col);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Total cost of an assignment, for optimality assertions in tests.
    fn assignment_cost(cost: &[Vec<f32>], assign: &[Option<usize>]) -> f32 {
        assign
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| c.map(|j| cost[i][j]))
            .sum()
    }

    /// Brute-force optimum over all row→column injections, for cross-checking the
    /// solver on small matrices. Mirrors the solver's semantics: **maximum
    /// feasible cardinality first, minimum cost second** (the solver pays a
    /// `big` sentinel per forbidden pair, so it matches as many real pairs as
    /// possible, then minimises cost). Returns that min cost; a row is left
    /// unassigned only when every remaining column is taken or forbidden.
    fn brute_force_min(cost: &[Vec<f32>]) -> f32 {
        let cols = if cost.is_empty() { 0 } else { cost[0].len() };
        let mut used = vec![false; cols];
        // best = (cardinality desc, cost asc).
        let mut best: (usize, f32) = (0, f32::INFINITY);
        fn rec(
            cost: &[Vec<f32>],
            row: usize,
            cols: usize,
            used: &mut [bool],
            count: usize,
            acc: f32,
            best: &mut (usize, f32),
        ) {
            if row == cost.len() {
                if count > best.0 || (count == best.0 && acc < best.1) {
                    *best = (count, acc);
                }
                return;
            }
            let mut any = false;
            for j in 0..cols {
                if used[j] || cost[row][j].is_infinite() {
                    continue;
                }
                any = true;
                used[j] = true;
                rec(
                    cost,
                    row + 1,
                    cols,
                    used,
                    count + 1,
                    acc + cost[row][j],
                    best,
                );
                used[j] = false;
            }
            // Leave this row unassigned only when it has no feasible column, or to
            // explore freeing a column for a later row (kept general for safety).
            let _ = any;
            rec(cost, row + 1, cols, used, count, acc, best);
        }
        rec(cost, 0, cols, &mut used, 0, 0.0, &mut best);
        best.1
    }

    #[test]
    fn empty_matrix_returns_empty() {
        let cost: Vec<Vec<f32>> = vec![];
        assert!(solve(&cost).is_empty());
    }

    #[test]
    fn zero_columns_all_unmatched() {
        let cost: Vec<Vec<f32>> = vec![vec![], vec![]];
        assert_eq!(solve(&cost), vec![None, None]);
    }

    #[test]
    fn single_cell_matches() {
        let cost = vec![vec![3.0]];
        assert_eq!(solve(&cost), vec![Some(0)]);
    }

    #[test]
    fn single_row_picks_min_column() {
        let cost = vec![vec![5.0, 2.0, 9.0]];
        assert_eq!(solve(&cost), vec![Some(1)]);
    }

    #[test]
    fn diagonal_optimum() {
        // Cheap on the diagonal → each row takes its own column.
        let cost = vec![
            vec![0.0, 9.0, 9.0],
            vec![9.0, 0.0, 9.0],
            vec![9.0, 9.0, 0.0],
        ];
        assert_eq!(solve(&cost), vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn beats_greedy_on_ambiguity() {
        // Greedy by row: row0 grabs col0 (1) → row1 forced to col1 (9), total 10.
        // Optimal: row0→col1 (2), row1→col0 (2), total 4. Hungarian must find 4.
        let cost = vec![vec![1.0, 2.0], vec![2.0, 9.0]];
        let a = solve(&cost);
        assert_eq!(a, vec![Some(1), Some(0)]);
        assert!((assignment_cost(&cost, &a) - 4.0).abs() < 1e-5);
    }

    #[test]
    fn rectangular_more_columns_uses_distinct() {
        // 2 rows, 3 cols → 2 assignments to distinct columns, one col unused.
        let cost = vec![vec![4.0, 1.0, 7.0], vec![2.0, 5.0, 3.0]];
        let a = solve(&cost);
        assert_eq!(a.len(), 2);
        let cols: Vec<usize> = a.iter().filter_map(|&c| c).collect();
        assert_eq!(cols.len(), 2, "both rows matched");
        assert_ne!(cols[0], cols[1], "distinct columns");
        // Optimal: row0→col1 (1), row1→col0 (2) = 3.
        assert!((assignment_cost(&cost, &a) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn rectangular_more_rows_leaves_one_unmatched() {
        // 3 rows, 2 cols → exactly one row is None.
        let cost = vec![vec![1.0, 8.0], vec![8.0, 1.0], vec![5.0, 5.0]];
        let a = solve(&cost);
        assert_eq!(a.len(), 3);
        let matched = a.iter().filter(|c| c.is_some()).count();
        assert_eq!(matched, 2, "two rows matched, one dropped");
        // Distinct columns among the matched.
        let cols: Vec<usize> = a.iter().filter_map(|&c| c).collect();
        assert_ne!(cols[0], cols[1]);
        // Optimal drops row2 (the expensive one): row0→0, row1→1 = 2.
        assert!((assignment_cost(&cost, &a) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn forbidden_cell_never_chosen() {
        // row0 prefers col0 by cost, but col0 is forbidden → routes to col1.
        let cost = vec![vec![f32::INFINITY, 5.0], vec![2.0, 9.0]];
        let a = solve(&cost);
        assert_eq!(a, vec![Some(1), Some(0)]);
    }

    #[test]
    fn fully_forbidden_row_is_unmatched() {
        // row1 has no feasible column → None; row0 still matched.
        let cost = vec![vec![3.0, 7.0], vec![f32::INFINITY, f32::INFINITY]];
        let a = solve(&cost);
        assert_eq!(a[1], None, "all-forbidden row unmatched");
        assert!(a[0].is_some(), "feasible row still matched");
    }

    #[test]
    fn deterministic_lowest_column_tiebreak() {
        // Single row, equal costs → lowest column index wins, reproducibly.
        let cost = vec![vec![2.0, 2.0, 2.0]];
        assert_eq!(solve(&cost), vec![Some(0)]);
    }

    #[test]
    fn matches_brute_force_on_4x4() {
        // A non-trivial square with a unique-ish optimum; cross-check the total
        // cost against exhaustive search.
        let cost = vec![
            vec![7.0, 2.0, 5.0, 8.0],
            vec![4.0, 9.0, 1.0, 3.0],
            vec![6.0, 5.0, 8.0, 2.0],
            vec![1.0, 4.0, 6.0, 7.0],
        ];
        let a = solve(&cost);
        let got = assignment_cost(&cost, &a);
        let want = brute_force_min(&cost);
        assert!((got - want).abs() < 1e-4, "got {got}, optimum {want}");
        // Every row matched to a distinct column (square, all feasible).
        let mut cols: Vec<usize> = a.iter().filter_map(|&c| c).collect();
        cols.sort_unstable();
        assert_eq!(cols, vec![0, 1, 2, 3]);
    }

    #[test]
    fn matches_brute_force_rectangular_with_forbidden() {
        // 3 rows × 4 cols, a couple forbidden — cross-check against brute force.
        let cost = vec![
            vec![5.0, f32::INFINITY, 3.0, 8.0],
            vec![2.0, 6.0, f32::INFINITY, 4.0],
            vec![9.0, 1.0, 7.0, 5.0],
        ];
        let a = solve(&cost);
        let got = assignment_cost(&cost, &a);
        let want = brute_force_min(&cost);
        assert!((got - want).abs() < 1e-4, "got {got}, optimum {want}");
        // No forbidden pair chosen.
        for (i, &c) in a.iter().enumerate() {
            if let Some(j) = c {
                assert!(cost[i][j].is_finite(), "row {i} chose forbidden col {j}");
            }
        }
    }

    #[test]
    fn negative_costs_supported() {
        // Costs may be negative (we pass -score). Optimum still global.
        let cost = vec![vec![-1.0, -3.0], vec![-2.0, -2.0]];
        let a = solve(&cost);
        let got = assignment_cost(&cost, &a);
        let want = brute_force_min(&cost);
        assert!((got - want).abs() < 1e-5, "got {got}, optimum {want}");
    }
}
