//! Minimum-cost assignment between tracks and detections.

/// Matched `(row, column)` pairs, sorted by row, and what was left over.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Assignment {
    pub matches: Vec<(usize, usize)>,
    pub unmatched_rows: Vec<usize>,
    pub unmatched_cols: Vec<usize>,
}

/// Minimum-cost assignment in which a row and a column may also stay
/// unmatched, at a cost of `cost_limit / 2` each.
///
/// This is the formulation of `lap.lapjv(cost, extend_cost=True,
/// cost_limit=...)` that ByteTrack uses: a pair is only made when it is
/// cheaper than leaving both sides unmatched, so no match costs
/// `cost_limit` or more. `costs` is row-major, `rows × cols`. Entries that are
/// not finite are forbidden pairs. With a `cost_limit` that is not positive
/// and finite, nothing is matched.
pub fn assign_with_cost_limit(
    rows: usize,
    cols: usize,
    costs: &[f64],
    cost_limit: f64,
) -> Assignment {
    assert_eq!(costs.len(), rows * cols, "cost matrix is not rows × cols");
    if rows == 0 || cols == 0 || !(cost_limit.is_finite() && cost_limit > 0.0) {
        return Assignment {
            matches: Vec::new(),
            unmatched_rows: (0..rows).collect(),
            unmatched_cols: (0..cols).collect(),
        };
    }
    // Square matrix of size rows + cols: real pairs top left, "row i stays
    // unmatched" top right, "column j stays unmatched" bottom left, and zeros
    // bottom right to pair the dummies with each other. A pair that costs
    // `cost_limit` or more is never better than both dummies, so clamping it
    // (and forbidden pairs) to 2 * cost_limit does not change the optimum.
    let n = rows + cols;
    let half = 0.5 * cost_limit;
    let mut extended = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            extended[r * n + c] = match (r < rows, c < cols) {
                (true, true) => {
                    let cost = costs[r * cols + c];
                    if cost.is_finite() && cost < cost_limit {
                        cost
                    } else {
                        2.0 * cost_limit
                    }
                }
                (true, false) | (false, true) => half,
                (false, false) => 0.0,
            };
        }
    }
    let row_to_col = hungarian(n, &extended);
    let mut matches = Vec::new();
    let mut col_matched = vec![false; cols];
    let mut unmatched_rows = Vec::new();
    for (r, &c) in row_to_col.iter().enumerate().take(rows) {
        let cost = if c < cols {
            costs[r * cols + c]
        } else {
            f64::NAN
        };
        if cost.is_finite() && cost < cost_limit {
            matches.push((r, c));
            col_matched[c] = true;
        } else {
            unmatched_rows.push(r);
        }
    }
    Assignment {
        matches,
        unmatched_rows,
        unmatched_cols: (0..cols).filter(|&c| !col_matched[c]).collect(),
    }
}

/// Kuhn–Munkres with potentials on a square `n × n` matrix of finite costs:
/// the column assigned to each row, minimizing the total cost. O(n³).
fn hungarian(n: usize, costs: &[f64]) -> Vec<usize> {
    // 1-based, column 0 is the virtual start of each augmenting path.
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; n + 1];
    let mut row_of_col = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];
    for row in 1..=n {
        row_of_col[0] = row;
        let mut col0 = 0;
        let mut min_slack = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[col0] = true;
            let row0 = row_of_col[col0];
            let mut delta = f64::INFINITY;
            let mut col1 = 0;
            for col in 1..=n {
                if used[col] {
                    continue;
                }
                let slack = costs[(row0 - 1) * n + (col - 1)] - u[row0] - v[col];
                if slack < min_slack[col] {
                    min_slack[col] = slack;
                    way[col] = col0;
                }
                if min_slack[col] < delta {
                    delta = min_slack[col];
                    col1 = col;
                }
            }
            for col in 0..=n {
                if used[col] {
                    u[row_of_col[col]] += delta;
                    v[col] -= delta;
                } else {
                    min_slack[col] -= delta;
                }
            }
            col0 = col1;
            if row_of_col[col0] == 0 {
                break;
            }
        }
        loop {
            let col1 = way[col0];
            row_of_col[col0] = row_of_col[col1];
            col0 = col1;
            if col0 == 0 {
                break;
            }
        }
    }
    let mut col_of_row = vec![usize::MAX; n];
    for col in 1..=n {
        if row_of_col[col] != 0 {
            col_of_row[row_of_col[col] - 1] = col - 1;
        }
    }
    col_of_row
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    /// Cost of an assignment under the extended formulation.
    fn total(a: &Assignment, cols: usize, costs: &[f64], limit: f64) -> f64 {
        a.matches
            .iter()
            .map(|&(r, c)| costs[r * cols + c])
            .sum::<f64>()
            + 0.5 * limit * (a.unmatched_rows.len() + a.unmatched_cols.len()) as f64
    }

    /// Best total over every partial matching, by enumeration.
    fn brute_force(rows: usize, cols: usize, costs: &[f64], limit: f64) -> f64 {
        fn go(
            r: usize,
            rows: usize,
            cols: usize,
            costs: &[f64],
            limit: f64,
            used: &mut Vec<bool>,
        ) -> f64 {
            if r == rows {
                return 0.5 * limit * used.iter().filter(|u| !**u).count() as f64;
            }
            let mut best = 0.5 * limit + go(r + 1, rows, cols, costs, limit, used);
            for c in 0..cols {
                let cost = costs[r * cols + c];
                if !used[c] && cost.is_finite() && cost < limit {
                    used[c] = true;
                    best = best.min(cost + go(r + 1, rows, cols, costs, limit, used));
                    used[c] = false;
                }
            }
            best
        }
        go(0, rows, cols, costs, limit, &mut vec![false; cols])
    }

    #[test]
    fn matches_cheap_pairs_and_leaves_expensive_ones() {
        // Row 0 fits column 1, row 1 fits nothing below the limit.
        let costs = [0.9, 0.1, 0.95, 0.85, 0.9, 0.99];
        let a = assign_with_cost_limit(2, 3, &costs, 0.8);
        assert_eq!(a.matches, vec![(0, 1)]);
        assert_eq!(a.unmatched_rows, vec![1]);
        assert_eq!(a.unmatched_cols, vec![0, 2]);
    }

    #[test]
    fn prefers_the_globally_cheaper_pairing() {
        let costs = [0.1, 0.2, 0.3, 0.25];
        let a = assign_with_cost_limit(2, 2, &costs, 1.0);
        assert_eq!(a.matches, vec![(0, 0), (1, 1)]);
        // Greedy would take (0, 0) at 0.1 and leave row 1 with 0.7.
        let costs = [0.1, 0.2, 0.15, 0.7];
        let a = assign_with_cost_limit(2, 2, &costs, 1.0);
        assert_eq!(a.matches, vec![(0, 1), (1, 0)]);
    }

    #[test]
    fn forbidden_pairs_empty_inputs_and_bad_limits() {
        let costs = [f64::INFINITY, f64::NAN];
        let a = assign_with_cost_limit(1, 2, &costs, 0.8);
        assert!(a.matches.is_empty());
        assert_eq!(a.unmatched_cols, vec![0, 1]);
        let a = assign_with_cost_limit(0, 3, &[], 0.8);
        assert_eq!(a.unmatched_cols, vec![0, 1, 2]);
        let a = assign_with_cost_limit(1, 1, &[0.1], f64::NAN);
        assert!(a.matches.is_empty());
    }

    #[test]
    fn optimal_against_enumeration_on_random_matrices() {
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..400 {
            let rows = rng.gen_range(1..=5);
            let cols = rng.gen_range(1..=5);
            let limit = rng.gen_range(0.2..1.2);
            let costs: Vec<f64> = (0..rows * cols)
                .map(|_| {
                    if rng.gen_bool(0.1) {
                        f64::INFINITY
                    } else {
                        rng.gen_range(0.0..1.0)
                    }
                })
                .collect();
            let a = assign_with_cost_limit(rows, cols, &costs, limit);
            let got = total(&a, cols, &costs, limit);
            let best = brute_force(rows, cols, &costs, limit);
            assert!(
                (got - best).abs() < 1e-9,
                "{rows}×{cols} limit {limit}: {got} vs {best}"
            );
            assert_eq!(a.matches.len() + a.unmatched_rows.len(), rows);
            assert_eq!(a.matches.len() + a.unmatched_cols.len(), cols);
            assert!(a.matches.iter().all(|&(r, c)| costs[r * cols + c] < limit));
        }
    }
}
