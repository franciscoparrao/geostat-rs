//! Spatial neighbor search: static kd-tree (kriging) and incremental
//! bucket grid (sequential simulation, where the conditioning set grows).
//!
//! Both return up-to-`k` nearest indices sorted by increasing distance,
//! optionally restricted to a search radius. Distances are Euclidean
//! (anisotropy applies to covariances, not to the search, matching
//! gstat/GSLIB behavior).

use std::collections::BinaryHeap;

/// Max-heap entry: `f64::to_bits` preserves ordering for non-negative
/// floats, so `(d2.to_bits(), idx)` sorts by squared distance.
type Entry = (u64, u32);

fn push_candidate(heap: &mut BinaryHeap<Entry>, k: usize, d2: f64, idx: u32) {
    if heap.len() < k {
        heap.push((d2.to_bits(), idx));
    } else if d2.to_bits() < heap.peek().unwrap().0 {
        heap.push((d2.to_bits(), idx));
        heap.pop();
    }
}

fn heap_worst(heap: &BinaryHeap<Entry>, k: usize) -> f64 {
    if heap.len() == k {
        f64::from_bits(heap.peek().unwrap().0)
    } else {
        f64::INFINITY
    }
}

fn into_sorted(heap: BinaryHeap<Entry>) -> Vec<usize> {
    let mut v = heap.into_vec();
    v.sort_unstable();
    v.into_iter().map(|(_, i)| i as usize).collect()
}

/// Static D-dimensional kd-tree over a fixed set of points.
#[derive(Debug)]
pub(crate) struct KdTree<const D: usize = 2> {
    idx: Vec<u32>,
    coords: Vec<[f64; D]>,
}

impl<const D: usize> KdTree<D> {
    /// Builds the tree in O(n log n) via median splits.
    pub fn build(coords: &[[f64; D]]) -> Self {
        let mut idx: Vec<u32> = (0..coords.len() as u32).collect();
        build_rec(&mut idx, coords, 0);
        Self {
            idx,
            coords: coords.to_vec(),
        }
    }

    /// Up-to-`k` nearest points to `target`, optionally within `radius`.
    pub fn k_nearest(&self, target: [f64; D], k: usize, radius: Option<f64>) -> Vec<usize> {
        if k == 0 || self.idx.is_empty() {
            return Vec::new();
        }
        let r2 = radius.map_or(f64::INFINITY, |r| r * r);
        let mut heap = BinaryHeap::with_capacity(k + 1);
        self.search(&self.idx, 0, target, k, r2, &mut heap);
        into_sorted(heap)
    }

    fn search(
        &self,
        node: &[u32],
        axis: usize,
        target: [f64; D],
        k: usize,
        r2: f64,
        heap: &mut BinaryHeap<Entry>,
    ) {
        if node.is_empty() {
            return;
        }
        let mid = node.len() / 2;
        let p = self.coords[node[mid] as usize];
        let mut d2 = 0.0;
        for d in 0..D {
            let dd = p[d] - target[d];
            d2 += dd * dd;
        }
        if d2 <= r2 {
            push_candidate(heap, k, d2, node[mid]);
        }
        let diff = target[axis] - p[axis];
        let (near, far) = if diff <= 0.0 {
            (&node[..mid], &node[mid + 1..])
        } else {
            (&node[mid + 1..], &node[..mid])
        };
        self.search(near, (axis + 1) % D, target, k, r2, heap);
        if diff * diff <= heap_worst(heap, k).min(r2) {
            self.search(far, (axis + 1) % D, target, k, r2, heap);
        }
    }
}

fn build_rec<const D: usize>(idx: &mut [u32], coords: &[[f64; D]], axis: usize) {
    if idx.len() <= 1 {
        return;
    }
    let mid = idx.len() / 2;
    idx.select_nth_unstable_by(mid, |&a, &b| {
        coords[a as usize][axis].total_cmp(&coords[b as usize][axis])
    });
    let (l, r) = idx.split_at_mut(mid);
    build_rec(l, coords, (axis + 1) % D);
    build_rec(&mut r[1..], coords, (axis + 1) % D);
}

/// Incremental D-dimensional bucket grid for sequential simulation: O(1)
/// insertion, shell-expanding nearest-neighbor queries. Indices are
/// insertion order.
#[derive(Debug)]
pub(crate) struct BucketGrid<const D: usize = 2> {
    origin: [f64; D],
    cell: f64,
    n: [usize; D],
    strides: [usize; D],
    buckets: Vec<Vec<u32>>,
    pts: Vec<[f64; D]>,
}

impl<const D: usize> BucketGrid<D> {
    /// Grid covering `[min, max]`, sized for ~1 point per bucket at
    /// `expected` total insertions.
    ///
    /// Dimensions with ~zero extent (all points share that coordinate —
    /// e.g. a transect, or 3-D data from a single bench) are excluded from
    /// the cell-size computation and get exactly one bucket along that
    /// axis. Including them would force `cell` to be tiny (derived from an
    /// ~zero volume), blowing up the bucket count in the *other*
    /// dimensions to an unallocatable size.
    pub fn new(min: [f64; D], max: [f64; D], expected: usize) -> Self {
        let extents: [f64; D] = std::array::from_fn(|d| max[d] - min[d]);
        let max_extent = extents.iter().copied().fold(0.0_f64, f64::max);
        let tol = max_extent * 1e-9;
        let active: [bool; D] = std::array::from_fn(|d| extents[d] > tol);
        let n_active = active.iter().filter(|&&a| a).count().max(1);

        let mut volume = 1.0_f64;
        for d in 0..D {
            if active[d] {
                volume *= extents[d];
            }
        }
        let mut cell = volume.powf(1.0 / n_active as f64).max(1e-12)
            * (1.0 / (expected.max(1) as f64).powf(1.0 / n_active as f64));

        // Defensive cap: however `cell` was derived, never materialize more
        // than a small multiple of the expected point count worth of
        // buckets — double `cell` until the (possibly overflow-checked)
        // total fits.
        let cap = expected.max(1).saturating_mul(4).max(64);
        let mut n = [1usize; D];
        loop {
            let mut total = 1usize;
            let mut overflowed = false;
            for d in 0..D {
                n[d] = if active[d] {
                    ((extents[d] / cell).ceil() as usize).max(1)
                } else {
                    1
                };
                total = match total.checked_mul(n[d]) {
                    Some(t) => t,
                    None => {
                        overflowed = true;
                        break;
                    }
                };
            }
            if !overflowed && total <= cap {
                break;
            }
            cell *= 2.0;
        }

        let mut strides = [1usize; D];
        let mut total = 1usize;
        for d in 0..D {
            strides[d] = total;
            total *= n[d];
        }
        Self {
            origin: min,
            cell,
            n,
            strides,
            buckets: vec![Vec::new(); total],
            pts: Vec::with_capacity(expected),
        }
    }

    fn cell_of(&self, p: [f64; D]) -> [usize; D] {
        let mut c = [0usize; D];
        for d in 0..D {
            c[d] = (((p[d] - self.origin[d]) / self.cell) as isize).clamp(0, self.n[d] as isize - 1)
                as usize;
        }
        c
    }

    fn bucket_index(&self, c: [usize; D]) -> usize {
        c.iter().zip(&self.strides).map(|(&ci, &s)| ci * s).sum()
    }

    /// Inserts a point; its index is the insertion order.
    pub fn insert(&mut self, p: [f64; D]) {
        let c = self.cell_of(p);
        let idx = self.bucket_index(c);
        self.buckets[idx].push(self.pts.len() as u32);
        self.pts.push(p);
    }

    /// Up-to-`k` nearest inserted points, optionally within `radius`.
    pub fn k_nearest(&self, target: [f64; D], k: usize, radius: Option<f64>) -> Vec<usize> {
        if k == 0 || self.pts.is_empty() {
            return Vec::new();
        }
        let r2 = radius.map_or(f64::INFINITY, |r| r * r);
        let tc = self.cell_of(target);
        let max_ring = self.n.iter().copied().max().unwrap_or(1);
        let mut heap = BinaryHeap::with_capacity(k + 1);

        for ring in 0..=max_ring {
            // Smallest possible distance from the target to any cell in
            // this shell (conservative bound).
            let ring_min = (ring as f64 - 1.0).max(0.0) * self.cell;
            if ring_min * ring_min > heap_worst(&heap, k).min(r2) {
                break;
            }
            self.for_shell_cells(tc, ring, |bucket| {
                for &i in bucket {
                    let p = self.pts[i as usize];
                    let mut d2 = 0.0;
                    for d in 0..D {
                        let dd = p[d] - target[d];
                        d2 += dd * dd;
                    }
                    if d2 <= r2 {
                        push_candidate(&mut heap, k, d2, i);
                    }
                }
            });
        }
        into_sorted(heap)
    }

    /// Visits the buckets at Chebyshev distance `ring` from `tc`, walking
    /// only the shell surface — `O(ring^(D-1))` cells instead of the full
    /// `(2 ring + 1)^D` bounding box.
    fn for_shell_cells<F: FnMut(&[u32])>(&self, tc: [usize; D], ring: usize, mut f: F) {
        let r = ring as isize;
        let mut t = [0isize; D];
        for d in 0..D {
            t[d] = tc[d] as isize;
        }
        let visit = |cur: &[isize; D], f: &mut F| {
            for (d, &v) in cur.iter().enumerate() {
                if v < 0 || v >= self.n[d] as isize {
                    return;
                }
            }
            let mut c = [0usize; D];
            for (cd, &v) in c.iter_mut().zip(cur) {
                *cd = v as usize;
            }
            f(&self.buckets[self.bucket_index(c)]);
        };
        if ring == 0 {
            visit(&t, &mut f);
            return;
        }
        // Each surface cell has max |offset| = r: enumerate the two faces of
        // every dimension `fd` exactly once by restricting dimensions before
        // `fd` to |offset| < r (they are covered by their own faces).
        for fd in 0..D {
            for side in [-r, r] {
                let mut lo = [0isize; D];
                let mut hi = [0isize; D];
                for d in 0..D {
                    let (l, h) = match d.cmp(&fd) {
                        std::cmp::Ordering::Less => (t[d] - r + 1, t[d] + r - 1),
                        std::cmp::Ordering::Equal => (t[d] + side, t[d] + side),
                        std::cmp::Ordering::Greater => (t[d] - r, t[d] + r),
                    };
                    lo[d] = l;
                    hi[d] = h;
                }
                if lo.iter().zip(&hi).any(|(l, h)| l > h) {
                    continue;
                }
                // Odometer over the face box.
                let mut cur = lo;
                'face: loop {
                    visit(&cur, &mut f);
                    for d in 0..D {
                        cur[d] += 1;
                        if cur[d] <= hi[d] {
                            continue 'face;
                        }
                        cur[d] = lo[d];
                    }
                    break;
                }
            }
        }
    }
}

/// Brute-force reference implementation for the search tests.
#[cfg(test)]
pub(crate) fn k_nearest_brute<const D: usize>(
    coords: &[[f64; D]],
    target: [f64; D],
    k: usize,
    radius: Option<f64>,
) -> Vec<usize> {
    if k == 0 {
        return Vec::new();
    }
    let r2 = radius.map(|r| r * r);
    let mut cand: Vec<(f64, usize)> = coords
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let mut d2 = 0.0;
            for d in 0..D {
                let dd = c[d] - target[d];
                d2 += dd * dd;
            }
            match r2 {
                Some(r2) if d2 > r2 => None,
                _ => Some((d2, i)),
            }
        })
        .collect();
    if cand.len() > k {
        cand.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
        cand.truncate(k);
    }
    cand.sort_by(|a, b| a.0.total_cmp(&b.0));
    cand.into_iter().map(|(_, i)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn random_points(n: usize, seed: u64) -> Vec<[f64; 2]> {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| [rng.uniform() * 100.0, rng.uniform() * 100.0])
            .collect()
    }

    #[test]
    fn kdtree_matches_brute_force() {
        let pts = random_points(500, 5);
        let tree = KdTree::build(&pts);
        let mut rng = Rng::new(6);
        for _ in 0..200 {
            let t = [rng.uniform() * 120.0 - 10.0, rng.uniform() * 120.0 - 10.0];
            for (k, r) in [(1, None), (8, None), (16, Some(15.0)), (1000, Some(5.0))] {
                let a = tree.k_nearest(t, k, r);
                let b = k_nearest_brute(&pts, t, k, r);
                assert_eq!(a, b, "k={k}, r={r:?}, target={t:?}");
            }
        }
    }

    #[test]
    fn bucket_grid_matches_brute_force_incrementally() {
        let pts = random_points(400, 9);
        let mut grid = BucketGrid::new([0.0, 0.0], [100.0, 100.0], 400);
        let mut inserted: Vec<[f64; 2]> = Vec::new();
        let mut rng = Rng::new(10);
        for (i, &p) in pts.iter().enumerate() {
            grid.insert(p);
            inserted.push(p);
            if i % 37 == 0 {
                let t = [rng.uniform() * 100.0, rng.uniform() * 100.0];
                for (k, r) in [(4, None), (16, Some(20.0))] {
                    let a = grid.k_nearest(t, k, r);
                    let b = k_nearest_brute(&inserted, t, k, r);
                    assert_eq!(a, b, "after {} inserts, k={k}, r={r:?}", i + 1);
                }
            }
        }
    }

    #[test]
    fn empty_and_degenerate_inputs() {
        let tree = KdTree::build(&[]);
        assert!(tree.k_nearest([0.0, 0.0], 5, None).is_empty());
        let grid = BucketGrid::new([0.0, 0.0], [1.0, 1.0], 0);
        assert!(grid.k_nearest([0.5, 0.5], 5, None).is_empty());
        let tree = KdTree::build(&[[1.0, 1.0]]);
        assert_eq!(tree.k_nearest([0.0, 0.0], 5, None), vec![0]);
        assert!(tree.k_nearest([0.0, 0.0], 5, Some(0.5)).is_empty());
    }

    /// Regression: a collinear point set (zero extent on one axis) used to
    /// force `cell` to ~1e-160 and blow up the bucket count to an
    /// unallocatable size (OOM abort). See AUDIT-2026-07-v3.md §1.1.
    #[test]
    fn collinear_points_do_not_blow_up_bucket_count() {
        let grid = BucketGrid::new([0.0, 5.0], [500.0, 5.0], 50);
        assert_eq!(grid.n, [50, 1]);

        let mut inserted: Vec<[f64; 2]> = Vec::new();
        let mut rng = Rng::new(1);
        let mut grid = grid;
        for i in 0..50u32 {
            let p = [i as f64 * 10.0, 5.0];
            grid.insert(p);
            inserted.push(p);
        }
        let t = [rng.uniform() * 500.0, 5.0];
        let a = grid.k_nearest(t, 5, None);
        let b = k_nearest_brute(&inserted, t, 5, None);
        assert_eq!(a, b);
    }

    /// Same failure mode in 3-D: a single-bench dataset with constant z.
    #[test]
    fn collinear_points_3d_do_not_blow_up_bucket_count() {
        let grid = BucketGrid::<3>::new([0.0, 0.0, 5.0], [500.0, 500.0, 5.0], 100);
        assert_eq!(grid.n[2], 1);
        assert!(grid.n[0] * grid.n[1] * grid.n[2] <= 4 * 100);
    }

    /// An all-coincident point set (every axis degenerate) must still
    /// produce a valid single-bucket grid, not a division that blows up.
    #[test]
    fn all_coincident_points_single_bucket() {
        let grid = BucketGrid::new([3.0, 3.0], [3.0, 3.0], 10);
        assert_eq!(grid.n, [1, 1]);
    }
}
