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

/// Static 2-D kd-tree over a fixed set of points.
#[derive(Debug)]
pub(crate) struct KdTree {
    idx: Vec<u32>,
    coords: Vec<[f64; 2]>,
}

impl KdTree {
    /// Builds the tree in O(n log n) via median splits.
    pub fn build(coords: &[[f64; 2]]) -> Self {
        let mut idx: Vec<u32> = (0..coords.len() as u32).collect();
        build_rec(&mut idx, coords, 0);
        Self {
            idx,
            coords: coords.to_vec(),
        }
    }

    /// Up-to-`k` nearest points to `target`, optionally within `radius`.
    pub fn k_nearest(&self, target: [f64; 2], k: usize, radius: Option<f64>) -> Vec<usize> {
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
        target: [f64; 2],
        k: usize,
        r2: f64,
        heap: &mut BinaryHeap<Entry>,
    ) {
        if node.is_empty() {
            return;
        }
        let mid = node.len() / 2;
        let p = self.coords[node[mid] as usize];
        let dx = p[0] - target[0];
        let dy = p[1] - target[1];
        let d2 = dx * dx + dy * dy;
        if d2 <= r2 {
            push_candidate(heap, k, d2, node[mid]);
        }
        let diff = target[axis] - p[axis];
        let (near, far) = if diff <= 0.0 {
            (&node[..mid], &node[mid + 1..])
        } else {
            (&node[mid + 1..], &node[..mid])
        };
        self.search(near, 1 - axis, target, k, r2, heap);
        if diff * diff <= heap_worst(heap, k).min(r2) {
            self.search(far, 1 - axis, target, k, r2, heap);
        }
    }
}

fn build_rec(idx: &mut [u32], coords: &[[f64; 2]], axis: usize) {
    if idx.len() <= 1 {
        return;
    }
    let mid = idx.len() / 2;
    idx.select_nth_unstable_by(mid, |&a, &b| {
        coords[a as usize][axis].total_cmp(&coords[b as usize][axis])
    });
    let (l, r) = idx.split_at_mut(mid);
    build_rec(l, coords, 1 - axis);
    build_rec(&mut r[1..], coords, 1 - axis);
}

/// Incremental bucket grid for sequential simulation: O(1) insertion,
/// ring-expanding nearest-neighbor queries. Indices are insertion order.
#[derive(Debug)]
pub(crate) struct BucketGrid {
    x0: f64,
    y0: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    buckets: Vec<Vec<u32>>,
    pts: Vec<[f64; 2]>,
}

impl BucketGrid {
    /// Grid covering `[min, max]`, sized for ~1 point per bucket at
    /// `expected` total insertions.
    pub fn new(min: [f64; 2], max: [f64; 2], expected: usize) -> Self {
        let w = (max[0] - min[0]).max(f64::MIN_POSITIVE);
        let h = (max[1] - min[1]).max(f64::MIN_POSITIVE);
        let cell = (w * h / expected.max(1) as f64).sqrt().max(1e-12);
        let nx = ((w / cell).ceil() as usize).max(1);
        let ny = ((h / cell).ceil() as usize).max(1);
        Self {
            x0: min[0],
            y0: min[1],
            cell,
            nx,
            ny,
            buckets: vec![Vec::new(); nx * ny],
            pts: Vec::with_capacity(expected),
        }
    }

    fn cell_of(&self, p: [f64; 2]) -> (usize, usize) {
        let cx = (((p[0] - self.x0) / self.cell) as isize).clamp(0, self.nx as isize - 1);
        let cy = (((p[1] - self.y0) / self.cell) as isize).clamp(0, self.ny as isize - 1);
        (cx as usize, cy as usize)
    }

    /// Inserts a point; its index is the insertion order.
    pub fn insert(&mut self, p: [f64; 2]) {
        let (cx, cy) = self.cell_of(p);
        self.buckets[cy * self.nx + cx].push(self.pts.len() as u32);
        self.pts.push(p);
    }

    /// Up-to-`k` nearest inserted points, optionally within `radius`.
    pub fn k_nearest(&self, target: [f64; 2], k: usize, radius: Option<f64>) -> Vec<usize> {
        if k == 0 || self.pts.is_empty() {
            return Vec::new();
        }
        let r2 = radius.map_or(f64::INFINITY, |r| r * r);
        let (tcx, tcy) = self.cell_of(target);
        let max_ring = self.nx.max(self.ny);
        let mut heap = BinaryHeap::with_capacity(k + 1);

        for ring in 0..=max_ring {
            // Smallest possible distance from the target to any cell in
            // this ring (conservative bound).
            let ring_min = (ring as f64 - 1.0).max(0.0) * self.cell;
            if ring_min * ring_min > heap_worst(&heap, k).min(r2) {
                break;
            }
            self.for_ring_cells(tcx, tcy, ring, |bucket| {
                for &i in bucket {
                    let p = self.pts[i as usize];
                    let dx = p[0] - target[0];
                    let dy = p[1] - target[1];
                    let d2 = dx * dx + dy * dy;
                    if d2 <= r2 {
                        push_candidate(&mut heap, k, d2, i);
                    }
                }
            });
        }
        into_sorted(heap)
    }

    /// Visits the buckets at Chebyshev distance `ring` from `(tcx, tcy)`.
    fn for_ring_cells<F: FnMut(&[u32])>(&self, tcx: usize, tcy: usize, ring: usize, mut f: F) {
        let (tcx, tcy, ring) = (tcx as isize, tcy as isize, ring as isize);
        let visit = |cx: isize, cy: isize, f: &mut F| {
            if cx >= 0 && cy >= 0 && (cx as usize) < self.nx && (cy as usize) < self.ny {
                f(&self.buckets[cy as usize * self.nx + cx as usize]);
            }
        };
        if ring == 0 {
            visit(tcx, tcy, &mut f);
            return;
        }
        for cx in (tcx - ring)..=(tcx + ring) {
            visit(cx, tcy - ring, &mut f);
            visit(cx, tcy + ring, &mut f);
        }
        for cy in (tcy - ring + 1)..(tcy + ring) {
            visit(tcx - ring, cy, &mut f);
            visit(tcx + ring, cy, &mut f);
        }
    }
}

/// Brute-force reference implementation for the search tests.
#[cfg(test)]
pub(crate) fn k_nearest_brute(
    coords: &[[f64; 2]],
    target: [f64; 2],
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
            let dx = c[0] - target[0];
            let dy = c[1] - target[1];
            let d2 = dx * dx + dy * dy;
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
}
