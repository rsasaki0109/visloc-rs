//! Basalt landmark and keyframe contracts, independent of generic PnP.
use crate::camera::DoubleSphereCamera;
use crate::fma::FusedMulAdd;
use nalgebra::{Matrix3, Matrix3x2, Matrix4, Point2, Point3, UnitQuaternion, Vector3, Vector4};
use std::collections::BTreeMap;
use visloc_core::geometry::SE3;

/// Eigen's fixed-size `Vector4f::head<3>().norm()` reduction is not the same
/// operation as a standalone `Vector3f::norm()`.  In the pinned AVX/FMA
/// build, the `VectorBlock` evaluator emits the squared terms in this packet
/// lane order: `y*y`, fused with `z*z`, then fused with `x*x`.  Keep that
/// reduction explicit at the f32 boundary; the more obvious `x*x + y*y +
/// z*z` (and the standalone-vector reduction) changes the normalization by a
/// ulp for some landmarks.
#[inline]
fn eigen_norm3_f32(x: f32, y: f32, z: f32) -> f32 {
    x.fused_mul_add(x, z.fused_mul_add(z, y * y)).sqrt()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereographicDirection {
    pub xy: Point2<f64>,
}

impl StereographicDirection {
    pub fn from_bearing(b: Vector3<f64>) -> Option<Self> {
        let b = b.try_normalize(1e-12)?;
        let denominator = 1.0 + b.z;
        if denominator.abs() < 1e-12 {
            return None;
        }
        Some(Self {
            xy: Point2::new(b.x / denominator, b.y / denominator),
        })
    }
    pub fn bearing(self) -> Vector3<f64> {
        let r2 = self.xy.coords.norm_squared();
        Vector3::new(2.0 * self.xy.x, 2.0 * self.xy.y, 1.0 - r2) / (1.0 + r2)
    }

    /// Jacobian of Basalt's stereographic unprojection with respect to
    /// `[u, v]`. This is the upper 3x2 block of upstream `Jup`.
    pub fn bearing_jacobian(self) -> Matrix3x2<f64> {
        let u = self.xy.x;
        let v = self.xy.y;
        let norm_inv = 2.0 / (1.0 + u * u + v * v);
        let norm_inv2 = norm_inv * norm_inv;
        Matrix3x2::new(
            norm_inv - u * u * norm_inv2,
            -u * v * norm_inv2,
            -u * v * norm_inv2,
            norm_inv - v * v * norm_inv2,
            -u * norm_inv2,
            -v * norm_inv2,
        )
    }

    /// Upstream `Keypoint<float>::direction` arithmetic. The stored public
    /// direction remains f64, but all conversion/normalization operations in
    /// this helper are owned by f32 before crossing back to the API type.
    pub fn from_bearing_f32(b: Vector3<f32>) -> Option<Self> {
        let norm = eigen_norm3_f32(b.x, b.y, b.z);
        if !norm.is_finite() || norm <= f32::EPSILON {
            return None;
        }
        let b = b / norm;
        let denominator = 1.0_f32 + b.z;
        if denominator.abs() < f32::EPSILON {
            return None;
        }
        Some(Self {
            xy: Point2::new((b.x / denominator) as f64, (b.y / denominator) as f64),
        })
    }

    /// Construct the stored direction with Basalt's
    /// `StereographicParam<float>::project` expression order.  Triangulation
    /// returns a homogeneous `[unit_direction, inverse_distance]` vector;
    /// upstream projects that direction directly.  Normalizing a reconstructed
    /// `direction / rho` point in f64 first is algebraically equivalent but
    /// crosses the float boundary at the wrong place and changes every new
    /// landmark's last bits.
    pub fn from_triangulated_f32(direction: Vector3<f32>) -> Option<Self> {
        if !direction.iter().all(|value| value.is_finite()) {
            return None;
        }
        let sqrt = eigen_norm3_f32(direction.x, direction.y, direction.z);
        let norm = direction.z + sqrt;
        if !sqrt.is_finite() || !norm.is_finite() || norm.abs() <= f32::EPSILON {
            return None;
        }
        let norm_inv = 1.0_f32 / norm;
        Some(Self {
            xy: Point2::new(
                (direction.x * norm_inv) as f64,
                (direction.y * norm_inv) as f64,
            ),
        })
    }

    pub fn bearing_f32(self) -> Vector3<f32> {
        let u = self.xy.x as f32;
        let v = self.xy.y as f32;
        let r2 = u * u + v * v;
        // StereographicParam<float>::unproject first materializes the common
        // scale `2 / (1 + r2)`, then forms z as `scale - 1`.  Dividing the
        // three numerator lanes by the denominator is algebraically equal
        // but changes the binary32 z lane for some points.
        let norm_inv = 2.0_f32 / (1.0_f32 + r2);
        Vector3::new(u * norm_inv, v * norm_inv, norm_inv - 1.0_f32)
    }

    pub fn bearing_jacobian_f32(self) -> nalgebra::SMatrix<f32, 3, 2> {
        let u = self.xy.x as f32;
        let v = self.xy.y as f32;
        // Keep the scalar temporaries in Basalt's
        // `StereographicParam<float>::unproject` order.  In particular,
        // Eigen evaluates `x2`, `y2`, and `r2` as separate f32 values before
        // forming `norm_inv`; spelling the denominator as `1 + u*u + v*v`
        // gives a different association and can move every active Jup lane.
        let x2 = u * u;
        let y2 = v * v;
        let r2 = x2 + y2;
        let norm_inv = 2.0_f32 / (1.0_f32 + r2);
        let norm_inv2 = norm_inv * norm_inv;
        let xy = u * v;
        nalgebra::SMatrix::from_row_slice(&[
            (-x2).fused_mul_add(norm_inv2, norm_inv),
            -xy * norm_inv2,
            -xy * norm_inv2,
            (-y2).fused_mul_add(norm_inv2, norm_inv),
            -u * norm_inv2,
            -v * norm_inv2,
        ])
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseDistanceLandmark {
    pub anchor_pose: u64,
    /// Camera stream of the host keyframe (`TimeCamId::cam_id` upstream).
    pub anchor_camera_id: u16,
    pub direction: StereographicDirection,
    pub inverse_distance: f64,
}

impl InverseDistanceLandmark {
    pub fn position_in_anchor(&self) -> Option<Vector3<f64>> {
        if !self.inverse_distance.is_finite() || self.inverse_distance <= 0.0 {
            return None;
        }
        Some(self.direction.bearing() / self.inverse_distance)
    }

    /// Basalt's homogeneous point `[bearing_x, bearing_y, bearing_z, rho]`.
    /// Keeping the point homogeneous avoids dividing by a small inverse
    /// distance during projection and gives the literal landmark Jacobian.
    pub fn homogeneous(self) -> [f64; 4] {
        let bearing = self.direction.bearing();
        [bearing.x, bearing.y, bearing.z, self.inverse_distance]
    }

    /// Apply the back-substituted `[du, dv, d_rho]` block. Upstream clamps
    /// inverse distance to zero after applying the increment.
    pub fn apply_increment(&mut self, increment: Vector3<f64>) -> bool {
        if !increment.iter().all(|value| value.is_finite()) {
            return false;
        }
        self.direction.xy.coords += increment.fixed_rows::<2>(0);
        self.inverse_distance = (self.inverse_distance + increment.z).max(0.0);
        self.direction
            .xy
            .coords
            .iter()
            .all(|value| value.is_finite())
            && self.inverse_distance.is_finite()
    }

    /// Apply the upstream float landmark writeback boundary.  The estimator's
    /// public window model stores f64 values, but Basalt's
    /// `LandmarkBlockAbsDynamic<float>` updates `direction` and `inv_dist` as
    /// binary32 values.  Round both operands to f32 before each add, then
    /// widen back to the model type; doing the add in f64 changes the next
    /// LM linearization by one or more ulps.
    pub fn apply_increment_f32(&mut self, increment: Vector3<f64>) -> bool {
        if !increment.iter().all(|value| value.is_finite()) {
            return false;
        }
        let mut x = self.direction.xy.x as f32;
        let mut y = self.direction.xy.y as f32;
        let mut rho = self.inverse_distance as f32;
        x += increment.x as f32;
        y += increment.y as f32;
        rho = (rho + increment.z as f32).max(0.0_f32);
        if !x.is_finite() || !y.is_finite() || !rho.is_finite() {
            return false;
        }
        self.direction.xy.x = x as f64;
        self.direction.xy.y = y as f64;
        self.inverse_distance = rho as f64;
        true
    }

    /// Re-express a finite world point in a new host camera without changing
    /// its geometry. `anchor_camera_pose` is `T_w_c`.
    pub fn reanchor(
        &mut self,
        anchor_pose: u64,
        anchor_camera_id: u16,
        anchor_camera_pose: &SE3,
        point_world: Point3<f64>,
    ) -> bool {
        let point_anchor = anchor_camera_pose.inverse().transform_point(&point_world);
        let distance = point_anchor.coords.norm();
        if !distance.is_finite() || distance <= 1e-12 {
            return false;
        }
        let Some(direction) = Self::direction_from_anchor_point(point_anchor) else {
            return false;
        };
        self.anchor_pose = anchor_pose;
        self.anchor_camera_id = anchor_camera_id;
        self.direction = direction;
        self.inverse_distance = 1.0 / distance;
        true
    }

    fn direction_from_anchor_point(point: Point3<f64>) -> Option<StereographicDirection> {
        StereographicDirection::from_bearing(point.coords)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BearingObservation {
    pub landmark_id: u64,
    pub frame_id: u64,
    pub bearing: Vector3<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LandmarkStatus {
    Active,
    Lost,
    Removed,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LandmarkRecord {
    pub landmark: InverseDistanceLandmark,
    pub connected_observations: u32,
    pub missed_frames: u32,
    pub status: LandmarkStatus,
}

/// The pinned Basalt build uses libstdc++'s default `std::unordered_map` and
/// `std::unordered_set` for landmark IDs and candidate observations.  Rust's
/// ordered maps are useful for the public data model, but iterating one here
/// changes the order in which floating-point normal-equation blocks are
/// accumulated.  This small order-only table mirrors the relevant
/// libstdc++ `_Hashtable` state: a prime bucket array and the global singly
/// linked node chain.  Values stay in the ordinary Rust map; this type only
/// owns the source-compatible iteration order.
///
/// The implementation follows GCC 11's `_Prime_rehash_policy` and the unique
/// key branches of `_M_insert_bucket_begin` / `_M_rehash_aux`.  It deliberately
/// does not contain any fixture IDs or track-specific rules.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LibstdcxxUnorderedOrder {
    nodes: Vec<UnorderedOrderNode>,
    buckets: Vec<Option<usize>>,
    head: Option<usize>,
    len: usize,
    next_resize: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UnorderedOrderNode {
    key: u64,
    hash: u64,
    next: Option<usize>,
}

const BEFORE_BEGIN: usize = usize::MAX;

impl Default for LibstdcxxUnorderedOrder {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            // libstdc++ starts with one logical bucket and allocates the
            // first real bucket array on the first insertion.
            buckets: vec![None],
            head: None,
            len: 0,
            next_resize: 0,
        }
    }
}

impl LibstdcxxUnorderedOrder {
    #[inline]
    const fn bucket_for(hash: u64, bucket_count: usize) -> usize {
        // std::hash<int64_t> is the identity hash on the pinned libstdc++
        // target, whose size_t is 64-bit: reduce the full 64-bit hash, so a
        // 32-bit target (wasm32) picks the same bucket (the VIO IDs used
        // here are non-negative).
        (hash % bucket_count as u64) as usize
    }

    fn find_node(&self, key: u64) -> Option<usize> {
        let mut current = self.head;
        while let Some(index) = current {
            let node = self.nodes[index];
            if node.key == key {
                return Some(index);
            }
            current = node.next;
        }
        None
    }

    #[inline]
    fn set_after(&mut self, predecessor: Option<usize>, next: Option<usize>) {
        match predecessor {
            Some(BEFORE_BEGIN) => self.head = next,
            Some(index) => self.nodes[index].next = next,
            None => self.head = next,
        }
    }

    #[inline]
    fn after(&self, predecessor: usize) -> Option<usize> {
        if predecessor == BEFORE_BEGIN {
            self.head
        } else {
            self.nodes[predecessor].next
        }
    }

    /// Return the same bucket count selected by GCC 11's sparse libstdc++
    /// prime table.  The table is copied from the pinned runtime's
    /// `_Prime_rehash_policy::__prime_list`; the fallback is only for a
    /// synthetic size beyond that ABI table.
    fn next_bucket(minimum: usize) -> usize {
        // The values selected by GCC 11's sparse prime table for the
        // doubling requests generated by _Prime_rehash_policy.  Keeping the
        // table sparse is important: e.g. 514 selects 541, not the first
        // ordinary prime (521) after the request.
        // Exact GCC 11 libstdc++ __prime_list used by
        // _Prime_rehash_policy::_M_next_bkt.  This table is intentionally
        // sparse (for example, a request for 514 selects 541, not 521).
        // `u64`: the 64-bit ABI's table runs past a 32-bit `usize` (wasm32),
        // where those entries can never be requested.
        const GCC11_BUCKETS: &[u64] = &[
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
            73,
            79,
            83,
            89,
            97,
            103,
            109,
            113,
            127,
            137,
            139,
            149,
            157,
            167,
            179,
            193,
            199,
            211,
            227,
            241,
            257,
            277,
            293,
            313,
            337,
            359,
            383,
            409,
            439,
            467,
            503,
            541,
            577,
            619,
            661,
            709,
            761,
            823,
            887,
            953,
            1031,
            1109,
            1193,
            1289,
            1381,
            1493,
            1613,
            1741,
            1879,
            2029,
            2179,
            2357,
            2549,
            2753,
            2971,
            3209,
            3469,
            3739,
            4027,
            4349,
            4703,
            5087,
            5503,
            5953,
            6427,
            6949,
            7517,
            8123,
            8783,
            9497,
            10273,
            11113,
            12011,
            12983,
            14033,
            15173,
            16411,
            17749,
            19183,
            20753,
            22447,
            24281,
            26267,
            28411,
            30727,
            33223,
            35933,
            38873,
            42043,
            45481,
            49201,
            53201,
            57557,
            62233,
            67307,
            72817,
            78779,
            85229,
            92203,
            99733,
            107897,
            116731,
            126271,
            136607,
            147793,
            159871,
            172933,
            187091,
            202409,
            218971,
            236897,
            256279,
            277261,
            299951,
            324503,
            351061,
            379787,
            410857,
            444487,
            480881,
            520241,
            562841,
            608903,
            658753,
            712697,
            771049,
            834181,
            902483,
            976369,
            1056323,
            1142821,
            1236397,
            1337629,
            1447153,
            1565659,
            1693859,
            1832561,
            1982627,
            2144977,
            2320627,
            2510653,
            2716249,
            2938679,
            3179303,
            3439651,
            3721303,
            4026031,
            4355707,
            4712381,
            5098259,
            5515729,
            5967347,
            6456007,
            6984629,
            7556579,
            8175383,
            8844859,
            9569143,
            10352717,
            11200489,
            12117689,
            13109983,
            14183539,
            15345007,
            16601593,
            17961079,
            19431899,
            21023161,
            22744717,
            24607243,
            26622317,
            28802401,
            31160981,
            33712729,
            36473443,
            39460231,
            42691603,
            46187573,
            49969847,
            54061849,
            58488943,
            63278561,
            68460391,
            74066549,
            80131819,
            86693767,
            93793069,
            101473717,
            109783337,
            118773397,
            128499677,
            139022417,
            150406843,
            162723577,
            176048909,
            190465427,
            206062531,
            222936881,
            241193053,
            260944219,
            282312799,
            305431229,
            330442829,
            357502601,
            386778277,
            418451333,
            452718089,
            489790921,
            529899637,
            573292817,
            620239453,
            671030513,
            725980837,
            785430967,
            849749479,
            919334987,
            994618837,
            1076067617,
            1164186217,
            1259520799,
            1362662261,
            1474249943,
            1594975441,
            1725587117,
            1866894511,
            2019773507,
            2185171673,
            2364114217,
            2557710269,
            2767159799,
            2993761039,
            3238918481,
            3504151727,
            3791104843,
            4101556399,
            4294967291,
            6442450933,
            8589934583,
            12884901857,
            17179869143,
            25769803693,
            34359738337,
            51539607367,
            68719476731,
            103079215087,
            137438953447,
            206158430123,
            274877906899,
            412316860387,
            549755813881,
            824633720731,
            1099511627689,
            1649267441579,
            2199023255531,
            3298534883309,
            4398046511093,
            6597069766607,
            8796093022151,
            13194139533241,
            17592186044399,
            26388279066581,
            35184372088777,
            52776558133177,
            70368744177643,
            105553116266399,
            140737488355213,
            211106232532861,
            281474976710597,
            562949953421231,
            1125899906842597,
            2251799813685119,
            4503599627370449,
            9007199254740881,
            18014398509481951,
            36028797018963913,
            72057594037927931,
            144115188075855859,
            288230376151711717,
            576460752303423433,
            1152921504606846883,
            2305843009213693951,
            4611686018427387847,
            9223372036854775783,
            18446744073709551557,
            18446744073709551557,
        ];
        if let Some(bucket) = GCC11_BUCKETS
            .iter()
            .find(|&&bucket| bucket >= minimum as u64)
            .and_then(|&bucket| usize::try_from(bucket).ok())
        {
            return bucket;
        }

        // The table above covers all practical VIO table sizes.  Keep a
        // deterministic source-equivalent fallback for larger synthetic
        // workloads rather than silently selecting a non-prime bucket.
        let mut candidate = minimum.max(2);
        loop {
            let mut divisor = 2usize;
            while divisor.saturating_mul(divisor) <= candidate {
                if candidate.is_multiple_of(divisor) {
                    break;
                }
                divisor += 1;
            }
            if divisor.saturating_mul(divisor) > candidate {
                return candidate;
            }
            candidate = candidate.saturating_add(1);
        }
    }

    fn rehash(&mut self, bucket_count: usize) {
        let mut buckets = vec![None; bucket_count];
        let mut old_node = self.head;
        let mut new_head = None;
        let mut begin_bucket = 0;

        // This is the unique-key `_M_rehash_aux` loop.  Rehashing is not a
        // stable insertion-order operation: each first node of a bucket is
        // pushed at the global head, exactly as in libstdc++.
        while let Some(index) = old_node {
            let next_old = self.nodes[index].next;
            let bucket = Self::bucket_for(self.nodes[index].hash, bucket_count);
            if buckets[bucket].is_none() {
                self.nodes[index].next = new_head;
                new_head = Some(index);
                buckets[bucket] = Some(BEFORE_BEGIN);
                if self.nodes[index].next.is_some() {
                    buckets[begin_bucket] = Some(index);
                }
                begin_bucket = bucket;
            } else {
                let predecessor = buckets[bucket].expect("occupied bucket has predecessor");
                let next = if predecessor == BEFORE_BEGIN {
                    new_head
                } else {
                    self.nodes[predecessor].next
                };
                self.nodes[index].next = next;
                if predecessor == BEFORE_BEGIN {
                    new_head = Some(index);
                } else {
                    self.nodes[predecessor].next = Some(index);
                }
            }
            old_node = next_old;
        }

        self.head = new_head;
        self.buckets = buckets;
        self.next_resize = bucket_count;
    }

    fn maybe_rehash_for_insert(&mut self) {
        if self.len.saturating_add(1) <= self.next_resize {
            return;
        }

        // `_M_next_resize == 0` denotes an unallocated table.  GCC 11 asks
        // for an initial size of 11, which `_M_next_bkt` rounds to 13.
        let minimum = (self.len.saturating_add(1)).max(if self.next_resize == 0 { 11 } else { 0 });
        if minimum < self.buckets.len() {
            self.next_resize = self.buckets.len();
            return;
        }

        let requested = (minimum.saturating_add(1)).max(self.buckets.len().saturating_mul(2));
        self.rehash(Self::next_bucket(requested));
    }

    fn insert(&mut self, key: u64) -> bool {
        if self.find_node(key).is_some() {
            return false;
        }
        self.insert_unique(key);
        true
    }

    /// Insert a key known to be absent.  Descriptor matcher source maps
    /// construct each key exactly once; skipping the defensive linear lookup
    /// keeps the order-only audit helper proportional to the bucket-table
    /// work instead of turning the full mapper run quadratic in feature count.
    fn insert_unique(&mut self, key: u64) {
        self.insert_unique_with_hash(key, key);
    }

    /// The identity is independent from the hash so collisions cannot merge
    /// distinct keys. Existing integer-ID callers retain the identity hash.
    fn insert_unique_with_hash(&mut self, key: u64, hash: u64) {
        self.maybe_rehash_for_insert();
        let bucket = Self::bucket_for(hash, self.buckets.len());
        let index = self.nodes.len();
        self.nodes.push(UnorderedOrderNode {
            key,
            hash,
            next: None,
        });
        if let Some(predecessor) = self.buckets[bucket] {
            let next = self.after(predecessor);
            self.nodes[index].next = next;
            self.set_after(Some(predecessor), Some(index));
        } else {
            self.nodes[index].next = self.head;
            self.head = Some(index);
            if let Some(old_head) = self.nodes[index].next {
                let old_bucket = Self::bucket_for(self.nodes[old_head].hash, self.buckets.len());
                self.buckets[old_bucket] = Some(index);
            }
            self.buckets[bucket] = Some(BEFORE_BEGIN);
        }
        self.len += 1;
    }

    fn remove(&mut self, key: u64) -> bool {
        let mut predecessor = None;
        let mut current = self.head;
        while let Some(index) = current {
            if self.nodes[index].key == key {
                let next = self.nodes[index].next;
                self.set_after(predecessor, next);
                self.nodes[index].next = None;
                self.len = self.len.saturating_sub(1);
                self.rebuild_buckets();
                return true;
            }
            predecessor = current;
            current = self.nodes[index].next;
        }
        false
    }

    fn rebuild_buckets(&mut self) {
        self.buckets.fill(None);
        let mut predecessor = None;
        let mut current = self.head;
        while let Some(index) = current {
            let bucket = Self::bucket_for(self.nodes[index].hash, self.buckets.len());
            if self.buckets[bucket].is_none() {
                self.buckets[bucket] = Some(predecessor.unwrap_or(BEFORE_BEGIN));
            }
            predecessor = current;
            current = self.nodes[index].next;
        }
    }

    const fn iter(&self) -> UnorderedOrderIter<'_> {
        UnorderedOrderIter {
            order: self,
            next: self.head,
        }
    }

    fn keys(&self) -> impl Iterator<Item = u64> + '_ {
        self.iter().copied()
    }
}

/// Return keys in the insertion/rehash order of the pinned libstdc++
/// `std::unordered_map<uint64_t, ...>` implementation.  The value payload is
/// intentionally left to the caller; this is used where only source
/// container traversal order affects a floating-point or RANSAC boundary.
pub(crate) fn libstdcxx_unordered_keys<I>(keys: I) -> Vec<u64>
where
    I: IntoIterator<Item = u64>,
{
    let mut order = LibstdcxxUnorderedOrder::default();
    for key in keys {
        order.insert_unique(key);
    }
    order.iter().copied().collect()
}

pub(crate) struct UnorderedOrderIter<'a> {
    order: &'a LibstdcxxUnorderedOrder,
    next: Option<usize>,
}

impl<'a> Iterator for UnorderedOrderIter<'a> {
    type Item = &'a u64;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.next?;
        let node = &self.order.nodes[index];
        self.next = node.next;
        Some(&node.key)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct LibstdcxxUnorderedSet {
    order: LibstdcxxUnorderedOrder,
}

impl LibstdcxxUnorderedSet {
    pub(crate) fn insert(&mut self, key: u64) -> bool {
        self.order.insert(key)
    }

    pub(crate) const fn iter(&self) -> UnorderedOrderIter<'_> {
        self.order.iter()
    }
}

/// Persistent native host-map traversal. Identity remains the full TimeCamId;
/// the combined hash is used only to choose libstdc++ buckets.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct NativeHostOrder {
    order: LibstdcxxUnorderedOrder,
    identities: BTreeMap<(u64, u16), u64>,
    slots: Vec<(u64, u16)>,
}

impl NativeHostOrder {
    fn hash(timestamp_ns: u64, camera_id: u16) -> u64 {
        let mut seed = 0_u64;
        for value in [timestamp_ns, camera_id as u64] {
            seed ^= value
                .wrapping_add(0x9e3779b97f4a7c15)
                .wrapping_add(seed << 12)
                .wrapping_add(seed >> 4);
        }
        seed
    }

    pub(crate) fn insert(&mut self, timestamp_ns: u64, camera_id: u16) -> bool {
        let key = (timestamp_ns, camera_id);
        if self.identities.contains_key(&key) {
            return false;
        }
        let identity = self.slots.len() as u64;
        self.slots.push(key);
        self.identities.insert(key, identity);
        self.order
            .insert_unique_with_hash(identity, Self::hash(timestamp_ns, camera_id));
        true
    }

    pub(crate) fn remove(&mut self, timestamp_ns: u64, camera_id: u16) -> bool {
        let Some(identity) = self.identities.remove(&(timestamp_ns, camera_id)) else {
            return false;
        };
        self.order.remove(identity)
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = (u64, u16)> + '_ {
        self.order
            .keys()
            .map(|identity| self.slots[identity as usize])
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct ObservationDb {
    pub landmarks: BTreeMap<u64, LandmarkRecord>,
    pub observations: Vec<BearingObservation>,
    /// libstdc++ iteration order for the upstream landmark hash map.  The
    /// payload remains in `landmarks` so all existing map APIs stay intact.
    landmark_order: LibstdcxxUnorderedOrder,
}

impl ObservationDb {
    pub fn insert(&mut self, id: u64, landmark: InverseDistanceLandmark) {
        self.landmarks.insert(
            id,
            LandmarkRecord {
                landmark,
                connected_observations: 0,
                missed_frames: 0,
                status: LandmarkStatus::Active,
            },
        );
        self.landmark_order.insert(id);
    }
    pub fn associate(&mut self, obs: BearingObservation) {
        if let Some(l) = self.landmarks.get_mut(&obs.landmark_id) {
            l.connected_observations += 1;
            l.missed_frames = 0;
            l.status = LandmarkStatus::Active;
        }
        self.observations.push(obs);
    }
    pub fn mark_frame_end(&mut self, observed: &[u64], remove_after_misses: u32) {
        for (id, l) in &mut self.landmarks {
            if !observed.contains(id) {
                l.missed_frames += 1;
                l.status = if l.missed_frames >= remove_after_misses {
                    LandmarkStatus::Removed
                } else {
                    LandmarkStatus::Lost
                };
            }
        }
    }

    /// Mark landmarks that were not seen in the current camera interval as
    /// lost, but defer physical removal until the estimator has completed its
    /// window solve/marginalization.
    ///
    /// Basalt builds `lost_landmaks` before `optimize_and_marg()` and only
    /// calls `removeLandmark()` from the marginalization path.  In particular,
    /// the first four frames are kept by the initialization gate, so a track
    /// must still be present while those frames accumulate.  The older
    /// [`Self::mark_frame_end`] helper intentionally keeps its standalone
    /// threshold semantics for callers/tests that need them; the VIO
    /// estimator uses this deferred variant for the upstream lifecycle.
    pub fn mark_frame_end_deferred(&mut self, observed: &[u64]) {
        for (id, landmark) in &mut self.landmarks {
            if !observed.contains(id) {
                landmark.missed_frames = landmark.missed_frames.saturating_add(1);
                landmark.status = LandmarkStatus::Lost;
            }
        }
    }

    /// Remove landmarks marked lost by [`Self::mark_frame_end_deferred`].
    /// Returns the number of records removed.  Observations are pruned with
    /// their owning landmark, matching `LandmarkDatabase::removeLandmark`.
    pub fn remove_deferred_lost(&mut self) -> usize {
        let before = self.landmarks.len();
        let removed_ids = self
            .landmarks
            .iter()
            .filter_map(|(id, landmark)| (landmark.status != LandmarkStatus::Active).then_some(*id))
            .collect::<Vec<_>>();
        self.landmarks
            .retain(|_, landmark| landmark.status == LandmarkStatus::Active);
        for id in removed_ids {
            self.landmark_order.remove(id);
        }
        let removed = before.saturating_sub(self.landmarks.len());
        if removed > 0 {
            self.observations
                .retain(|observation| self.landmarks.contains_key(&observation.landmark_id));
        }
        removed
    }

    /// Remove every landmark hosted by one of the keyframes selected for
    /// marginalization, together with all observations belonging to those
    /// landmarks. This is the Rust equivalent of Basalt's
    /// `LandmarkDatabase::removeKeyframes` host branch: a selected host is
    /// deleted, never reanchored to the oldest retained keyframe.
    ///
    /// Observations of surviving landmarks are intentionally left untouched;
    /// target-frame cleanup is performed separately by
    /// [`Self::remove_frame_observations`], matching the two branches in the
    /// upstream implementation.
    pub fn remove_host_landmarks(&mut self, host_frame_ids: &[u64]) -> usize {
        if host_frame_ids.is_empty() || self.landmarks.is_empty() {
            return 0;
        }
        let removed_ids = self
            .landmarks
            .iter()
            .filter_map(|(id, record)| {
                host_frame_ids
                    .contains(&record.landmark.anchor_pose)
                    .then_some(*id)
            })
            .collect::<Vec<_>>();
        if removed_ids.is_empty() {
            return 0;
        }

        self.landmarks
            .retain(|_, record| !host_frame_ids.contains(&record.landmark.anchor_pose));
        for id in &removed_ids {
            self.landmark_order.remove(*id);
        }
        self.observations
            .retain(|observation| self.landmarks.contains_key(&observation.landmark_id));
        removed_ids.len()
    }

    /// Remove the given landmarks and their observations, as Basalt's
    /// `LandmarkDatabase::removeLandmark` does for an outlier landmark.
    pub fn remove_landmarks(&mut self, ids: &[u64]) -> usize {
        let before = self.landmarks.len();
        for id in ids {
            if self.landmarks.remove(id).is_some() {
                self.landmark_order.remove(*id);
            }
        }
        let removed = before - self.landmarks.len();
        if removed > 0 {
            self.observations
                .retain(|observation| self.landmarks.contains_key(&observation.landmark_id));
        }
        removed
    }

    /// Remove observations whose target frame left the active VIO window.
    /// Basalt's `LandmarkDatabase::removeKeyframes` performs this cleanup for
    /// dropped poses and full states while retaining observations of a state
    /// converted to a pose-only block.  The estimator's solver uses its
    /// separate `window_observations` index, so this bookkeeping cleanup does
    /// not alter factor construction or accumulation order.
    pub fn remove_frame_observations(&mut self, frame_ids: &[u64]) -> usize {
        if frame_ids.is_empty() || self.observations.is_empty() {
            return 0;
        }
        let before = self.observations.len();
        self.observations
            .retain(|observation| !frame_ids.contains(&observation.frame_id));
        let removed = before.saturating_sub(self.observations.len());
        if removed > 0 && self.observations.capacity() > self.observations.len().saturating_mul(2) {
            // `retain` leaves the old allocation in place.  Once a window
            // shift discarded enough history, release that capacity so a
            // long run does not keep the all-frames high-water mark resident.
            // The factor-of-two guard avoids reallocating on every shift when
            // the active history is close to the previous capacity.
            self.observations.shrink_to_fit();
        }
        removed
    }
    pub fn active_count(&self) -> usize {
        self.landmarks
            .values()
            .filter(|l| l.status == LandmarkStatus::Active)
            .count()
    }

    /// Iterate landmark IDs in the order produced by the upstream
    /// `aligned_unordered_map` range-for.  Missing IDs are filtered defensively
    /// because the public BTreeMap field remains available to callers.
    pub(crate) fn ordered_landmark_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.landmark_order
            .keys()
            .filter(|id| self.landmarks.contains_key(id))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyframeConfig {
    pub max_states: usize,
    pub max_kfs: usize,
    pub min_frames_after_kf: u64,
    pub new_kf_threshold: f64,
}
impl Default for KeyframeConfig {
    fn default() -> Self {
        Self {
            max_states: 3,
            max_kfs: 7,
            min_frames_after_kf: 5,
            new_kf_threshold: 0.7,
        }
    }
}

pub fn connected_ratio(connected: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        connected as f64 / total as f64
    }
}
pub fn should_insert_keyframe(
    frame_id: u64,
    last_kf_frame: Option<u64>,
    connected: usize,
    total: usize,
    states: usize,
    kfs: usize,
    cfg: KeyframeConfig,
) -> bool {
    if states >= cfg.max_states || kfs >= cfg.max_kfs {
        return false;
    }
    if let Some(last) = last_kf_frame {
        if frame_id.saturating_sub(last) < cfg.min_frames_after_kf {
            return false;
        }
    }
    connected_ratio(connected, total) < cfg.new_kf_threshold
}

pub fn bearing_from_pixel(camera: &DoubleSphereCamera, pixel: Point2<f64>) -> Option<Vector3<f64>> {
    camera.unproject(&pixel)
}

/// Triangulate from two world poses and normalized camera-frame bearings.
pub fn triangulate_two_rays(
    pose_a: &SE3,
    bearing_a: Vector3<f64>,
    pose_b: &SE3,
    bearing_b: Vector3<f64>,
) -> Option<(Vector3<f64>, f64)> {
    let a = bearing_a.try_normalize(1e-12)?;
    let b = bearing_b.try_normalize(1e-12);
    let b = b?;
    let oa = pose_a.translation;
    let ob = pose_b.translation;
    let da = pose_a.rotation.transform_vector(&a);
    let db = pose_b.rotation.transform_vector(&b);
    let v = ob - oa;
    let d = da.dot(&da);
    let e = da.dot(&db);
    let f = db.dot(&db);
    let den = d * f - e * e;
    if den.abs() < 1e-10 {
        return None;
    }
    let s = (v.dot(&da) * f - v.dot(&db) * e) / den;
    let t = (v.dot(&da) * e - v.dot(&db) * d) / den;
    let pa = oa + da * s;
    let pb = ob + db * t;
    let p = (pa + pb) * 0.5;
    if s <= 0.0 || t <= 0.0 {
        return None;
    }
    Some((p, 1.0 / s))
}

/// Basalt's `BundleAdjustmentBase::triangulate` contract: DLT/SVD in the
/// target camera frame, followed by unit-direction normalization and the
/// homogeneous inverse-distance component. `pose_target` and
/// `pose_candidate` are world-from-camera transforms.
pub fn triangulate_dlt(
    pose_target: &SE3,
    bearing_target: Vector3<f64>,
    pose_candidate: &SE3,
    bearing_candidate: Vector3<f64>,
) -> Option<(Vector3<f64>, f64)> {
    let f0 = bearing_target;
    let f1 = bearing_candidate;
    if !f0.iter().all(|value| value.is_finite())
        || !f1.iter().all(|value| value.is_finite())
        || f0.norm() <= 1e-12
        || f1.norm() <= 1e-12
    {
        return None;
    }
    let target_from_candidate = pose_target.inverse().compose(pose_candidate);
    let candidate_from_target = target_from_candidate.inverse().matrix();
    let identity = Matrix4::<f64>::identity();

    let mut a = Matrix4::<f64>::zeros();
    for column in 0..4 {
        a[(0, column)] = f0.x * identity[(2, column)] - f0.z * identity[(0, column)];
        a[(1, column)] = f0.y * identity[(2, column)] - f0.z * identity[(1, column)];
        a[(2, column)] =
            f1.x * candidate_from_target[(2, column)] - f1.z * candidate_from_target[(0, column)];
        a[(3, column)] =
            f1.y * candidate_from_target[(2, column)] - f1.z * candidate_from_target[(1, column)];
    }
    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    let mut homogeneous = Vector4::new(v_t[(3, 0)], v_t[(3, 1)], v_t[(3, 2)], v_t[(3, 3)]);
    let direction_norm = homogeneous.fixed_rows::<3>(0).norm();
    if !direction_norm.is_finite() || direction_norm <= 1e-12 {
        return None;
    }
    homogeneous /= direction_norm;
    let mut direction = Vector3::new(homogeneous[0], homogeneous[1], homogeneous[2]);
    let mut inverse_distance = homogeneous[3];
    // Same sign convention as the upstream implementation: the target ray
    // and recovered direction must point into the same half-space.
    if f0.dot(&direction) < 0.0 {
        direction = -direction;
        inverse_distance = -inverse_distance;
    }
    if !direction.iter().all(|value| value.is_finite()) || !inverse_distance.is_finite() {
        return None;
    }
    Some((direction, inverse_distance))
}

/// f32-owned counterpart of [`triangulate_dlt`].  Poses and bearings are
/// already converted at the estimator boundary, so the DLT matrix, SVD and
/// homogeneous normalization all round at the same points as Basalt's
/// `BundleAdjustmentBase<float>`.
pub fn triangulate_dlt_f32(
    target_rotation: &UnitQuaternion<f32>,
    target_translation: Vector3<f32>,
    target_bearing: Vector3<f32>,
    candidate_rotation: &UnitQuaternion<f32>,
    candidate_translation: Vector3<f32>,
    candidate_bearing: Vector3<f32>,
) -> Option<(Vector3<f32>, f32)> {
    let target = F32RigidPose {
        rotation: *target_rotation,
        translation: target_translation,
    };
    let candidate = F32RigidPose {
        rotation: *candidate_rotation,
        translation: candidate_translation,
    };
    triangulate_dlt_matrix_f32(
        target_bearing,
        candidate_bearing,
        target.inverse().compose(candidate).inverse(),
    )
}

/// Float-only rigid transform used by Basalt's triangulation path.  Sophus's
/// `SO3<float>` product and inverse both cross an Eigen quaternion boundary:
/// the packet QuaternionProduct is followed by packet normalization.  Keep
/// that boundary in the local packet helper rather than letting nalgebra
/// choose a scalar expression order.
#[derive(Clone, Copy)]
struct F32RigidPose {
    rotation: UnitQuaternion<f32>,
    translation: Vector3<f32>,
}

impl F32RigidPose {
    fn compose(self, other: Self) -> Self {
        Self {
            rotation: sophus_so3_product(self.rotation, other.rotation),
            translation: sophus_rotate(self.rotation, other.translation) + self.translation,
        }
    }

    fn inverse(self) -> Self {
        let rotation = sophus_so3_inverse(self.rotation);
        Self {
            rotation,
            translation: -sophus_rotate(rotation, self.translation),
        }
    }
}

pub(crate) fn sophus_so3_product(
    first: UnitQuaternion<f32>,
    second: UnitQuaternion<f32>,
) -> UnitQuaternion<f32> {
    sophus_packet::quaternion_product(first, second)
}

pub(crate) fn sophus_so3_inverse(rotation: UnitQuaternion<f32>) -> UnitQuaternion<f32> {
    sophus_packet::quaternion_inverse(rotation)
}

/// The pinned Eigen `Packet4f` quaternion path, expressed without unsafe
/// target-specific intrinsics.  The AVX/FMA branch preserves the coefficient
/// lane mapping and fused operation order recovered from the upstream
/// disassembly; the fallback keeps the same coefficient mapping with ordinary
/// scalar operations for targets without those features.
mod sophus_packet {
    use crate::fma::FusedMulAdd;
    use nalgebra::{Quaternion, UnitQuaternion};

    pub(super) fn quaternion_product(
        first: UnitQuaternion<f32>,
        second: UnitQuaternion<f32>,
    ) -> UnitQuaternion<f32> {
        let a = first.quaternion();
        let b = second.quaternion();
        let product = if packet_fma_available() {
            quaternion_product_fma([a.i, a.j, a.k, a.w], [b.i, b.j, b.k, b.w])
        } else {
            quaternion_product_scalar([a.i, a.j, a.k, a.w], [b.i, b.j, b.k, b.w])
        };
        let normalized = quaternion_normalize(product);
        UnitQuaternion::new_unchecked(Quaternion::new(
            normalized[3],
            normalized[0],
            normalized[1],
            normalized[2],
        ))
    }

    pub(super) fn quaternion_inverse(rotation: UnitQuaternion<f32>) -> UnitQuaternion<f32> {
        let q = rotation.quaternion();
        let conjugate = [-q.i, -q.j, -q.k, q.w];
        let normalized = quaternion_normalize(conjugate);
        UnitQuaternion::new_unchecked(Quaternion::new(
            normalized[3],
            normalized[0],
            normalized[1],
            normalized[2],
        ))
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn packet_fma_available() -> bool {
        std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma")
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[inline]
    fn packet_fma_available() -> bool {
        false
    }

    /// Eigen's packet product in `[x, y, z, w]` order.  Each `mul_add` is a
    /// scalar spelling of the corresponding packed FMA lane, and the order
    /// is deliberately the one emitted by the pinned native build.
    #[inline]
    fn quaternion_product_fma(first: [f32; 4], second: [f32; 4]) -> [f32; 4] {
        let [ax, ay, az, aw] = first;
        let [bx, by, bz, bw] = second;
        [
            // x = aw*bx + ax*bw + ay*bz - az*by
            (-az).fused_mul_add(by, ay.fused_mul_add(bz, aw.fused_mul_add(bx, ax * bw))),
            // y = aw*by + ay*bw + az*bx - ax*bz
            (-ax).fused_mul_add(bz, az.fused_mul_add(bx, aw.fused_mul_add(by, ay * bw))),
            // z = aw*bz + az*bw + ax*by - ay*bx
            (-ay).fused_mul_add(bx, ax.fused_mul_add(by, aw.fused_mul_add(bz, az * bw))),
            // w = aw*bw - ax*bx - ay*by - az*bz
            (-az).fused_mul_add(
                bz,
                (-ay).fused_mul_add(by, bw.fused_mul_add(aw, -(bx * ax))),
            ),
        ]
    }

    #[inline]
    fn quaternion_product_scalar(first: [f32; 4], second: [f32; 4]) -> [f32; 4] {
        let [ax, ay, az, aw] = first;
        let [bx, by, bz, bw] = second;
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by + ay * bw + az * bx - ax * bz,
            aw * bz + az * bw + ax * by - ay * bx,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    }

    /// Dispatch normalization separately so both the product and reduction
    /// honor the same runtime feature boundary.
    #[inline]
    fn quaternion_normalize(value: [f32; 4]) -> [f32; 4] {
        if packet_fma_available() {
            quaternion_normalize_fma(value)
        } else {
            quaternion_normalize_scalar(value)
        }
    }

    /// `vmulps; vmovhlps/vaddps; vmovshdup/vaddps; vsqrtss; broadcast;
    /// vdivps`, reduced in the same pairwise order as Eigen's packet path.
    #[inline]
    fn quaternion_normalize_fma(value: [f32; 4]) -> [f32; 4] {
        let [x, y, z, w] = value;
        let norm_sq = (x * x + z * z) + (y * y + w * w);
        let scale = norm_sq.sqrt();
        [x / scale, y / scale, z / scale, w / scale]
    }

    #[inline]
    fn quaternion_normalize_scalar(value: [f32; 4]) -> [f32; 4] {
        let [x, y, z, w] = value;
        let norm_sq = (x * x + z * z) + (y * y + w * w);
        let scale = norm_sq.sqrt();
        [x / scale, y / scale, z / scale, w / scale]
    }
}

fn sophus_rotate(rotation: UnitQuaternion<f32>, point: Vector3<f32>) -> Vector3<f32> {
    let q = rotation.quaternion();
    let uv = q.vector().cross(&point);
    let uv2 = uv + uv;
    point + q.w * uv2 + q.vector().cross(&uv2)
}

fn sophus_rotation_matrix(rotation: UnitQuaternion<f32>) -> Matrix3<f32> {
    let q = rotation.quaternion();
    let tx = 2.0_f32 * q.i;
    let ty = 2.0_f32 * q.j;
    let tz = 2.0_f32 * q.k;
    let txx = tx * q.i;
    let txy = ty * q.i;
    let txz = tz * q.i;
    let tyy = ty * q.j;
    let tyz = tz * q.j;
    let tzz = tz * q.k;
    Matrix3::new(
        1.0_f32 - (tyy + tzz),
        (-tz).fused_mul_add(q.w, txy),
        ty.fused_mul_add(q.w, txz),
        tz.fused_mul_add(q.w, txy),
        1.0_f32 - (txx + tzz),
        (-tx).fused_mul_add(q.w, tyz),
        (-ty).fused_mul_add(q.w, txz),
        tx.fused_mul_add(q.w, tyz),
        1.0_f32 - (txx + tyy),
    )
}

/// Literal `sqrt_keypoint_vio.cpp` triangulation transform:
/// `T_0_1 = T_i_c[0]^-1 * (T_w_i[0]^-1 * T_w_i[1]) * T_i_c[1]`.
/// Keeping the IMU/extrinsic products separate is material for f32 because
/// the upstream estimator never first forms world camera poses in f64.
// Pinned native measure() rig composition uses packet SO3 actions (including
// the inverse translations). Keep this separate from triangulate()'s final
// inverse, whose existing schedule is independently captured and unchanged.
fn triangulation_rig_pose_f32(
    target: F32RigidPose,
    candidate: F32RigidPose,
    target_extrinsic: F32RigidPose,
    candidate_extrinsic: F32RigidPose,
) -> F32RigidPose {
    use crate::vio::aom::sophus_rotate_step_packet_f32 as rotate;
    let inverse = |p: F32RigidPose| {
        let rotation = sophus_so3_inverse(p.rotation);
        F32RigidPose {
            rotation,
            translation: -rotate(rotation, p.translation),
        }
    };
    let compose = |a: F32RigidPose, b: F32RigidPose| F32RigidPose {
        rotation: sophus_so3_product(a.rotation, b.rotation),
        translation: rotate(a.rotation, b.translation) + a.translation,
    };
    let relative = compose(inverse(target), candidate);
    compose(
        compose(inverse(target_extrinsic), relative),
        candidate_extrinsic,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn triangulate_dlt_rig_f32(
    target_imu_rotation: &UnitQuaternion<f32>,
    target_imu_translation: Vector3<f32>,
    target_extrinsic_rotation: &UnitQuaternion<f32>,
    target_extrinsic_translation: Vector3<f32>,
    target_bearing: Vector3<f32>,
    candidate_imu_rotation: &UnitQuaternion<f32>,
    candidate_imu_translation: Vector3<f32>,
    candidate_extrinsic_rotation: &UnitQuaternion<f32>,
    candidate_extrinsic_translation: Vector3<f32>,
    candidate_bearing: Vector3<f32>,
) -> Option<(Vector3<f32>, f32)> {
    let target_imu = F32RigidPose {
        rotation: *target_imu_rotation,
        translation: target_imu_translation,
    };
    let candidate_imu = F32RigidPose {
        rotation: *candidate_imu_rotation,
        translation: candidate_imu_translation,
    };
    let target_extrinsic = F32RigidPose {
        rotation: *target_extrinsic_rotation,
        translation: target_extrinsic_translation,
    };
    let candidate_extrinsic = F32RigidPose {
        rotation: *candidate_extrinsic_rotation,
        translation: candidate_extrinsic_translation,
    };
    let target_camera_from_candidate = triangulation_rig_pose_f32(
        target_imu,
        candidate_imu,
        target_extrinsic,
        candidate_extrinsic,
    );
    let candidate_from_target = target_camera_from_candidate.inverse();
    triangulate_dlt_matrix_f32(target_bearing, candidate_bearing, candidate_from_target)
}

/// Diagnostic-only output from one f32 rig triangulation attempt.  The
/// production path still returns only the direction/rho pair; this fixed-size
/// payload is constructed only when the opt-in triangulation trace is active.
/// Arrays are row-major at the serialization boundary so they can be compared
/// directly with the pinned native captures.
#[derive(Debug, Clone)]
pub(crate) struct TriangulationDltTraceF32 {
    pub(crate) input_valid: bool,
    pub(crate) target_bearing: [f32; 3],
    pub(crate) candidate_bearing: [f32; 3],
    pub(crate) relative_imu_qxyzw_t: [f32; 7],
    pub(crate) camera_prefix_qxyzw_t: [f32; 7],
    pub(crate) target_camera_from_candidate_qxyzw_t: [f32; 7],
    pub(crate) candidate_pose_qxyzw_t: [f32; 7],
    pub(crate) candidate_rotation: [f32; 9],
    pub(crate) candidate_matrix: [f32; 16],
    pub(crate) dlt_matrix: [f32; 16],
    pub(crate) scale: Option<f32>,
    pub(crate) scaled_matrix: Option<[f32; 16]>,
    pub(crate) singular_values_scaled: Option<[f32; 4]>,
    pub(crate) singular_values: Option<[f32; 4]>,
    pub(crate) rank_estimate: Option<usize>,
    pub(crate) homogeneous_raw: Option<[f32; 4]>,
    pub(crate) direction_norm: Option<f32>,
    pub(crate) homogeneous_normalized: Option<[f32; 4]>,
    pub(crate) direction_before_sign: Option<[f32; 3]>,
    pub(crate) rho_before_sign: Option<f32>,
    pub(crate) sign_flipped: Option<bool>,
    pub(crate) direction_after_sign: Option<[f32; 3]>,
    pub(crate) rho_after_sign: Option<f32>,
}

impl Default for TriangulationDltTraceF32 {
    fn default() -> Self {
        Self {
            input_valid: false,
            target_bearing: [0.0; 3],
            candidate_bearing: [0.0; 3],
            relative_imu_qxyzw_t: [0.0; 7],
            camera_prefix_qxyzw_t: [0.0; 7],
            target_camera_from_candidate_qxyzw_t: [0.0; 7],
            candidate_pose_qxyzw_t: [0.0; 7],
            candidate_rotation: [0.0; 9],
            candidate_matrix: [0.0; 16],
            dlt_matrix: [0.0; 16],
            scale: None,
            scaled_matrix: None,
            singular_values_scaled: None,
            singular_values: None,
            rank_estimate: None,
            homogeneous_raw: None,
            direction_norm: None,
            homogeneous_normalized: None,
            direction_before_sign: None,
            rho_before_sign: None,
            sign_flipped: None,
            direction_after_sign: None,
            rho_after_sign: None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TriangulationAttemptF32 {
    pub(crate) result: Option<(Vector3<f32>, f32)>,
    pub(crate) dlt: TriangulationDltTraceF32,
}

/// Run the exact production f32 rig transform while retaining fixed-size
/// diagnostic intermediates.  No diagnostic allocation or arithmetic is
/// reached by `triangulate_dlt_rig_f32`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn triangulate_dlt_rig_f32_traced(
    target_imu_rotation: &UnitQuaternion<f32>,
    target_imu_translation: Vector3<f32>,
    target_extrinsic_rotation: &UnitQuaternion<f32>,
    target_extrinsic_translation: Vector3<f32>,
    target_bearing: Vector3<f32>,
    candidate_imu_rotation: &UnitQuaternion<f32>,
    candidate_imu_translation: Vector3<f32>,
    candidate_extrinsic_rotation: &UnitQuaternion<f32>,
    candidate_extrinsic_translation: Vector3<f32>,
    candidate_bearing: Vector3<f32>,
) -> TriangulationAttemptF32 {
    let target_imu = F32RigidPose {
        rotation: *target_imu_rotation,
        translation: target_imu_translation,
    };
    let candidate_imu = F32RigidPose {
        rotation: *candidate_imu_rotation,
        translation: candidate_imu_translation,
    };
    let target_extrinsic = F32RigidPose {
        rotation: *target_extrinsic_rotation,
        translation: target_extrinsic_translation,
    };
    let candidate_extrinsic = F32RigidPose {
        rotation: *candidate_extrinsic_rotation,
        translation: candidate_extrinsic_translation,
    };
    use crate::vio::aom::sophus_rotate_step_packet_f32 as rotate;
    let inverse = |pose: F32RigidPose| {
        let rotation = sophus_so3_inverse(pose.rotation);
        F32RigidPose {
            rotation,
            translation: -rotate(rotation, pose.translation),
        }
    };
    let compose = |first: F32RigidPose, second: F32RigidPose| F32RigidPose {
        rotation: sophus_so3_product(first.rotation, second.rotation),
        translation: rotate(first.rotation, second.translation) + first.translation,
    };
    let relative_imu = compose(inverse(target_imu), candidate_imu);
    let camera_prefix = compose(inverse(target_extrinsic), relative_imu);
    let target_camera_from_candidate = compose(camera_prefix, candidate_extrinsic);
    let candidate_from_target = target_camera_from_candidate.inverse();
    let pose_values = |pose: F32RigidPose| {
        let q = pose.rotation.quaternion();
        [
            q.i,
            q.j,
            q.k,
            q.w,
            pose.translation.x,
            pose.translation.y,
            pose.translation.z,
        ]
    };
    let candidate_q = candidate_from_target.rotation.quaternion();
    let mut dlt = TriangulationDltTraceF32 {
        target_bearing: [target_bearing.x, target_bearing.y, target_bearing.z],
        candidate_bearing: [
            candidate_bearing.x,
            candidate_bearing.y,
            candidate_bearing.z,
        ],
        relative_imu_qxyzw_t: pose_values(relative_imu),
        camera_prefix_qxyzw_t: pose_values(camera_prefix),
        target_camera_from_candidate_qxyzw_t: pose_values(target_camera_from_candidate),
        candidate_pose_qxyzw_t: [
            candidate_q.i,
            candidate_q.j,
            candidate_q.k,
            candidate_q.w,
            candidate_from_target.translation.x,
            candidate_from_target.translation.y,
            candidate_from_target.translation.z,
        ],
        candidate_rotation: matrix3_row_major(sophus_rotation_matrix(
            candidate_from_target.rotation,
        )),
        ..TriangulationDltTraceF32::default()
    };
    dlt.target_bearing = [target_bearing.x, target_bearing.y, target_bearing.z];
    dlt.candidate_bearing = [
        candidate_bearing.x,
        candidate_bearing.y,
        candidate_bearing.z,
    ];
    let result = triangulate_dlt_matrix_f32_with_trace(
        target_bearing,
        candidate_bearing,
        candidate_from_target,
        &mut dlt,
    );
    TriangulationAttemptF32 { result, dlt }
}

fn matrix3_row_major(matrix: Matrix3<f32>) -> [f32; 9] {
    let mut output = [0.0_f32; 9];
    for row in 0..3 {
        for column in 0..3 {
            output[row * 3 + column] = matrix[(row, column)];
        }
    }
    output
}

fn matrix4_row_major(matrix: Matrix4<f32>) -> [f32; 16] {
    let mut output = [0.0_f32; 16];
    for row in 0..4 {
        for column in 0..4 {
            output[row * 4 + column] = matrix[(row, column)];
        }
    }
    output
}

fn triangulate_dlt_matrix_f32(
    f0: Vector3<f32>,
    f1: Vector3<f32>,
    candidate_from_target: F32RigidPose,
) -> Option<(Vector3<f32>, f32)> {
    let mut trace = None;
    triangulate_dlt_matrix_f32_with_optional_trace(f0, f1, candidate_from_target, &mut trace)
}

fn triangulate_dlt_matrix_f32_with_trace(
    f0: Vector3<f32>,
    f1: Vector3<f32>,
    candidate_from_target: F32RigidPose,
    trace: &mut TriangulationDltTraceF32,
) -> Option<(Vector3<f32>, f32)> {
    let mut optional_trace = Some(trace);
    triangulate_dlt_matrix_f32_with_optional_trace(
        f0,
        f1,
        candidate_from_target,
        &mut optional_trace,
    )
}

fn triangulate_dlt_matrix_f32_with_optional_trace(
    f0: Vector3<f32>,
    f1: Vector3<f32>,
    candidate_from_target: F32RigidPose,
    trace: &mut Option<&mut TriangulationDltTraceF32>,
) -> Option<(Vector3<f32>, f32)> {
    if !f0.iter().all(|value| value.is_finite())
        || !f1.iter().all(|value| value.is_finite())
        || f0.norm() <= f32::EPSILON
        || f1.norm() <= f32::EPSILON
    {
        return None;
    }
    let candidate_rotation = sophus_rotation_matrix(candidate_from_target.rotation);
    let mut candidate_matrix = Matrix4::<f32>::identity();
    candidate_matrix
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&candidate_rotation);
    candidate_matrix
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&candidate_from_target.translation);

    if let Some(trace) = trace.as_mut() {
        trace.input_valid = true;
        trace.candidate_rotation = matrix3_row_major(candidate_rotation);
        trace.candidate_matrix = matrix4_row_major(candidate_matrix);
    }

    let mut a = Matrix4::<f32>::zeros();
    for column in 0..4 {
        a[(0, column)] = f0.x * if column == 2 { 1.0_f32 } else { 0.0_f32 }
            - f0.z * if column == 0 { 1.0_f32 } else { 0.0_f32 };
        a[(1, column)] = f0.y * if column == 2 { 1.0_f32 } else { 0.0_f32 }
            - f0.z * if column == 1 { 1.0_f32 } else { 0.0_f32 };
        a[(2, column)] = (-f1.z).fused_mul_add(
            candidate_matrix[(0, column)],
            f1.x * candidate_matrix[(2, column)],
        );
        // Seven fixed-size DLT product lanes use the association above. The
        // pinned compiler emits only the final (row3,col3) lane with the
        // opposite FMA operand at 4b2cd5, after rounding the other product.
        a[(3, column)] = if column == 3 {
            f1.y.fused_mul_add(
                candidate_matrix[(2, column)],
                (-f1.z) * candidate_matrix[(1, column)],
            )
        } else {
            (-f1.z).fused_mul_add(
                candidate_matrix[(1, column)],
                f1.y * candidate_matrix[(2, column)],
            )
        };
    }
    if let Some(trace) = trace.as_mut() {
        trace.dlt_matrix = matrix4_row_major(a);
    }
    let mut homogeneous = eigen_jacobi_svd_v_f32_with_optional_trace(a, trace)?;
    if let Some(trace) = trace.as_mut() {
        trace.homogeneous_raw = Some([
            homogeneous[0],
            homogeneous[1],
            homogeneous[2],
            homogeneous[3],
        ]);
    }
    let direction_norm = eigen_norm3_f32(homogeneous[0], homogeneous[1], homogeneous[2]);
    if !direction_norm.is_finite() || direction_norm <= f32::EPSILON {
        return None;
    }
    if let Some(trace) = trace.as_mut() {
        trace.direction_norm = Some(direction_norm);
    }
    homogeneous /= direction_norm;
    if let Some(trace) = trace.as_mut() {
        trace.homogeneous_normalized = Some([
            homogeneous[0],
            homogeneous[1],
            homogeneous[2],
            homogeneous[3],
        ]);
    }
    let mut direction = Vector3::new(homogeneous[0], homogeneous[1], homogeneous[2]);
    let mut inverse_distance = homogeneous[3];
    if let Some(trace) = trace.as_mut() {
        trace.direction_before_sign = Some([direction.x, direction.y, direction.z]);
        trace.rho_before_sign = Some(inverse_distance);
    }
    let sign_flipped = f0.dot(&direction) < 0.0;
    if sign_flipped {
        direction = -direction;
        inverse_distance = -inverse_distance;
    }
    if let Some(trace) = trace.as_mut() {
        trace.sign_flipped = Some(sign_flipped);
        trace.direction_after_sign = Some([direction.x, direction.y, direction.z]);
        trace.rho_after_sign = Some(inverse_distance);
    }
    if !direction.iter().all(|value| value.is_finite()) || !inverse_distance.is_finite() {
        return None;
    }
    Some((direction, inverse_distance))
}

/// The pinned Eigen `JacobiSVD<Matrix4f>` path.  `nalgebra::SVD` uses a
/// bidiagonal QR decomposition, which is mathematically equivalent but has a
/// different f32 rotation/order sequence.  Basalt asks Eigen for full V and
/// takes its last column for DLT, so port only that fixed-size real branch.
fn eigen_jacobi_svd_v_f32(matrix: Matrix4<f32>) -> Option<Vector4<f32>> {
    let mut trace = None;
    eigen_jacobi_svd_v_f32_with_optional_trace(matrix, &mut trace)
}

fn eigen_jacobi_svd_v_f32_with_optional_trace(
    matrix: Matrix4<f32>,
    trace: &mut Option<&mut TriangulationDltTraceF32>,
) -> Option<Vector4<f32>> {
    let mut scale = 0.0_f32;
    for value in matrix.iter() {
        scale = scale.max(value.abs());
    }
    if !scale.is_finite() {
        return None;
    }
    if scale == 0.0 {
        scale = 1.0;
    }
    let mut work = matrix.map(|value| value / scale);
    if let Some(trace) = trace.as_mut() {
        trace.scale = Some(scale);
        trace.scaled_matrix = Some(matrix4_row_major(work));
    }
    let mut v = Matrix4::<f32>::identity();
    let mut max_diag_entry = (0..4)
        .map(|index| work[(index, index)].abs())
        .fold(0.0_f32, f32::max);
    let precision = 2.0_f32 * f32::EPSILON;
    let consider_as_zero = f32::MIN_POSITIVE;

    loop {
        let mut finished = true;
        for p in 1..4 {
            for q in 0..p {
                let threshold = consider_as_zero.max(precision * max_diag_entry);
                if work[(p, q)].abs() > threshold || work[(q, p)].abs() > threshold {
                    finished = false;
                    let (left, right) = real_2x2_jacobi_svd_f32(
                        work[(p, p)],
                        work[(p, q)],
                        work[(q, p)],
                        work[(q, q)],
                    );
                    apply_left_f32(&mut work, p, q, left.0, left.1);
                    apply_right_f32(&mut work, p, q, right.0, right.1);
                    apply_right_f32(&mut v, p, q, right.0, right.1);
                    max_diag_entry = max_diag_entry
                        .max(work[(p, p)].abs())
                        .max(work[(q, q)].abs());
                }
            }
        }
        if finished {
            break;
        }
    }

    // Eigen sorts singular values descending and swaps the corresponding V
    // columns.  The scale cancels from this ordering, so no singular values
    // need to be retained beyond the diagonal magnitudes.
    for i in 0..4 {
        let mut max_pos = i;
        let mut max_value = work[(i, i)].abs();
        for j in (i + 1)..4 {
            if work[(j, j)].abs() > max_value {
                max_value = work[(j, j)].abs();
                max_pos = j;
            }
        }
        if max_value == 0.0 {
            break;
        }
        if max_pos != i {
            for row in 0..4 {
                let value = v[(row, i)];
                v[(row, i)] = v[(row, max_pos)];
                v[(row, max_pos)] = value;
            }
            let value = work[(i, i)];
            work[(i, i)] = work[(max_pos, max_pos)];
            work[(max_pos, max_pos)] = value;
        }
    }
    if let Some(trace) = trace.as_mut() {
        let singular_values_scaled = [
            work[(0, 0)].abs(),
            work[(1, 1)].abs(),
            work[(2, 2)].abs(),
            work[(3, 3)].abs(),
        ];
        let singular_values = singular_values_scaled.map(|value| value * scale);
        let rank_threshold = singular_values_scaled[0].max(1.0) * f32::EPSILON;
        trace.singular_values_scaled = Some(singular_values_scaled);
        trace.singular_values = Some(singular_values);
        trace.rank_estimate = Some(
            singular_values_scaled
                .iter()
                .filter(|value| **value > rank_threshold)
                .count(),
        );
    }
    let result = Vector4::new(v[(0, 3)], v[(1, 3)], v[(2, 3)], v[(3, 3)]);
    Some(result)
}

fn apply_left_f32(matrix: &mut Matrix4<f32>, p: usize, q: usize, c: f32, s: f32) {
    for column in 0..4 {
        let xp = matrix[(p, column)];
        let xq = matrix[(q, column)];
        // Eigen's scalar Jacobi path is compiled with contraction enabled on
        // the pinned native build. Spell the two sums as FMAs so the f32
        // rounding boundary is the same (the temporary values are kept
        // before either destination is written).
        matrix[(p, column)] = c.fused_mul_add(xp, s * xq);
        matrix[(q, column)] = (-s).fused_mul_add(xp, c * xq);
    }
}

fn apply_right_f32(matrix: &mut Matrix4<f32>, p: usize, q: usize, c: f32, s: f32) {
    for row in 0..4 {
        let xp = matrix[(row, p)];
        let xq = matrix[(row, q)];
        matrix[(row, p)] = c.fused_mul_add(xp, (-s) * xq);
        matrix[(row, q)] = s.fused_mul_add(xp, c * xq);
    }
}

fn real_2x2_jacobi_svd_f32(m00: f32, m01: f32, m10: f32, m11: f32) -> ((f32, f32), (f32, f32)) {
    let t = m00 + m11;
    let d = m10 - m01;
    let (left1_c, left1_s) = if d.abs() < f32::MIN_POSITIVE {
        (1.0_f32, 0.0_f32)
    } else {
        let u = t / d;
        let tmp = u.fused_mul_add(u, 1.0_f32).sqrt();
        (u / tmp, 1.0_f32 / tmp)
    };
    // `rot1 * m` is the temporary 2x2 block used by Eigen before deriving
    // its right Jacobi rotation.
    let a00 = left1_c.fused_mul_add(m00, left1_s * m10);
    let a01 = left1_c.fused_mul_add(m01, left1_s * m11);
    let a11 = (-left1_s).fused_mul_add(m01, left1_c * m11);
    let deno = 2.0_f32 * a01.abs();
    let (right_c, right_s) = if deno < f32::MIN_POSITIVE {
        (1.0_f32, 0.0_f32)
    } else {
        let tau = (a00 - a11) / deno;
        let w = tau.fused_mul_add(tau, 1.0_f32).sqrt();
        let t = if tau > 0.0_f32 {
            1.0_f32 / (tau + w)
        } else {
            1.0_f32 / (tau - w)
        };
        let sign_t = if t > 0.0_f32 { 1.0_f32 } else { -1.0_f32 };
        let n = 1.0_f32 / t.fused_mul_add(t, 1.0_f32).sqrt();
        (n, -sign_t * (a01 / a01.abs()) * t.abs() * n)
    };
    // Eigen stores j_left = rot1 * j_right.transpose().
    let left_c = left1_c.fused_mul_add(right_c, left1_s * right_s);
    let left_s = (-left1_c).fused_mul_add(right_s, left1_s * right_c);
    ((left_c, left_s), (right_c, right_s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires external native operand capture on E"]
    fn m11_frame7_triangulation_all_operand_probe() {
        use serde_json::Value;
        let root = std::env::var("M11_TRI_PROBE_ROOT").expect("capture root");
        let directory = format!("{root}/m11_native_frame7_triangulation_operands_20260908/r1");
        for gate in [
            "engine.rc",
            "gdb.wrapper.rc",
            "probe.validation.rc",
            "neutrality.validation.rc",
            "binding.pre_post.cmp.rc",
        ] {
            assert_eq!(
                std::fs::read_to_string(format!("{directory}/{gate}"))
                    .unwrap()
                    .trim(),
                "0"
            );
        }
        fn pose(v: &Value) -> F32RigidPose {
            let f =
                |i: usize| f32::from_bits(u32::from_str_radix(v[i].as_str().unwrap(), 16).unwrap());
            F32RigidPose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(f(3), f(0), f(1), f(2))),
                translation: Vector3::new(f(4), f(5), f(6)),
            }
        }
        let text = std::fs::read_to_string(format!("{directory}/relative_operands.jsonl")).unwrap();
        let mut unique = std::collections::BTreeSet::new();
        let mut counts = [0usize; 4];
        let mut total = 0;
        for line in text.lines() {
            let v: Value = serde_json::from_str(line).unwrap();
            let target = pose(&v["target_stack_300"]);
            let candidate = pose(&v["candidate_qxyzw_t"]);
            let native_inverse = pose(&v["inverse_target_qxyzw_t"]);
            let inverse_rotation = sophus_so3_inverse(target.rotation);
            let candidates = [
                -sophus_rotate(inverse_rotation, target.translation),
                -crate::vio::aom::sophus_rotate_step_packet_f32(
                    inverse_rotation,
                    target.translation,
                ),
                sophus_rotate(native_inverse.rotation, candidate.translation),
                crate::vio::aom::sophus_rotate_step_packet_f32(
                    native_inverse.rotation,
                    candidate.translation,
                ),
            ];
            let expected = [
                native_inverse.translation.x.to_bits(),
                native_inverse.translation.y.to_bits(),
                native_inverse.translation.z.to_bits(),
            ];
            let rotated: Vec<u32> = v["candidate_rotated_translation"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| u32::from_str_radix(x.as_str().unwrap(), 16).unwrap())
                .collect();
            let mut exact = [false; 4];
            for (i, c) in candidates.iter().enumerate() {
                let bits: Vec<u32> = c.iter().map(|x| x.to_bits()).collect();
                exact[i] = bits.as_slice() == if i < 2 { &expected[..] } else { &rotated[..] };
                counts[i] += usize::from(exact[i]);
            }
            unique.insert(format!(
                "{}:{}",
                v["target_stack_300"], v["candidate_qxyzw_t"]
            ));
            println!(
                "M11_ALL_OPERAND {}",
                serde_json::json!({"ordinal":v["ordinal"],"exact":exact})
            );
            total += 1;
        }
        assert_eq!(total, 159);
        println!(
            "M11_ALL_OPERAND_SUMMARY {}",
            serde_json::json!({"records":total,"unique_pose_pairs":unique.len(),"order":["inverse_generic","inverse_packet","candidate_generic","candidate_packet"],"exact_records":counts})
        );
    }

    #[test]
    #[ignore = "requires external native/Rust triangulation captures on E"]
    fn m11_frame7_triangulation_pose_stage_probe() {
        use serde_json::Value;
        let root = std::env::var("M11_TRI_PROBE_ROOT").expect("capture root");
        let trace = std::fs::read_to_string(format!(
            "{root}/m11_active_observations_triangulation_20260908/r1/triangulation.jsonl"
        ))
        .unwrap();
        let record: Value = trace
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .find(|r| r["event_frame_id"] == 7 && r["track_id"] == 127)
            .unwrap();
        let attempt = &record["attempts"][0];
        fn pose(v: &Value, normalize: bool) -> F32RigidPose {
            let q = &v["rotation_wxyz"]["values"];
            let scalar = |i: usize| q[i].as_f64().unwrap() as f32;
            let q = Quaternion::new(scalar(0), scalar(1), scalar(2), scalar(3));
            let t = &v["translation"]["values"];
            F32RigidPose {
                rotation: if normalize {
                    UnitQuaternion::from_quaternion(q)
                } else {
                    UnitQuaternion::new_unchecked(q)
                },
                translation: Vector3::new(
                    t[0].as_f64().unwrap() as f32,
                    t[1].as_f64().unwrap() as f32,
                    t[2].as_f64().unwrap() as f32,
                ),
            }
        }
        fn words(p: F32RigidPose) -> Vec<String> {
            [
                p.rotation.i,
                p.rotation.j,
                p.rotation.k,
                p.rotation.w,
                p.translation.x,
                p.translation.y,
                p.translation.z,
            ]
            .iter()
            .map(|v| format!("{:08x}", v.to_bits()))
            .collect()
        }
        let target = pose(&record["target_history"]["imu_pose_current_raw"], false);
        let candidate = pose(&attempt["candidate"]["imu_pose_current_raw"], false);
        let te = pose(&attempt["target_extrinsic"], true);
        let ce = pose(&attempt["candidate_extrinsic"], true);
        let relative = target.inverse().compose(candidate);
        let inverse_rotation = sophus_so3_inverse(target.rotation);
        let inverse_generic = -sophus_rotate(inverse_rotation, target.translation);
        let inverse_packet =
            -crate::vio::aom::sophus_rotate_step_packet_f32(inverse_rotation, target.translation);
        let candidate_generic = sophus_rotate(inverse_rotation, candidate.translation);
        let candidate_packet =
            crate::vio::aom::sophus_rotate_step_packet_f32(inverse_rotation, candidate.translation);
        let vector_words = |v: Vector3<f32>| {
            v.iter()
                .map(|x| format!("{:08x}", x.to_bits()))
                .collect::<Vec<_>>()
        };
        println!(
            "M11_TRI_OPERANDS {}",
            serde_json::json!({
                "target": words(target), "candidate": words(candidate),
                "inverse_generic": words(F32RigidPose { rotation: inverse_rotation, translation: inverse_generic }),
                "inverse_packet": words(F32RigidPose { rotation: inverse_rotation, translation: inverse_packet }),
                "candidate_generic": vector_words(candidate_generic),
                "candidate_packet": vector_words(candidate_packet)
            })
        );
        let relative_translation_variants = [
            inverse_generic + candidate_generic,
            inverse_generic + candidate_packet,
            inverse_packet + candidate_generic,
            inverse_packet + candidate_packet,
        ]
        .map(|v| {
            v.iter()
                .map(|x| format!("{:08x}", x.to_bits()))
                .collect::<Vec<_>>()
        });
        println!(
            "M11_TRI_RELATIVE_VARIANTS {}",
            serde_json::json!({"order":["generic_generic","generic_packet","packet_generic","packet_packet"],"translations":relative_translation_variants})
        );
        let prefix = te.inverse().compose(relative);
        let suffix_rotated = sophus_rotate(prefix.rotation, ce.translation);
        let forward = prefix.compose(ce);
        let native_matches: Value = serde_json::from_str(
            &std::fs::read_to_string(format!(
                "{root}/m11_native_frame7_triangulation_inputs_20260908/r1/bearing_matches.json"
            ))
            .unwrap(),
        )
        .unwrap();
        let matched = native_matches["results"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["track_id"] == 127)
            .unwrap();
        let native = &matched["matches"][0]["T_0_1_qxyzw_t"];
        let f = |i: usize| {
            f32::from_bits(u32::from_str_radix(native[i].as_str().unwrap(), 16).unwrap())
        };
        let native_forward = F32RigidPose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(f(3), f(0), f(1), f(2))),
            translation: Vector3::new(f(4), f(5), f(6)),
        };
        assert_eq!(
            words(triangulation_rig_pose_f32(target, candidate, te, ce)),
            words(native_forward)
        );
        // Diagnostic only: keep quaternion operations fixed and identify the
        // remaining camera translation schedules against the captured chain.
        let packet_relative = F32RigidPose {
            rotation: relative.rotation,
            translation: inverse_packet + candidate_packet,
        };
        let rotate = |q, p, packet| {
            if packet {
                crate::vio::aom::sophus_rotate_step_packet_f32(q, p)
            } else {
                sophus_rotate(q, p)
            }
        };
        for mask in 0..8 {
            let qi = sophus_so3_inverse(te.rotation);
            let ti = -rotate(qi, te.translation, mask & 1 != 0);
            let p = F32RigidPose {
                rotation: sophus_so3_product(qi, packet_relative.rotation),
                translation: rotate(qi, packet_relative.translation, mask & 2 != 0) + ti,
            };
            let suffix = rotate(p.rotation, ce.translation, mask & 4 != 0);
            let result = F32RigidPose {
                rotation: sophus_so3_product(p.rotation, ce.rotation),
                translation: suffix + p.translation,
            };
            let actual = words(result);
            let expected = words(native_forward);
            println!(
                "M11_TRI_CAMERA_SCHEDULE {}",
                serde_json::json!({
                    "mask":mask,"bit_order":["extrinsic_inverse_packet","prefix_action_packet","suffix_action_packet"],
                    "prefix":words(p),"suffix_rotated":vector_words(suffix),"forward":actual,
                    "exact":actual.iter().zip(expected.iter()).filter(|(a,b)|a==b).count()
                })
            );
        }
        println!(
            "M11_TRI_POSE {}",
            serde_json::json!({"relative":words(relative),"prefix":words(prefix),"suffix_rotated":suffix_rotated.iter().map(|v| format!("{:08x}",v.to_bits())).collect::<Vec<_>>(),"rust_forward":words(forward),"native_forward":words(native_forward),"rust_inverse":words(forward.inverse()),"rust_inverse_from_native_forward":words(native_forward.inverse())})
        );
        assert_eq!(native.as_array().unwrap().len(), 7);
    }

    #[test]
    fn libstdcxx_unordered_order_matches_gcc11_growth_and_rehash() {
        let mut candidate_set = LibstdcxxUnorderedSet::default();
        for key in 0..135 {
            candidate_set.insert(key);
        }
        let set_order = candidate_set.iter().copied().collect::<Vec<_>>();
        let expected_set_order = (127..=134)
            .rev()
            .chain((29..=58).rev())
            .chain((0..=12).rev())
            .chain(13..=28)
            .chain(59..=126)
            .collect::<Vec<_>>();
        assert_eq!(set_order, expected_set_order);

        let mut map_from_insertion_order = LibstdcxxUnorderedOrder::default();
        for key in 0..135 {
            map_from_insertion_order.insert(key);
        }
        assert_eq!(
            map_from_insertion_order.keys().collect::<Vec<_>>(),
            expected_set_order
        );

        // Filter the generic candidate stream, as the upstream triangulation
        // pass does when only a subset of observations succeeds.  No
        // landmark/fixture ID is encoded in the implementation itself.
        let mut map_order = LibstdcxxUnorderedOrder::default();
        for key in set_order.iter().copied().filter(|key| key % 2 == 0) {
            map_order.insert(key);
        }
        assert_eq!(map_order.keys().count(), 68);
        assert_eq!(
            map_order
                .keys()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            68
        );
    }

    #[test]
    fn libstdcxx_unordered_order_handles_erase_and_reinsert() {
        let mut order = LibstdcxxUnorderedOrder::default();
        for key in 0..135 {
            order.insert(key);
        }
        for key in [0, 13, 58, 127, 64, 91] {
            assert!(order.remove(key));
        }
        for key in [200, 201, 202, 203, 13, 58] {
            assert!(order.insert(key));
        }

        let expected = [58, 13, 203, 202, 201, 200]
            .into_iter()
            .chain((128..=134).rev())
            .chain((29..=57).rev())
            .chain((1..=12).rev())
            .chain(14..=28)
            .chain(59..=63)
            .chain(65..=90)
            .chain(92..=126)
            .collect::<Vec<_>>();
        assert_eq!(order.keys().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn libstdcxx_unordered_order_preserves_distinct_hash_collisions() {
        let mut order = LibstdcxxUnorderedOrder::default();
        for key in [1, 2, 3] {
            order.insert_unique_with_hash(key, 7);
        }
        assert_eq!(order.keys().collect::<Vec<_>>(), vec![3, 2, 1]);
        assert!(order.remove(2));
        assert_eq!(order.keys().collect::<Vec<_>>(), vec![3, 1]);
        order.rehash(53);
        assert_eq!(order.keys().collect::<Vec<_>>(), vec![1, 3]);
        order.insert_unique_with_hash(2, 7);
        assert_eq!(order.keys().collect::<Vec<_>>(), vec![2, 1, 3]);
        for key in [1, 2, 3] {
            assert!(order.find_node(key).is_some());
        }
    }

    #[test]
    fn native_host_order_tracks_timestamp_camera_identity_and_reinsertion() {
        let mut hosts = NativeHostOrder::default();
        let first = 1403636579763555584;
        let second = 1403636580113555456;
        assert!(hosts.insert(first, 0));
        assert!(hosts.insert(second, 0));
        assert_eq!(
            hosts.keys().collect::<Vec<_>>(),
            vec![(second, 0), (first, 0)]
        );
        assert!(!hosts.insert(first, 0));
        assert!(hosts.insert(first, 1));
        assert!(hosts.remove(first, 0));
        assert!(!hosts.remove(first, 0));
        assert!(hosts.insert(first, 0));
        let keys = hosts.keys().collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            keys,
            [(first, 0), (first, 1), (second, 0)].into_iter().collect()
        );
        assert_eq!(hosts.keys().count(), 3);
        assert_eq!(hosts.clone(), hosts);
    }

    #[test]
    #[ignore = "requires external GCC11 host-order operation oracle"]
    fn native_host_order_matches_gcc11_operation_oracle() {
        use std::io::BufRead;
        let path = std::env::var("M11_HOST_ORDER_ORACLE").expect("explicit oracle path required");
        let mut hosts = NativeHostOrder::default();
        let mut count = 0;
        for line in std::io::BufReader::new(std::fs::File::open(path).unwrap()).lines() {
            let r: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
            let decode = |v: &serde_json::Value| {
                (
                    v[0].as_u64().unwrap(),
                    u16::try_from(v[1].as_u64().unwrap()).unwrap(),
                )
            };
            let (timestamp, camera) = decode(&r["key"]);
            assert_eq!(
                NativeHostOrder::hash(timestamp, camera),
                r["hash"].as_u64().unwrap()
            );
            let changed = match r["operation"].as_str().unwrap() {
                "insert" => hosts.insert(timestamp, camera),
                "remove" => hosts.remove(timestamp, camera),
                other => panic!("unexpected operation {other}"),
            };
            assert_eq!(changed, r["changed"].as_bool().unwrap());
            assert_eq!(
                hosts.order.buckets.len() as u64,
                r["buckets"].as_u64().unwrap()
            );
            let expected = r["order"]
                .as_array()
                .unwrap()
                .iter()
                .map(decode)
                .collect::<Vec<_>>();
            assert_eq!(
                hosts.keys().collect::<Vec<_>>(),
                expected,
                "operation {count}"
            );
            count += 1;
        }
        assert_eq!(count, 227);
    }

    #[test]
    fn libstdcxx_unordered_order_uses_sparse_prime_growth_beyond_fixture_sizes() {
        // These requests exercise the sparse GCC 11 table at and beyond the
        // sizes where an "all ordinary primes" approximation first diverges.
        assert_eq!(LibstdcxxUnorderedOrder::next_bucket(514), 541);
        assert_eq!(LibstdcxxUnorderedOrder::next_bucket(1_082), 1_109);
        assert_eq!(LibstdcxxUnorderedOrder::next_bucket(10_000), 10_273);
        assert_eq!(LibstdcxxUnorderedOrder::next_bucket(2_000_000), 2_144_977);

        // Also drive the order through ten thousand generic keys.  This is
        // intentionally independent of any MH01 track IDs; the two-million
        // boundary above is covered directly through the same rehash policy
        // without making a quadratic duplicate lookup test.
        let mut order = LibstdcxxUnorderedOrder::default();
        for key in 0..10_001 {
            assert!(order.insert(key));
        }
        assert_eq!(order.len, 10_001);
        assert_eq!(order.buckets.len(), 10_273);
        assert_eq!(order.keys().count(), 10_001);
    }

    use nalgebra::{Quaternion, UnitQuaternion, Vector3};

    #[test]
    fn f32_sophus_packet_product_and_normalize_match_golden_bits() {
        // The pinned Eigen build takes this path only when Packet4f is
        // available.  The portable fallback remains covered by the ordinary
        // geometry tests, while this exact contract is meaningful on the
        // x86_64 AVX/FMA runner used for the Basalt replay.
        if !simd_available() {
            return;
        }
        let s = UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(0x3f182ffd),
            f32::from_bits(0xbd582e43),
            f32::from_bits(0xbf4d686f),
            f32::from_bits(0x00000000),
        ));
        let sinv = sophus_so3_inverse(s);
        let sinv_s = sophus_so3_product(sinv, s);
        let sinv_q = sinv.quaternion();
        let sinv_s_q = sinv_s.quaternion();
        let sinv_bits = [
            sinv_q.i.to_bits(),
            sinv_q.j.to_bits(),
            sinv_q.k.to_bits(),
            sinv_q.w.to_bits(),
        ];
        let sinv_s_bits = [
            sinv_s_q.i.to_bits(),
            sinv_s_q.j.to_bits(),
            sinv_s_q.k.to_bits(),
            sinv_s_q.w.to_bits(),
        ];
        assert_eq!(sinv_bits, [0x3d582e43, 0x3f4d686f, 0x80000000, 0x3f182ffd]);
        assert_eq!(
            sinv_s_bits,
            [0x2f9fd648, 0xb1a4b598, 0x30391c34, 0x3f800000]
        );
    }

    #[test]
    fn f32_sophus_packet_relative_stage_matches_golden_bits() {
        if !simd_available() {
            return;
        }
        let pose = |q: [u32; 4], t: [u32; 3]| F32RigidPose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(q[3]),
                f32::from_bits(q[0]),
                f32::from_bits(q[1]),
                f32::from_bits(q[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(t[0]),
                f32::from_bits(t[1]),
                f32::from_bits(t[2]),
            ),
        };
        let t0 = pose(
            [0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e],
            [0xbc896b48, 0xbd8d2fdc, 0x3ba86617],
        );
        let t1 = pose(
            [0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6],
            [0xbc76fa76, 0x3d290319, 0x3b4f4832],
        );
        let state = pose(
            [0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd],
            [0x00000000, 0x00000000, 0x00000000],
        );
        let stage = t0
            .inverse()
            .compose(state.inverse().compose(state))
            .compose(t1);
        let q = stage.rotation.quaternion();
        assert_eq!(
            [
                stage.translation.x.to_bits(),
                stage.translation.y.to_bits(),
                stage.translation.z.to_bits(),
            ],
            [0x3de1c150, 0xb87a2400, 0x39ac1888]
        );
        assert_eq!(
            [q.i.to_bits(), q.j.to_bits(), q.k.to_bits(), q.w.to_bits()],
            [0x3befaa96, 0x39eadb91, 0x3a8bfffa, 0x3f7ffe35]
        );
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_available() -> bool {
        std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma")
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_available() -> bool {
        false
    }

    #[test]
    fn stereographic_round_trip_is_three_parameter_contract() {
        let b = Vector3::new(0.2, -0.3, 0.9327379).normalize();
        let d = StereographicDirection::from_bearing(b).unwrap();
        assert!((d.bearing() - b).norm() < 1e-7);
        assert!(InverseDistanceLandmark {
            anchor_pose: 0,
            anchor_camera_id: 0,
            direction: d,
            inverse_distance: 0.5
        }
        .position_in_anchor()
        .is_some());
    }

    #[test]
    fn upstream_f32_landmark_writeback_rounds_each_add() {
        let mut landmark = InverseDistanceLandmark {
            anchor_pose: 0,
            anchor_camera_id: 0,
            direction: StereographicDirection {
                xy: Point2::new(
                    f32::from_bits(0xbe912959) as f64,
                    f32::from_bits(0xbe205423) as f64,
                ),
            },
            inverse_distance: f32::from_bits(0x3e3e267e) as f64,
        };
        assert!(landmark.apply_increment_f32(Vector3::new(
            f32::from_bits(0xb93d57e3) as f64,
            f32::from_bits(0xb9540d36) as f64,
            f32::from_bits(0x3b1cb25e) as f64,
        )));
        assert_eq!(landmark.direction.xy.x as f32, f32::from_bits(0xbe914104));
        assert_eq!(landmark.direction.xy.y as f32, f32::from_bits(0xbe208926));
        assert_eq!(landmark.inverse_distance as f32, f32::from_bits(0x3e409947));
    }

    #[test]
    fn temporal_triangulation_recovers_synthetic_point() {
        let a = SE3::identity();
        let b = SE3::new(UnitQuaternion::identity(), Vector3::new(1.0, 0.0, 0.0));
        let p = Vector3::new(0.0, 0.0, 4.0);
        let pa = a.inverse().rotation.transform_vector(&p);
        let pb = b.inverse().rotation.transform_vector(&(p - b.translation));
        let (q, inv) = triangulate_two_rays(&a, pa, &b, pb).unwrap();
        assert!((q - p).norm() < 1e-10);
        assert!((inv - 0.25).abs() < 1e-10);
    }

    #[test]
    fn dlt_triangulation_matches_basalt_homogeneous_contract() {
        let target = SE3::identity();
        let candidate = SE3::new(UnitQuaternion::identity(), Vector3::new(1.0, 0.0, 0.0));
        let point = Vector3::new(0.2, -0.1, 4.0);
        let target_bearing = point.normalize();
        let candidate_bearing = (point - candidate.translation).normalize();
        let (direction, inverse_distance) =
            triangulate_dlt(&target, target_bearing, &candidate, candidate_bearing).unwrap();
        assert!((direction - target_bearing).norm() < 1e-10);
        assert!((inverse_distance - 1.0 / point.norm()).abs() < 1e-10);
    }

    #[test]
    fn camera_aware_reanchor_preserves_world_point() {
        let old_anchor = SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.02, -0.03, 0.01)),
            Vector3::new(0.2, -0.1, 0.05),
        );
        let new_anchor = SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(-0.04, 0.01, 0.03)),
            Vector3::new(-0.15, 0.08, 0.12),
        );
        let point_world = Point3::new(0.3, -0.2, 3.5);
        let point_old = old_anchor.inverse().transform_point(&point_world);
        let mut landmark = InverseDistanceLandmark {
            anchor_pose: 1,
            anchor_camera_id: 1,
            direction: StereographicDirection::from_bearing(point_old.coords).unwrap(),
            inverse_distance: 1.0 / point_old.coords.norm(),
        };
        assert!(landmark.reanchor(9, 0, &new_anchor, point_world));
        let reconstructed =
            new_anchor.transform_point(&Point3::from(landmark.position_in_anchor().unwrap()));
        assert_eq!(landmark.anchor_pose, 9);
        assert_eq!(landmark.anchor_camera_id, 0);
        assert!((reconstructed - point_world).norm() < 1e-12);
    }
    #[test]
    fn association_and_lost_landmark_lifecycle_is_deterministic() {
        let d = StereographicDirection::from_bearing(Vector3::z()).unwrap();
        let mut db = ObservationDb::default();
        db.insert(
            1,
            InverseDistanceLandmark {
                anchor_pose: 0,
                anchor_camera_id: 0,
                direction: d,
                inverse_distance: 1.0,
            },
        );
        db.associate(BearingObservation {
            landmark_id: 1,
            frame_id: 0,
            bearing: Vector3::z(),
        });
        assert_eq!(db.active_count(), 1);
        db.mark_frame_end(&[], 2);
        assert_eq!(db.landmarks[&1].status, LandmarkStatus::Lost);
        db.mark_frame_end(&[], 2);
        assert_eq!(db.landmarks[&1].status, LandmarkStatus::Removed);
    }

    #[test]
    fn deferred_loss_keeps_records_until_marginalization() {
        let d = StereographicDirection::from_bearing(Vector3::z()).unwrap();
        let mut db = ObservationDb::default();
        db.insert(
            9,
            InverseDistanceLandmark {
                anchor_pose: 0,
                anchor_camera_id: 0,
                direction: d,
                inverse_distance: 1.0,
            },
        );
        db.mark_frame_end_deferred(&[]);
        assert_eq!(db.landmarks[&9].status, LandmarkStatus::Lost);
        assert_eq!(db.landmarks.len(), 1);
        assert_eq!(db.remove_deferred_lost(), 1);
        assert!(db.landmarks.is_empty());
    }

    #[test]
    fn selected_host_landmarks_are_deleted_without_reanchoring() {
        let d = StereographicDirection::from_bearing(Vector3::z()).unwrap();
        let mut db = ObservationDb::default();
        db.insert(
            1,
            InverseDistanceLandmark {
                anchor_pose: 7,
                anchor_camera_id: 0,
                direction: d,
                inverse_distance: 1.0,
            },
        );
        db.insert(
            2,
            InverseDistanceLandmark {
                anchor_pose: 9,
                anchor_camera_id: 0,
                direction: d,
                inverse_distance: 1.0,
            },
        );
        for (landmark_id, frame_id) in [(1, 8), (1, 9), (2, 8), (2, 9)] {
            db.associate(BearingObservation {
                landmark_id,
                frame_id,
                bearing: Vector3::z(),
            });
        }

        assert_eq!(db.remove_host_landmarks(&[7]), 1);
        assert!(!db.landmarks.contains_key(&1));
        assert_eq!(db.landmarks[&2].landmark.anchor_pose, 9);
        assert_eq!(db.observations.len(), 2);
        assert!(db
            .observations
            .iter()
            .all(|observation| observation.landmark_id == 2));
        assert_eq!(db.ordered_landmark_ids().collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn frame_removal_prunes_only_dropped_observation_history() {
        let d = StereographicDirection::from_bearing(Vector3::z()).unwrap();
        let mut db = ObservationDb::default();
        db.insert(
            9,
            InverseDistanceLandmark {
                anchor_pose: 0,
                anchor_camera_id: 0,
                direction: d,
                inverse_distance: 1.0,
            },
        );
        for frame_id in [3, 4, 5] {
            db.associate(BearingObservation {
                landmark_id: 9,
                frame_id,
                bearing: Vector3::z(),
            });
        }
        assert_eq!(db.remove_frame_observations(&[3, 4]), 2);
        assert_eq!(db.observations.len(), 1);
        assert_eq!(db.observations[0].frame_id, 5);
    }

    #[test]
    fn m7_eigen_vector_block_norm_reduction_matches_pinned_lanes() {
        let bits = |value: u32| f32::from_bits(value);
        // `BundleAdjustmentBase::triangulate` normalizes a Vec4 through
        // `world_point.head<3>().norm()`.  Eigen's fixed VectorBlock packet
        // evaluator uses `x*x + (z*z + y*y)` with both additions contracted;
        // this is intentionally distinct from a standalone Vector3 norm.
        assert_eq!(
            eigen_norm3_f32(bits(0xbf1e7195), bits(0xbe82723d), bits(0x3f381964)).to_bits(),
            0x3f7b7f59
        );
        assert_eq!(
            eigen_norm3_f32(bits(0xbf219189), bits(0xbd3d9529), bits(0x3f4311e5)).to_bits(),
            0x3f7d9191
        );
    }

    #[test]
    fn m7_first_track_triangulation_matches_pinned_native_bits() {
        let bits = |value: u32| f32::from_bits(value);
        let target_camera_from_candidate = F32RigidPose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                bits(0x3f7ffe35),
                bits(0x3befaa96),
                bits(0x39eadb91),
                bits(0x3a8bfffa),
            )),
            translation: Vector3::new(bits(0x3de1c150), bits(0xb87a2400), bits(0x39ac1888)),
        };
        let candidate_from_target = target_camera_from_candidate.inverse();
        let f0 = Vector3::new(bits(0xbf2a7307), bits(0xbe9af9d0), bits(0x3f2e94d6));
        let f1 = Vector3::new(bits(0xbf2d0c0d), bits(0xbe93697d), bits(0x3f2da937));

        let candidate_rotation = sophus_rotation_matrix(candidate_from_target.rotation);
        let mut candidate_matrix = Matrix4::<f32>::identity();
        candidate_matrix
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&candidate_rotation);
        candidate_matrix
            .fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&candidate_from_target.translation);
        let mut a = Matrix4::<f32>::zeros();
        for column in 0..4 {
            a[(0, column)] = f0.x * if column == 2 { 1.0 } else { 0.0 }
                - f0.z * if column == 0 { 1.0 } else { 0.0 };
            a[(1, column)] = f0.y * if column == 2 { 1.0 } else { 0.0 }
                - f0.z * if column == 1 { 1.0 } else { 0.0 };
            a[(2, column)] = (-f1.z).mul_add(
                candidate_matrix[(0, column)],
                f1.x * candidate_matrix[(2, column)],
            );
            a[(3, column)] = (-f1.z).mul_add(
                candidate_matrix[(1, column)],
                f1.y * candidate_matrix[(2, column)],
            );
        }
        let expected_a = [
            0xbf2e94d6, 0x80000000, 0xbf2a7307, 0x80000000, 0x80000000, 0xbf2e94d6, 0xbe9af9d0,
            0x80000000, 0xbf2dd17a, 0x3c0a2d1b, 0xbf2ce029, 0x3d99bcd8, 0x3a9af4a4, 0xbf2c905f,
            0xbe987a21, 0xb898995a,
        ];
        for row in 0..4 {
            for column in 0..4 {
                assert_eq!(a[(row, column)].to_bits(), expected_a[row * 4 + column]);
            }
        }

        let raw = eigen_jacobi_svd_v_f32(a).unwrap();
        assert_eq!(
            raw.as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![0xbf28a750, 0xbe994a0a, 0x3f2cbdf9, 0x3e147902]
        );
        let actual = triangulate_dlt_matrix_f32(f0, f1, candidate_from_target).unwrap();
        assert_eq!(
            [
                actual.0.x.to_bits(),
                actual.0.y.to_bits(),
                actual.0.z.to_bits(),
                actual.1.to_bits()
            ],
            [0xbf2a746e, 0xbe9aed26, 0x3f2e9645, 0x3e160ef3]
        );
        let projected = StereographicDirection::from_triangulated_f32(actual.0).unwrap();
        assert_eq!((projected.xy.x as f32).to_bits(), 0xbecaaef7);
        assert_eq!((projected.xy.y as f32).to_bits(), 0xbe383810);
    }

    #[test]
    fn m7_second_track_triangulation_matches_pinned_native_bits() {
        let bits = |value: u32| f32::from_bits(value);
        // Raw camera rays and the relative camera transform are copied from
        // the authoritative Eigen/Sophus track-2 probe.  The probe's
        // unprojection is intentionally retained here: BundleAdjustmentBase
        // consumes these non-unit vectors directly.
        let candidate_from_target = F32RigidPose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                bits(0x3f7ffe35),
                bits(0xbbefaa96),
                bits(0xb9eadb91),
                bits(0xba8bfffa),
            )),
            translation: Vector3::new(bits(0xbde1c0f0), bits(0x3997d306), bits(0xb9e136c3)),
        };
        let f0 = Vector3::new(bits(0xbf214708), bits(0xbe84d05b), bits(0x3f3b6451));
        let f1 = Vector3::new(bits(0xbf24c3d6), bits(0xbe79c13a), bits(0x3f39b6ef));

        let candidate_rotation = sophus_rotation_matrix(candidate_from_target.rotation);
        let mut candidate_matrix = Matrix4::<f32>::identity();
        candidate_matrix
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&candidate_rotation);
        candidate_matrix
            .fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&candidate_from_target.translation);
        let mut a = Matrix4::<f32>::zeros();
        for column in 0..4 {
            a[(0, column)] = f0.x * if column == 2 { 1.0 } else { 0.0 }
                - f0.z * if column == 0 { 1.0 } else { 0.0 };
            a[(1, column)] = f0.y * if column == 2 { 1.0 } else { 0.0 }
                - f0.z * if column == 1 { 1.0 } else { 0.0 };
            a[(2, column)] = (-f1.z).mul_add(
                candidate_matrix[(0, column)],
                f1.x * candidate_matrix[(2, column)],
            );
            a[(3, column)] = (-f1.z).mul_add(
                candidate_matrix[(1, column)],
                f1.y * candidate_matrix[(2, column)],
            );
        }
        let expected_a = [
            0xbf3b6451, 0x80000000, 0xbf214708, 0x80000000, 0x80000000, 0xbf3b6451, 0xbe84d05b,
            0x80000000, 0xbf39dd41, 0x3c00c532, 0xbf249574, 0x3da456b3, 0x3aad5b59, 0xbf38c7f1,
            0xbe824c28, 0xb8dcd7b3,
        ];
        for row in 0..4 {
            for column in 0..4 {
                assert_eq!(a[(row, column)].to_bits(), expected_a[row * 4 + column]);
            }
        }

        let raw = eigen_jacobi_svd_v_f32(a).unwrap();
        assert_eq!(
            raw.as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![0xbf1e7195, 0xbe82723d, 0x3f381964, 0x3e3f355b]
        );
        let actual = triangulate_dlt_matrix_f32(f0, f1, candidate_from_target).unwrap();
        assert_eq!(
            [
                actual.0.x.to_bits(),
                actual.0.y.to_bits(),
                actual.0.z.to_bits(),
                actual.1.to_bits(),
            ],
            [0xbf2147c1, 0xbe84c818, 0x3f3b6525, 0x3e42a1b2]
        );
        let projected = StereographicDirection::from_triangulated_f32(actual.0).unwrap();
        assert_eq!((projected.xy.x as f32).to_bits(), 0xbeba3c0f);
        assert_eq!((projected.xy.y as f32).to_bits(), 0xbe195391);
    }

    #[test]
    fn m7_third_track_triangulation_matches_pinned_native_bits() {
        let bits = |value: u32| f32::from_bits(value);
        let candidate_from_target = F32RigidPose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                bits(0x3f7ffe35),
                bits(0xbbefaa96),
                bits(0xb9eadb91),
                bits(0xba8bfffa),
            )),
            translation: Vector3::new(bits(0xbde1c0f0), bits(0x3997d306), bits(0xb9e136c3)),
        };
        let f0 = Vector3::new(bits(0xbf26bda3), bits(0xbdacb6dc), bits(0x3f410c36));
        let f1 = Vector3::new(bits(0xbf2939fc), bits(0xbd90e159), bits(0x3f3f3be4));

        let candidate_rotation = sophus_rotation_matrix(candidate_from_target.rotation);
        let mut candidate_matrix = Matrix4::<f32>::identity();
        candidate_matrix
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&candidate_rotation);
        candidate_matrix
            .fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&candidate_from_target.translation);
        let expected_candidate = [
            0x3f7fffd3, 0x3b0c6cef, 0xba66c162, 0xbde1c0f0, 0xbb0b910f, 0x3f7ff8d7, 0x3c6facec,
            0x3997d306, 0x3a6ef276, 0xbc6fa4e5, 0x3f7ff8f6, 0xb9e136c3, 0x00000000, 0x00000000,
            0x00000000, 0x3f800000,
        ];
        for row in 0..4 {
            for column in 0..4 {
                assert_eq!(
                    candidate_matrix[(row, column)].to_bits(),
                    expected_candidate[row * 4 + column]
                );
            }
        }
        let mut a = Matrix4::<f32>::zeros();
        for column in 0..4 {
            a[(0, column)] = f0.x * if column == 2 { 1.0 } else { 0.0 }
                - f0.z * if column == 0 { 1.0 } else { 0.0 };
            a[(1, column)] = f0.y * if column == 2 { 1.0 } else { 0.0 }
                - f0.z * if column == 1 { 1.0 } else { 0.0 };
            a[(2, column)] = (-f1.z).mul_add(
                candidate_matrix[(0, column)],
                f1.x * candidate_matrix[(2, column)],
            );
            a[(3, column)] = (-f1.z).mul_add(
                candidate_matrix[(1, column)],
                f1.y * candidate_matrix[(2, column)],
            );
        }
        let expected_a = [
            0xbf410c36, 0x80000000, 0xbf26bda3, 0x80000000, 0x80000000, 0xbf410c36, 0xbdacb6dc,
            0x80000000, 0xbf3f633f, 0x3c04309b, 0xbf290a3d, 0x3da938a4, 0x3ac81016, 0xbf3ef2bb,
            0xbda73ea0, 0xb942f6a9,
        ];
        for row in 0..4 {
            for column in 0..4 {
                assert_eq!(a[(row, column)].to_bits(), expected_a[row * 4 + column]);
            }
        }
        let raw = eigen_jacobi_svd_v_f32(a).unwrap();
        assert_eq!(
            raw.as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![0xbf251a77, 0xbdaa9cc6, 0x3f3f26ea, 0x3e0f4631]
        );
        let actual = triangulate_dlt_matrix_f32(f0, f1, candidate_from_target).unwrap();
        assert_eq!(
            [
                actual.0.x.to_bits(),
                actual.0.y.to_bits(),
                actual.0.z.to_bits(),
                actual.1.to_bits(),
            ],
            [0xbf26be5b, 0xbdac4eac, 0x3f410d0d, 0x3e10b291]
        );
        let projected = StereographicDirection::from_triangulated_f32(actual.0).unwrap();
        assert_eq!((projected.xy.x as f32).to_bits(), 0xbebe1e3b);
        assert_eq!((projected.xy.y as f32).to_bits(), 0xbd447636);
    }

    #[test]
    fn m7_fourth_track_triangulation_matches_pinned_native_bits() {
        let bits = |value: u32| f32::from_bits(value);
        let candidate_from_target = F32RigidPose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                bits(0x3f7ffe35),
                bits(0xbbefaa96),
                bits(0xb9eadb91),
                bits(0xba8bfffa),
            )),
            translation: Vector3::new(bits(0xbde1c0f0), bits(0x3997d306), bits(0xb9e136c3)),
        };
        let f0 = Vector3::new(bits(0xbf231e16), bits(0xbd3f802c), bits(0x3f44f0ad));
        let f1 = Vector3::new(bits(0xbf259d03), bits(0xbd0a1b9f), bits(0x3f4305bb));
        let candidate_rotation = sophus_rotation_matrix(candidate_from_target.rotation);
        let mut candidate_matrix = Matrix4::<f32>::identity();
        candidate_matrix
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&candidate_rotation);
        candidate_matrix
            .fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&candidate_from_target.translation);
        let mut a = Matrix4::<f32>::zeros();
        for column in 0..4 {
            a[(0, column)] = f0.x * if column == 2 { 1.0 } else { 0.0 }
                - f0.z * if column == 0 { 1.0 } else { 0.0 };
            a[(1, column)] = f0.y * if column == 2 { 1.0 } else { 0.0 }
                - f0.z * if column == 1 { 1.0 } else { 0.0 };
            a[(2, column)] = (-f1.z).mul_add(
                candidate_matrix[(0, column)],
                f1.x * candidate_matrix[(2, column)],
            );
            a[(3, column)] = (-f1.z).mul_add(
                candidate_matrix[(1, column)],
                f1.y * candidate_matrix[(2, column)],
            );
        }
        let expected_a = [
            0xbf44f0ad, 0x80000000, 0xbf231e16, 0x80000000, 0x80000000, 0xbf44f0ad, 0xbd3f802c,
            0x80000000, 0xbf432c3e, 0x3c0049bb, 0xbf256c82, 0x3dac8cb7, 0x3ad09df9, 0xbf42dff4,
            0xbd37bd58, 0xb958224d,
        ];
        for row in 0..4 {
            for column in 0..4 {
                assert_eq!(a[(row, column)].to_bits(), expected_a[row * 4 + column]);
            }
        }

        let raw = eigen_jacobi_svd_v_f32(a).unwrap();
        assert_eq!(
            raw.as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![0xbf219189, 0xbd3d9529, 0x3f4311e5, 0x3e0ccb9a]
        );

        let actual = triangulate_dlt_matrix_f32(f0, f1, candidate_from_target).unwrap();
        assert_eq!(
            [
                actual.0.x.to_bits(),
                actual.0.y.to_bits(),
                actual.0.z.to_bits(),
                actual.1.to_bits(),
            ],
            [0xbf231e23, 0xbd3f6687, 0x3f44f0bb, 0x3e0e2536]
        );
        let projected = StereographicDirection::from_triangulated_f32(actual.0).unwrap();
        assert_eq!((projected.xy.x as f32).to_bits(), 0xbeb8630d);
        assert_eq!((projected.xy.y as f32).to_bits(), 0xbcd85b88);
    }

    #[test]
    fn triangulation_trace_synthetic_fixture_matches_legacy_bits() {
        let bits = |value: u32| f32::from_bits(value);
        let target_camera_from_candidate = F32RigidPose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                bits(0x3f7ffe35),
                bits(0x3befaa96),
                bits(0x39eadb91),
                bits(0x3a8bfffa),
            )),
            translation: Vector3::new(bits(0x3de1c150), bits(0xb87a2400), bits(0x39ac1888)),
        };
        let candidate_from_target = target_camera_from_candidate.inverse();
        let f0 = Vector3::new(bits(0xbf2a7307), bits(0xbe9af9d0), bits(0x3f2e94d6));
        let f1 = Vector3::new(bits(0xbf2d0c0d), bits(0xbe93697d), bits(0x3f2da937));
        let expected = triangulate_dlt_matrix_f32(f0, f1, candidate_from_target).unwrap();
        let mut trace = TriangulationDltTraceF32::default();
        let traced =
            triangulate_dlt_matrix_f32_with_trace(f0, f1, candidate_from_target, &mut trace)
                .unwrap();
        assert_eq!(
            [
                expected.0.x.to_bits(),
                expected.0.y.to_bits(),
                expected.0.z.to_bits(),
                expected.1.to_bits(),
            ],
            [
                traced.0.x.to_bits(),
                traced.0.y.to_bits(),
                traced.0.z.to_bits(),
                traced.1.to_bits(),
            ]
        );
        assert!(trace.input_valid);
        assert_eq!(
            trace
                .dlt_matrix
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![
                0xbf2e94d6, 0x80000000, 0xbf2a7307, 0x80000000, 0x80000000, 0xbf2e94d6, 0xbe9af9d0,
                0x80000000, 0xbf2dd17a, 0x3c0a2d1b, 0xbf2ce029, 0x3d99bcd8, 0x3a9af4a4, 0xbf2c905f,
                0xbe987a21, 0xb8989959,
            ]
        );
        assert!(trace.scale.is_some());
        assert!(trace.singular_values.is_some());
        assert!(trace.rank_estimate.is_some());
        assert_eq!(trace.homogeneous_raw.unwrap().len(), 4);
        assert_eq!(trace.homogeneous_normalized.unwrap().len(), 4);
        assert_eq!(trace.direction_after_sign.unwrap().len(), 3);
        assert!(trace.rho_after_sign.is_some());
    }

    #[test]
    fn triangulation_native_track621_captured_a_svd_schedule_fixture() {
        // The native attempt-2 A/V/singular lanes are a fixed f32 fixture.
        // Feeding the captured A directly isolates the SVD schedule from the
        // differing estimator pose state observed in the live native/Rust
        // runs; this test never changes the production triangulation path.
        let bits = |value: u32| f32::from_bits(value);
        let a = Matrix4::from_row_slice(&[
            bits(3211266404),
            bits(2147483648),
            bits(3185500296),
            bits(2147483648),
            bits(2147483648),
            bits(3211266404),
            bits(3201367167),
            bits(2147483648),
            bits(3211557697),
            bits(986538904),
            bits(3189133176),
            bits(1037132919),
            bits(999840401),
            bits(3211256627),
            bits(3200984778),
            bits(3169479315),
        ]);
        let mut trace = TriangulationDltTraceF32::default();
        let mut optional_trace = Some(&mut trace);
        let v = eigen_jacobi_svd_v_f32_with_optional_trace(a, &mut optional_trace)
            .expect("captured native A must have a finite SVD");
        assert_eq!(
            v.as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![3184779977, 3200692015, 1063038671, 1050561757]
        );
        assert_eq!(
            trace
                .singular_values
                .expect("SVD trace must include singular values")
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![1068824116, 1067840172, 1033883501, 961512108]
        );
    }

    #[test]
    fn m11_frame49_track2106_native_svd_raw_bits() {
        // Native and Rust DLT inputs/matrix are bit-identical. The pinned
        // Eigen loop returns this pre-normalization V column, so this oracle
        // isolates Jacobi FMA association from the following norm/project.
        let bits = |value: u32| f32::from_bits(value);
        let a = Matrix4::from_row_slice(&[
            bits(0xbf607428),
            bits(0x80000000),
            bits(0xbdee9249),
            bits(0x80000000),
            bits(0x80000000),
            bits(0xbf607428),
            bits(0xbeeee409),
            bits(0x80000000),
            bits(0xbf64bf97),
            bits(0x3b7c5343),
            bits(0xbe1bdde2),
            bits(0x3dcd4c42),
            bits(0xbaa4fc9c),
            bits(0xbf5fe153),
            bits(0xbeea7357),
            bits(0xbcc13efe),
        ]);
        let raw = eigen_jacobi_svd_v_f32(a).expect("captured frame49 DLT matrix is finite");
        assert_eq!(
            raw.map(|value| value.to_bits()),
            Vector4::new(0xbde4082f, 0xbee457e8, 0x3f566348, 0x3e9757f3)
        );
    }

    #[test]
    fn m11_frame49_track2106_native_jacobi_rotation_bits() {
        let pairs = [
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (3, 2),
            (1, 0),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (3, 2),
            (1, 0),
            (2, 0),
            (3, 0),
        ];
        let cases = [
            (
                [0xbe2e6f89, 0xbf800000, 0xbe057f24, 0xbf7b3196],
                [0xbf32c955, 0x3f37399b],
                [0x3f7d17af, 0x3e19e47e],
            ),
            (
                [0x3cd4a0b2, 0xbb4536bb, 0xbf042839, 0xbf7b3196],
                [0x3f7ffda2, 0x3c0b4aba],
                [0x3f628911, 0x3eee79e1],
            ),
            (
                [0xbcd844a5, 0xbda094bc, 0xbda47100, 0x3fb56335],
                [0xbf7fa037, 0xbd5d613b],
                [0x3f7f9b5a, 0xbd62ec5c],
            ),
            (
                [0x3cfbe3c4, 0x3f8cdce3, 0xbb7153a8, 0xbf8dec86],
                [0x3f35a9ac, 0x3f345fa2],
                [0x3f7ff816, 0x3c7e96df],
            ),
            (
                [0x3c9d82bb, 0xbbb4febc, 0xbda04311, 0x3cc7a4fa],
                [0xbe72c331, 0x3f78b3b2],
                [0x3f7483fe, 0x3e97a201],
            ),
            (
                [0xbfc7fc68, 0xbd63d7e0, 0xbd68c2b7, 0xbfb5f1be],
                [0x3f719bbb, 0x3ea94013],
                [0x3f71a472, 0xbea90e4d],
            ),
            (
                [0xb9e1e80b, 0x3d4bb178, 0xbb1d6655, 0xbfb36cd8],
                [0x3f7fd6cd, 0x3d1137fb],
                [0x3f7fffe8, 0x3adedd26],
            ),
            (
                [0xba07343d, 0x3c90eabf, 0x3b9a4ebf, 0xbfca8150],
                [0x3f7ffbe7, 0x3c3733af],
                [0x3f7fffb6, 0xbb4348b5],
            ),
            (
                [0xbdacc798, 0x3c840bc4, 0x3ad3d9b0, 0xbfb389cf],
                [0x3f7ffb97, 0x3c3e176c],
                [0x3f7fffe4, 0xbaf27a4b],
            ),
            (
                [0xbdacbadd, 0x3b8f805e, 0xba976489, 0xbfca84c8],
                [0x3f7fffc3, 0x3b335b22],
                [0x3f7ffffe, 0x3a1920ba],
            ),
            (
                [0xbdacbc5f, 0xb770dcec, 0x383a584d, 0xb9f2c911],
                [0xbf7ffffe, 0x3a09d7fd],
                [0x3f800000, 0xb92f74bd],
            ),
            (
                [0xbfca8500, 0x3729b5a7, 0xba31d7e7, 0xbfb38cec],
                [0x3f7fffdf, 0x3b01983b],
                [0x3f7fffe6, 0xbae4f472],
            ),
            (
                [0x39f2ca12, 0xb1bc206d, 0xb5e57da3, 0xbfb38ce7],
                [0x3f800000, 0xb1709200],
                [0x3f800000, 0x35a39a18],
            ),
            (
                [0x3dacbc61, 0xb2dd38a2, 0x34d4c6c0, 0xbfb38ce7],
                [0xbf800000, 0x33185e78],
                [0x3f800000, 0xb498d4ec],
            ),
        ];
        for (index, (m, expected_left, expected_right)) in cases.into_iter().enumerate() {
            let (left, right) = real_2x2_jacobi_svd_f32(
                f32::from_bits(m[0]),
                f32::from_bits(m[1]),
                f32::from_bits(m[2]),
                f32::from_bits(m[3]),
            );
            assert_eq!(
                [left.0.to_bits(), left.1.to_bits()],
                expected_left,
                "left case {index}"
            );
            assert_eq!(
                [right.0.to_bits(), right.1.to_bits()],
                expected_right,
                "right case {index}"
            );
        }

        let b = |value: u32| f32::from_bits(value);
        let matrix = Matrix4::from_row_slice(&[
            b(0xbf607428),
            b(0x80000000),
            b(0xbdee9249),
            b(0x80000000),
            b(0x80000000),
            b(0xbf607428),
            b(0xbeeee409),
            b(0x80000000),
            b(0xbf64bf97),
            b(0x3b7c5343),
            b(0xbe1bdde2),
            b(0x3dcd4c42),
            b(0xbaa4fc9c),
            b(0xbf5fe153),
            b(0xbeea7357),
            b(0xbcc13efe),
        ]);
        let scale = matrix
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        let mut work = matrix.map(|value| value / scale);
        let mut v = Matrix4::<f32>::identity();
        let mut max_diag = (0..4).map(|i| work[(i, i)].abs()).fold(0.0_f32, f32::max);
        let mut index = 0;
        loop {
            let mut finished = true;
            for p in 1..4 {
                for q in 0..p {
                    let threshold = f32::MIN_POSITIVE.max((2.0_f32 * f32::EPSILON) * max_diag);
                    if work[(p, q)].abs() > threshold || work[(q, p)].abs() > threshold {
                        finished = false;
                        assert_eq!((p, q), pairs[index], "pair case {index}");
                        let actual = [
                            work[(p, p)].to_bits(),
                            work[(p, q)].to_bits(),
                            work[(q, p)].to_bits(),
                            work[(q, q)].to_bits(),
                        ];
                        assert_eq!(actual, cases[index].0, "input case {index}");
                        let (left, right) = real_2x2_jacobi_svd_f32(
                            work[(p, p)],
                            work[(p, q)],
                            work[(q, p)],
                            work[(q, q)],
                        );
                        apply_left_f32(&mut work, p, q, left.0, left.1);
                        apply_right_f32(&mut work, p, q, right.0, right.1);
                        apply_right_f32(&mut v, p, q, right.0, right.1);
                        max_diag = max_diag.max(work[(p, p)].abs()).max(work[(q, q)].abs());
                        index += 1;
                    }
                }
            }
            if finished {
                break;
            }
        }
        assert_eq!(index, cases.len());
    }

    #[test]
    fn keyframe_gate_matches_basalt_limits_and_ratio() {
        let c = KeyframeConfig::default();
        assert!(should_insert_keyframe(10, Some(0), 6, 10, 2, 2, c));
        assert!(!should_insert_keyframe(4, Some(0), 0, 10, 2, 2, c));
        assert!(!should_insert_keyframe(10, Some(0), 0, 10, 3, 2, c));
        assert!(!should_insert_keyframe(10, Some(0), 0, 10, 2, 7, c));
    }
}
