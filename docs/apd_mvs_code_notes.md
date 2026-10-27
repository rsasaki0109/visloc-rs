# APD-MVS (CVPR 2023): implementation-level summary

Notes from reading the authors' released code (MIT licence, Y. Wang et al., "Adaptive Patch Deformation for
Textureless-Resilient Multi-View Stereo", CVPR 2023), taken while porting its deformable patches to
visloc-mvs ([`mvs.md`](mvs.md#tried-and-not-adopted-apd-mvss-deformable-patches)). That port alone did not pay;
the parts of APD's pipeline it left out are listed in §7, and the plan to try them is in
[`next_development_plan.md`](next_development_plan.md#on-hold-gpu-work-eth3d-mvs).

Source files: `{APD.cu, APD.cpp, APD.h, main.cpp, main.h}` of that code (and `colmap2mvsnet.py` for the data
prep). `file:line` references point at those files. Everything below describes what the code does. Where the
code has quirks or bugs, they are flagged, together with what was probably intended.

Conventions used throughout the code:

- A plane hypothesis is `float4 (nx, ny, nz, w)`. **During PatchMatch** it is in ref-camera coordinates with
  `n·X + w = 0`, so `w = -n·X(p, depth)` (`GetDistance2Origin`, APD.cu:187). Depth comes from
  `ComputeDepthfromPlaneHypothesis` (APD.cu:206):
  `d = -w*fx / ((x-cx)*nx + (fx/fy)*(y-cy)*ny + fx*nz)`.
- **On disk / between passes** the same buffer holds `(world normal, depth)`. `GetDepthandNormal` (APD.cu:1587)
  converts from the PatchMatch form to this one, and `RandomInitialization` (APD.cu:826-834) converts back.
- Pixel states (main.h:69): `WEAK=0, STRONG=1, UNKNOWN=2`.
- Images are grayscale float in 0..255. Textures use linear filtering, `cudaAddressModeWrap`, and +0.5 offsets
  (APD.cpp:598-600). Wrap addressing means out-of-image samples wrap around; a reimplementation should clamp.
- `NEIGHBOUR_NUM = 9` (main.h:38): the center plus 8 anchors. `MAX_SEARCH_RADIUS = 4096`.
- Parameters (main.h:75-94): `strong_radius=5, strong_increment=2, weak_radius=5, weak_increment=5,
  top_k=4, geom_factor=0.2, max_iterations=3`. `sigma_spatial` and `sigma_color` are **never used**.

---

## 1. Overall pipeline

### 1.1 Scales and passes (main.cpp:140-233)

`ComputeRoundNum` (main.cpp:72-88) starts with `round_num = 1` and computes
`while (max(W,H) > 1000) { max /= 2; round_num++; }`. Round `i` runs at `scale_size = 2^(round_num-1-i)`, so the
coarsest scale comes first.

- The README converts ETH3D at half resolution (`--scale_factor 2`, giving 3024x2016; colmap2mvsnet resizes
  with INTER_NEAREST).
- At that resolution `round_num = 3` and the scales are 1/4 (756x504), 1/2 (1512x1008) and 1/1 (3024x2016).

`InuputInitialization` downsamples each image with `cv::resize(INTER_LINEAR)`, without a Gaussian pre-blur, and
scales `K` to match (APD.cpp:464-488). The depth range is widened to `[0.6*dmin_cam, 1.2*dmax_cam]`
(APD.cpp:454-455). colmap2mvsnet already sets `dmin_cam` to the 1st percentile of sparse depths x0.75 and
`dmax_cam` to the 99th percentile x1.25.

Source views: pair.txt lists up to **20** source views (`num_view = min(20, N-1)`, colmap2mvsnet.py:419). The
score is the number of shared sparse points, set to 0 if the 75th-percentile triangulation angle is below 1
degree. APD keeps every view with score > 0 (main.cpp:42). This is **twice the 10 views ACMM/ACMMP normally
use**, and it applies to both PatchMatch and fusion.

For each round `i`, all images are processed sequentially, 4 times. Each time is a "pass", and each pass is
one full `ProcessProblem` / `RunPatchMatch`:

| pass | `state` | `use_APD` | `geom_consistency` | `weak_peak_radius` | notes |
|---|---|---|---|---|---|
| init (main.cpp:170-185) | i==0: `FIRST_INIT`, else `REFINE_INIT` | `i>0` | false | 6 | i>0: `ransac_threshold = 0.01 - 0.00125*i`, `rotate_time = min(2^i, 4)` |
| iter j=0,1,2 (main.cpp:192-215) | `REFINE_ITER` | `i>0` | **true** | `max(4-2j, 2)`, i.e. 4, 2, 2 | same ransac/rotate settings |

With 3 rounds, round 1 uses `rotate_time=2` and `ransac_threshold=0.00875`, and round 2 uses `rotate_time=4`
and `ransac_threshold=0.0075`.

- **Round 0 never uses APD.** With `use_APD=false`, every pixel is set to STRONG at load (APD.cpp:540-548). Round
  0 only produces the first weak/strong map, which `DepthToWeak` writes into `weak.bin`.
- Passes communicate through files in each image's result folder: `depths.dmb`, `normals.dmb` (world normals),
  `weak.bin` (the state map) and `selected_views.bin` (a per-pixel u32 view bitmask). At a new scale, all four
  are upsampled to the new size with **nearest neighbour** (`RescaleMatToTargetSize`, APD.cpp:752-774; its
  row/col scale factors are swapped, which is harmless for isotropic 2x).
- **There is no joint bilateral upsampling and no ACMP-style triangulation prior.**
- Geometric consistency reads the other images' `depths.dmb`. Images are processed in order, so earlier images
  already hold this pass's result (Gauss-Seidel style, as in ACMM).
- After each pass, `ProcessProblem` (main.cpp:105-115) sets depth to 0 and the state to **UNKNOWN** wherever
  depth falls outside `[depth_min, depth_max]`.

### 1.2 One pass: `APD::RunPatchMatch` (APD.cu:2386-2495)

```
InitRandomStates
FindNearestStrongPoint        // only for WEAK pixels: nearest STRONG pixel in a 201x201 window
GenNeighbours                 // only for WEAK pixels: anchor search + RANSAC; sets weak_reliable
NeigbourUpdate                // WEAK pixels with weak_reliable != 1 become UNKNOWN
RandomInitialization          // FIRST_INIT: random plane + top-k init cost; otherwise load plane, recompute cost
for it in 0..max_iterations(=3):
    BlackPixelUpdateStrong(it); RedPixelUpdateStrong(it)   // every pixel with state != WEAK (STRONG and UNKNOWN)
    RANSACToGetFitPlane                                    // WEAK: fit plane from current anchor depths
    BlackPixelUpdateWeak(it);   RedPixelUpdateWeak(it)     // only WEAK pixels
GetDepthandNormal             // plane becomes (world normal, depth)
BlackPixelFilterStrong; RedPixelFilterStrong   // median depth filter on non-WEAK pixels, STRONG neighbours only
DepthToWeak                   // reclassify every pixel as WEAK / STRONG / UNKNOWN
LocalRefine                   // 11-step 1-D depth search for every pixel
```

- The checkerboard colouring is standard: black is `(x+y)` even and red is `(x+y)` odd (APD.cu:1510-1585).
- `GenNeighbours` runs **before** `RandomInitialization`, while the plane buffer still holds
  `(world normal, depth)` from disk. That is why it can use `.w` directly as a depth (APD.cu:1866, 1874).
- Anchors are computed **once per pass**. The fitted plane is recomputed **every iteration**, after the strong
  sweep.

### 1.3 Initial cost

- `FIRST_INIT`: `ComputeMultiViewInitialCostandSelectedViews` (APD.cu:616-662). It computes the standard NCC for
  all views, sorts them, and sets `top_k = min(#views with cost < 2, 4)`. The cost is the mean of the best
  top_k. The selected-view bitmask is every view whose cost is <= the k-th best.
- Otherwise: `ComputeMultiViewInitialCost` (APD.cu:664-693). It takes the mean standard NCC over the loaded
  selected views and unsets any view whose cost is 2.
  - Quirk: `unSetBit` (APD.cu:47-50) computes `x &= 0xFFFFFFFE << n`, which clears bits **0..n**, not only
    bit n.

### 1.4 "Strong" and "weak" processing

| Pixel state | Kernels | Cost function | Geometric consistency |
|---|---|---|---|
| STRONG or UNKNOWN | `*UpdateStrong`: ACMH/ACMM-style propagation | standard NCC (`ComputeBilateralNCCOld`) | **never added** |
| WEAK | `*UpdateWeak`: APD propagation | deformable NCC (`ComputeBilateralNCCNew`) | added in REFINE_ITER passes |

- Geometric consistency enters the strong path only through `DepthToWeak` and `LocalRefine`. In round 0 every
  pixel is STRONG, so the geometric passes there differ from the photometric one only through those two
  kernels.
- This is a real departure from ACMM/ACMMP, which add the geometric term to every pixel in geometric passes.

---

## 2. Weak/strong classification: `DepthToWeak` (APD.cu:1990-2144)

It runs at the end of every pass on **every** pixel, using the final filtered depth and normal. Its result is
the state map for the next pass.

```
if p within 6 px of the border           -> UNKNOWN
plane = (R_ref * n_world, depth)          // normal back in ref-camera coordinates
if depth == 0                             -> UNKNOWN
S = { src views v with bit v set in selected_views[p] }
if S empty                                -> UNKNOWN
B = mean_{v in S} |C_ref - C_v|           // mean baseline
disp = fx * B / depth                     // pseudo-disparity
for k in -30..30:                         // 61 samples, 1-"pixel" disparity steps (uniform in inverse depth)
    d_k = fx * B / (disp + k)
    if d_k outside [depth_min, depth_max]: c[k] = 2; continue
    plane_k = same normal, w = -n·X(p, d_k)
    c[k] = min(2, Σ_{v∈S} vw[v] * (NCC_old(p,v,plane_k) + (geom ? 0.2*Geom(p,v,plane_k) : 0)) / Σ_{v∈S} vw[v])
```

Here `vw` is the per-pixel Monte-Carlo view weight (uchar count out of 15) from the last propagation iteration,
stored in `view_weight_cuda`.

Next it finds strict local minima (the code calls them "peaks") at indices i = 2..58:
`c[i-1] > c[i] && c[i+1] > c[i]`. `min_peak` is the lowest one. If there is none, `min_peak = 0` and
`min_cost = 2`.

```
if |min_peak - 30| > weak_peak_radius  or  c[min_peak] > 0.5     -> WEAK
if peak_count == 1:  c[min_peak] <= 0.15 ? STRONG : WEAK
else:
    spread = sqrt( Σ_{other peaks} (c_i - c_min)^2 ) / (peak_count - 1)
    spread > 0.2 ? STRONG : WEAK
```

What this means:

- A pixel is STRONG when its cost curve along the ray, at fixed normal, has one clear minimum close to the
  current estimate. With multiple minima, the others must sit well above the best one.
- `weak_peak_radius` is 6 in the init pass and 4, 2, 2 in the geometric passes. It is measured in disparity
  steps at the current scale.
- In geometric passes the curve includes `0.2*min(3, reprojection error)`, which sharpens the minimum for pixels
  that agree across views. Such pixels are easier to call STRONG.
- `cost_now`, computed at APD.cu:2022-2051, is not used for the decision.

Other ways a pixel becomes UNKNOWN:

- Out-of-range depth (main.cpp:109-112).
- A WEAK pixel whose anchor search or RANSAC fails (`NeigbourUpdate`, APD.cu:1971-1987, at the start of the next
  pass).

UNKNOWN pixels go through the **strong** kernels with the standard NCC. They are never used as anchors and never
used as median-filter neighbours.

---

## 3. The deformable patch

### 3.1 `FindNearestStrongPoint` (APD.cu:2234-2270)

For each WEAK pixel, it brute-forces the 201x201 window (`radius = 100`, "ETH 100") for the STRONG pixel with
the smallest Euclidean distance. Ties go to the first one found in x-major loop order. The result goes into
`weak_nearest_strong[p]` as short2, or (-1,-1) if none is found. Non-WEAK pixels get (-1,-1).

For a GPU port, a jump-flooding pass gives the same map much more cheaply. This is a suggestion, not what the
code does.

### 3.2 Anchor search and RANSAC filtering: `GenNeighbours` (APD.cu:1750-1969)

Storage:

- `neighbours_map[p]` is the row-major index of p among WEAK pixels (APD.cpp:526-537).
- `neighbours[idx*9 + 0..8]` are short2 values. Slot 0 is the pixel itself (APD.cu:1781). Slots 1..8 are anchors,
  with (-1,-1) for an empty slot.

**Direction search** (APD.cu:1788-1853):

```
angle      = 45° / rotate_time           // rotate_time = 2 at round 1, 4 at rounds >= 2
thresh     = cos(angle/2)
shift_rng  = max(int(tan(angle/2) * 20), 1)   // 3 for rotate_time=2, 1 for rotate_time=4
for each base dir in the 8-neighbourhood {(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)} (normalised):
  dir = base
  for r_it in 0..rotate_time:                      // 8*rotate_time sectors in total (16 or 32), evenly covering 360°
    for radius = 2; radius <= 4096; radius = min(2*radius, radius+25):   // 2,4,8,16,32,57,82,107,...
        if p + dir*radius is outside the image: break
        repeat 4 times:
            jdir = normalize(dir*20 + (sx, sy))    // random jitter, sx,sy in [0, shift_rng)  (see quirk)
            q = (short)(p + jdir*radius)
            if q within 6 px of the border: continue
            if state[q] != STRONG:
                q = weak_nearest_strong[q]         // (-1,-1) if q was UNKNOWN or nothing nearby
                if q == (-1,-1): continue
            if dot(normalize(q - p), dir) > thresh: record q as this sector's anchor; stop this sector
    dir = rotate(dir, angle)
if fewer than 4 sectors found an anchor (strong_point_size <= 3): weak_reliable = 0; return
```

So each sector takes the first STRONG pixel met along its ray, sampled sparsely at radii 2, 4, 8, 16, 32, 57,
and so on. When a sample lands on a WEAK pixel, the sector instead takes that sample's nearest STRONG pixel,
provided it falls inside the sector's angular window. There is no limit on distance, so anchors can be hundreds
of pixels away.

Quirk: `(curand%2==0 ? 1 : -1) * curand % shift_range` is evaluated in **unsigned** arithmetic, so the jitter
is always in `[0, shift_rng)` and never negative. With `rotate_time=4` the jitter is always 0.

**RANSAC coplanarity filter** (APD.cu:1859-1968):

1. Back-project every found anchor to a 3D point in the ref camera, using its current depth from the previous
   pass. The center's own 3D point `Xc` is built the same way.
2. Run 50 iterations. Each draws 3 random indices. Duplicates are skipped, and a skip still uses up an
   iteration.
3. Keep a sample only if `PointinTriangle(A,B,C,p)` holds (APD.cu:91-112): every side is longer than 2 px, the
   triangle is non-degenerate, and p lies inside or on an edge.
4. Fit the plane `n = normalize((A-C)x(B-C))`, `w = -n·A`.
5. Count inliers: `|n·X_i + w| / (depth_max - depth_min) < ransac_threshold`, normalised by the depth range.
6. Reject the plane if it has fewer than 6 inliers.
7. Keep the plane with the most inliers. Ties go to the smaller `|n·Xc + w|`, the current center depth's
   distance to the plane.
8. If no plane is found: `weak_reliable = 0`, and the pixel becomes UNKNOWN.

**Anchor ordering**:

- Outliers are removed.
- Each inlier is weighted by its plane distance. The 3 sample points get `distance - 1`, which puts them first,
  so the triangle around p always survives.
- The inliers are sorted ascending and `neighbours[1..8]` takes the first 8. Any spare slots stay (-1,-1).
- Set `weak_reliable = 1`.

Result: up to 8 anchors. All are STRONG, all are nearly coplanar, and their convex hull contains p.

### 3.3 Matching cost

**Standard NCC: `ComputeBilateralNCCOld`** (APD.cu:530-614)

```
H = K_s (R_rel - t_rel n^T / w) K_r^{-1}         // ComputeHomography (APD.cu:303-363)
if H*p is outside src: return 2
samples: (p.x+i, p.y+j) for i,j in {-5,-3,-1,1,3,5}   // 11x11 window, stride 2, 36 samples, center not sampled
weights: ALL 1.0 (no bilateral weighting, despite the name)
cost = (var_ref < 1e-5 || var_src < 1e-5) ? 2 : clamp(1 - cov/sqrt(var_ref*var_src), 0, 2)
```

**Deformable NCC: `ComputeBilateralNCCNew`** (APD.cu:400-528), used only for WEAK pixels (it prints "error"
otherwise):

```
H = homography of the CENTER pixel's plane hypothesis (one H for every sub-patch)
if H*p is outside src: return 2
center_cost = 0; anchor_sum = 0; anchor_cnt = 0
for k in 0..8:
    a = neighbours[k]; if a == (-1,-1): continue          // k = 0 is p itself
    if H*a is outside the image:
        if k == 0: return 2
        if bit (src-1) is set in selected_views[a]:        // the anchor itself thinks this view sees it
            anchor_sum += 2; anchor_cnt++                 // penalty
        continue
    (rad, inc) = k==0 ? (5, 2) : (5, 5)
    // k==0: offsets {-5,-3,-1,1,3,5}^2 = 36 samples, the same as the standard NCC
    // k>0 : offsets {-5,0,5}^2         = 9 samples, a sparse 3x3 grid spaced 5 px apart, centred on the anchor
    for each offset (i,j): r = I_ref(a + (i,j)); s = I_src(H * (a + (i,j)))   // uniform weights
    c_k = NCC cost as above (2 if var < 1e-5)
    if k == 0: center_cost = c_k else { anchor_sum += c_k; anchor_cnt++ }
cost = anchor_cnt == 0 ? center_cost
                       : 0.25 * center_cost + 0.75 * min(anchor_sum/anchor_cnt, 2)
```

How the hypothesis reaches the anchors:

- Each anchor's sub-patch is warped with the **center pixel's** plane homography, i.e. the code assumes the anchor
  lies on the center's plane.
- A plane hypothesis therefore scores well only if extending it to the textured anchor locations makes those
  sub-patches match.
- The anchors' own planes play no part in the warp.
- This is per-view NCC averaging: 1 center sub-patch plus up to 8 anchor sub-patches, with no bilateral or
  distance weights.

Cost per view per hypothesis is 36 + 8x9 = 108 samples, against 36 for the standard NCC.

---

## 4. Strong vs. weak propagation and refinement

### 4.1 `CheckerboardPropagationStrong` (APD.cu:982-1321): ACMH/ACMM unchanged except for state gating

**Candidates.** The 8 regions use costs from the shared `costs` buffer and take the minimum-cost pixel in each:

- 4 straight "far" strips at distance 3 + 2i for i = 0..10, i.e. 3..23 (11 samples, APD.cu:1021-1095).
- 4 "near" V shapes (APD.cu:1097-1199): the direct neighbour plus the pairs (±(1+i), -(2+i)) for i = 0..2,
  shown here for up (7 samples).
- Candidates may be pixels in any state. Weak pixels store a standard-NCC cost (§4.2, step 8), so all costs are
  on the same scale.

**Per-view costs.** `cost_array[8][nsrc]` uses the standard NCC for every view.

**View priors** (APD.cu:1210-1222): over the 4 direct neighbours whose near flag is valid, add 0.9 if the view's
bit is set in that neighbour's selected_views, else 0.1.

**View selection** (APD.cu:1224-1259), the same as ACMM:

- `tau = 0.8*exp(-it²/90)`.
- For each view, over the 8 candidates: `good = #(c < tau)`, `w = Σ_good exp(-c²/0.18)`, `bad = #(c > 1.2)`.
- `prob = (good > 2 && bad < 3) ? w/good : (bad < 3 ? exp(-tau²/0.32) : 0)`.
- Multiply `prob` by the prior, build the CDF, and draw **15** samples. `view_weights[v]` is the number of times
  view v was drawn (uchar).

**Choosing a candidate:**

- `final_cost[i] = Σ_v vw[v]*c[i][v] / Σ vw`. Take the argmin; `FindMinCostIndex` uses `<=`, so ties go to the
  last index.
- The current plane is re-costed with the new weights, and `costs[p]` is overwritten.
- The candidate is accepted if its depth is in range and its cost is lower. **Only then** does
  `selected_views[p]` become the new bitmask.

**Refinement.** `PlaneHypothesisRefinementStrong` (APD.cu:837-890) tries 5 hypotheses, each paired as
(depth, normal):

| depth | normal |
|---|---|
| d_rand | n_cur |
| d_cur | n_rand |
| d_rand | n_rand |
| d_cur | n_pert |
| d_pert | n_cur |

- `d_pert` is uniform in d·[0.98, 1.02].
- `n_pert` rotates the normal by Euler angles, each uniform in ±0.01π (about ±1.8°).
- The perturbation does not decay.
- Each hypothesis is costed with the standard NCC under the same view weights, and accepted if it is cheaper and
  its depth is in range.

**Write-back** (APD.cu:1311-1320). In `REFINE_INIT`, the first pass at a new scale, the result is written only if
`cost_new < cost_cur - 0.1`. This hysteresis keeps the upsampled coarse estimate unless something clearly beats
it. In other passes the result is always written.

**Quirk.** `float cost_array[8][32] = {2.0f}` sets only `[0][0]` to 2; every other entry is **0**.

- Invalid candidates near the border therefore have cost 0. They count as "good" in view selection and usually
  win the argmin.
- When they win, the `flag[min_idx]` check fails and no propagation happens.
- ACMM has the same quirk. The intended behaviour is cost 2 for missing candidates.

### 4.2 `CheckerboardPropagationWeak` (APD.cu:1323-1508)

Differences from the strong version:

1. **Candidates** (APD.cu:1352-1363): exactly the 8 anchors `neighbours[1..8]`, used only if they are STRONG.
   - Each candidate hypothesis is the anchor's current plane, in global ref-camera plane form, so it can be
     propagated over a long distance.
   - **There is no propagation from adjacent pixels:** a WEAK pixel never sees its immediate neighbours'
     planes.
   - Per-view costs use the **deformable** NCC.
2. **View priors** (APD.cu:1371-1384) come from all 8 anchors' selected_views (0.9 / 0.1), not from the 4 direct
   neighbours.
3. **View selection** uses the same thresholds and the same 15 samples.
   - The zero-init quirk matters more here: any pixel with fewer than 8 anchors has all-zero rows.
   - Without geometric consistency (the init pass), those rows cost 0 and block anchor propagation for that pixel
     entirely.
   - With geometric consistency they cost `0.2*3 = 0.6` (APD.cu:1446).
   - Treat missing anchors as cost 2 in a reimplementation.
4. **Geometric consistency** applies in REFINE_ITER passes:
   `final_cost[i] = Σ_v vw[v]*(c[i][v] + 0.2*Geom(p, v, plane_i)) / Σ vw` (APD.cu:1436-1456). The current plane is
   costed the same way (APD.cu:1460-1473).
   - `ComputeGeomConsistencyCost` (APD.cu:752-789): project p at the hypothesis depth into src and read the src
     depth at the truncated integer pixel. If that depth is 0, return 3.
   - Otherwise back-project at the float src location, project into ref, and return
     `min(3, |p - p_back|)` in pixels.
   - There is no bounds check (the texture wraps).
5. Acceptance is the same as the strong version, including the update of `selected_views`.
6. **Refinement** uses `PlaneHypothesisRefinementWeak` (APD.cu:892-980):
   - First it tests `fit_plane[p]` from `RANSACToGetFitPlane` with the deformable NCC (+ geom), and accepts it
     if cheaper and its depth is in range.
   - If `fit_plane` has a zero normal (RANSAC failed), the function **returns right away**, skipping the random
     refinement.
   - Otherwise it runs the same 5 random/perturbed hypotheses as the strong version, with the deformable NCC
     (+ geom).
7. Same `REFINE_INIT` `-0.1` hysteresis.
8. **Last step** (APD.cu:1499-1507): `costs[p]` is recomputed with the **standard** NCC under the current view
   weights. The shared `costs` buffer, which strong pixels use to pick their minimum-cost samples, therefore
   always holds standard-NCC costs.

---

## 5. `RANSACToGetFitPlane` and `LocalRefine`

### 5.1 `RANSACToGetFitPlane` (APD.cu:2272-2384): runs every iteration after the strong sweep

```
non-WEAK p:  fit_plane[p] = plane[p]
WEAK p:
  anchors 1..8 (valid only) -> X_i = backproject(a_i, depth_from_plane(plane[a_i], a_i))   // CURRENT strong depths
  if count < 3: fit_plane[p] = plane[p]; return
  50 iterations: 3 distinct random anchors whose 2-D triangle contains p (PointinTriangle)
      n = normalize((A-C)x(B-C)), w = -n·A
      score = Σ_{other anchors} |n·X_i + w|        // L1, unnormalised, no inlier threshold
      keep the minimum score; stop early if it reaches 0
  found:     flip (n, w) so that n·viewdir(p) <= 0; fit_plane[p] = (n, w)   // ref-camera plane, usable as a hypothesis directly
  not found: fit_plane[p] = 0
```

Use: it is the first extra candidate in `PlaneHypothesisRefinementWeak`. In effect it is a local planar prior
built from coplanar, reliable anchors around p, and it is refit every iteration as the strong depths improve.
It differs from ACMP/ACMMP's global Delaunay prior: it is per pixel, it comes from coplanarity-checked anchors
that enclose p, and it is only a **candidate**, judged by the deformable photometric cost (+ geom). It is never
added as a prior term in the cost.

### 5.2 `LocalRefine` (APD.cu:2146-2232): runs once per pass after `DepthToWeak`, on every pixel

```
plane = (R_ref * n_world, depth); if depth == 0: return
S, B, disp as in DepthToWeak; vw = view weights
cost_now = Σ_{v∈S} vw[v]*(NCC_old + 0.2*Geom [if geom]) / Σ vw
for k in -5..5: d_k = fx*B/(disp + k); skip if out of range
    cost_k with the same normal and the same formula; keep the minimum (d*, c*)
if cost_now - c* > 0.1: depth[p] = d*        // depth only; the normal is unchanged
```

This is a 1-D search along the ray over ±5 disparity steps, accepted only if it improves the cost by more than
0.1. It runs after classification, so `DepthToWeak` sees the depth from before this refinement.

### 5.3 Median filter: `CheckerboardFilterStrong` (APD.cu:1604-1714)

- Applies to non-WEAK pixels with `cost >= 0.001`, which in practice is nearly all of them.
- Takes the median of the center depth plus up to 20 neighbours, counting only STRONG ones. Offsets are
  (0,±1), (0,±3), (0,±5), (±1,0), (±3,0), (±5,0), (±2,±1) and (±1,±2), the ACMM pattern.
- With an even count it averages the two middle values.
- It filters depth only and runs as black then red.

---

## 6. Fusion: `RunFusion`, ETH version (APD.cpp:826-977)

```
for each ref image (in order), each pixel with depth > 0 and not yet masked:
  X = backproject(ref pixel)
  for each src in pair.txt (up to 20, score > 0):
     project X -> nearest src pixel (src_r, src_c); skip if out of image, masked, or depth <= 0
     reproject the src point into ref:
        e   = reprojection error (px)
        rd  = |d_proj - d_ref| / d_ref
        ang = angle(n_ref, n_src)                     // world normals
     if e < 2.0 && rd < 0.01 && ang < 0.174533 (10°):
        used[j] = (src_c, src_r); num++; score += exp(-(e + 200*rd + 10*ang))
  thr = (state[ref pixel] == WEAK) ? 0.45 : 0.30
  if num >= 1 && score > thr * num:                    // the mean per-view exp score must exceed thr
     emit point X (the ref point, not averaged); color = mean of ref and used src colors
     mask every used src pixel so it is not fused again
```

Key differences from ACMM/ACMMP fusion:

- **Only 1 consistent view is required** (ACMM requires 2 or more). Precision is protected by a mean "dynamic"
  consistency score.
- WEAK pixels need a stricter score: mean `exp(-x) > 0.45` means `x < 0.80`, against `x < 1.20` for other pixels.
- There are up to 20 candidate views.
- An optional `blocks/mask_<id>.jpg` image mask is supported.

`RunFusion_TAT_*` (APD.cpp:979-1296) are unused alternatives.

---

## 7. What most plausibly explains the gains on low-texture ETH3D (office, playground, facade, kicker)

1. **The deformable cost for WEAK pixels (§3.3).** A textureless center patch carries almost no information. Up
   to 8 sparse 3x3 sub-patches at textured, coplanar locations, warped by the center's plane, turn the hypothesis
   test into "does this plane extended to nearby textured points match?", which is discriminative for both depth
   and normal. The anchor weight of 0.75 dominates.
2. **Long-range plane candidates instead of local propagation inside weak regions (§4.2).** WEAK pixels draw
   candidates only from anchor planes, often tens to hundreds of pixels away, plus a RANSAC-fitted local plane.
   Local propagation of bad textureless hypotheses is cut off. Large walls and floors are filled from their
   reliable borders in a single step.
3. **Coplanarity-verified anchors (§3.2).** RANSAC with a depth-range-normalised threshold (0.875% then 0.75% of
   the range), at least 6 inliers, and p inside the sample triangle. This stops anchors on a different surface
   (across a wall/floor edge, or on foreground objects) from steering the cost. Pixels where this cannot be done
   fall back to UNKNOWN and the standard path.
4. **Ambiguity-based reliability (§2) instead of a texture or variance test.** Pixels are classified from the
   shape of their own multi-view cost curve, including the geometric term. The map is recomputed after every
   pass and tightened as `weak_peak_radius` goes 6, 4, 2, 2. It refreshes which pixels may serve as anchors as
   the depth improves.
5. **Coarse-to-fine with hysteresis.** Round 0 at 1/4 scale is plain ACMH, where textureless areas are smaller in
   pixels. Upsampled estimates are kept unless a new hypothesis beats them by 0.1 (`REFINE_INIT`). Each scale
   then gets 3 geometric passes of 3 iterations each.
6. **Smaller items:**
   - `LocalRefine`: disparity-step depth snapping, accepted on a 0.1 gain.
   - A median filter that uses only STRONG neighbours.
   - Geometric consistency on WEAK pixels, with weight 0.2 and a cap of 3 px.
   - Up to 20 source views.
   - The fusion rule: 1 or more consistent views plus a dynamic score, stricter for WEAK pixels. Against ACMMP's
     "2 or more views", this alone can move completeness noticeably on ETH3D.
7. **Things APD does not do that ACMMP does:**
   - No triangulation planar prior.
   - No joint bilateral upsampling (nearest neighbour only).
   - No bilateral NCC weights (uniform weights everywhere).
   - No geometric term on STRONG pixels.

   So in a codebase that already has ACMMP, items 1-4 are the parts to add, as an extra state-gated path. They do
   not replace the existing strong path.

### Notes on code vs. paper, and quirks not to copy blindly

- The functions are named `...BilateralNCC...`, but the weights are all 1.0 and `sigma_spatial`/`sigma_color` are
  never read. The reference's numbers are therefore achieved with **plain** NCC.
- The anchor search covers 8 x `rotate_time` angular sectors (16 or 32), each taking the first STRONG hit along a
  sparsely sampled ray or the nearest STRONG pixel of a weak sample. It then keeps at most 8 RANSAC inliers,
  sorted by plane distance. "One anchor in each of 8 directions" is the right intuition, but the code is more
  involved than that.
- Bugs and quirks:
  - The jitter in the anchor search is never negative (unsigned arithmetic).
  - `unSetBit` clears bits 0..n.
  - `cost_array = {2.0f}` zero-initialises the missing candidates.
  - Textures use wrap addressing, so out-of-image samples are not penalised.
  - `RescaleMatToTargetSize` swaps its x and y scale factors.
  - `GeneratePertubedPlaneHypothesis` and `TriangleArea` are dead code.

### Minimal reimplementation checklist (WGSL)

| Buffer | Type | Notes |
|---|---|---|
| `state` | u8 | per pass: written by `DepthToWeak`, read next pass; WEAK becomes UNKNOWN on anchor failure |
| `nearest_strong` | i16x2 | brute 201x201 or JFA |
| `anchors` | 8 x i16x2 per pixel, or compacted | built per pass by `GenNeighbours` |
| `fit_plane` | vec4 | per iteration |
| `view_weights` | u8 x nsrc per pixel | needed by `DepthToWeak` / `LocalRefine` |

Per iteration, dispatch in this order: strong black, strong red, fit-plane, weak black, weak red.

Per pass, finish with: depth/normal conversion, median filter (non-WEAK), `DepthToWeak` (61-step curve),
`LocalRefine` (11-step).
