#!/usr/bin/env python3
"""A3 ranking-lab: offline retrieval-ranking evaluation, no Rust re-run needed.

Context: `docs/visual_slam_sequential_sfm_plan.md` section "A3 -- Sound
long-range loop closure", stage-1b's own dense-cadence A/B found that the
tightest MH_01 GT revisit (`i=42, j=456`, GT distance `0.160 m`) is a proven
retrieval RANKING miss: queries issued at arrivals 453/458 never rank
candidate ~42 into the current index's own top-3 (best-ranked candidates
there were arrivals 199/196 at similarity 0.39/0.30). This script is "ranking
slice A": an OFFLINE laboratory to try ranking/aggregation methods against
the EXACT raw material (`examples/euroc_dpvo_vo_demo.rs --ll-dump-frame-descriptors`'s
per-frame SuperPoint keypoints + 256-d descriptors, dumped in patch-grid
coordinates -- see `crates/vision/src/dpvo/npz.rs::write_npy_f32`'s own doc
for the on-disk format) without needing a fresh Rust run + ONNX inference for
every idea.

# What the CURRENT Rust index actually does (read directly from
# `pipelines/slam/src/dpvo_long_loop.rs` before writing this script, not
# assumed)

`crate::dpvo_long_loop`'s own module doc ("Design choice: VLAD, not the
vocab-tree TF-IDF index") states the retrieval front-end is **VLAD**
(`visloc_vision::place_recognition::vlad`) over a `Vocabulary` built ONCE
from the first `vocab_bootstrap_frames` (default 40) frames' pooled
descriptors (`k = vocab_words`, default 32, via `Vocabulary::build`), scored
by plain cosine similarity (`DpvoLongLoopIndex::query_candidates`). This is
**not** a TF-IDF/vocab-tree index (that machinery -- `visloc_vision::vocab_tree`
-- exists in this workspace for the COLMAP port but is explicitly NOT what
`dpvo_long_loop.rs` uses). Method (c) below ("k-means visual words + TF-IDF")
is offered because the task brief asked for it as an approximation of "the
current index" -- but per this reading, the honest comparison point for the
REAL Rust mechanism is method (a)/(b) (plain pooled-descriptor cosine, which
VLAD is a richer relative of: VLAD = per-visual-word residual pooling, mean-
pool is the k=1 degenerate case), not (c). This discrepancy is called out
again in `build_report`'s own output notes.

# The four methods

(a) mean-pooled descriptor cosine: frame vector = L2-normalized mean of the
    frame's own raw SP descriptors.
(b) GeM pooling (p=3) + cosine: a sharper (max-like) pooling than the mean;
    SEE `gem_pool`'s own doc for why this uses a SIGNED generalization of the
    textbook (non-negative-features-only) GeM formula.
(c) k-means visual words (k=256 and k=1024) + TF-IDF cosine: a from-scratch,
    numpy-only bag-of-words vocabulary FIT ONLY on arrivals `< --vocab-fit-max-arrival`
    (default 150) -- respecting the SAME "streaming honesty" this whole
    exercise cares about (a real streaming index cannot fit its vocabulary on
    frames it has not seen yet) -- then used to encode EVERY frame (including
    ones ingested long after the fit). TF-IDF weighting is computed from
    document frequencies over the WHOLE indexed corpus (not causally
    per-query) -- an explicit, DIFFERENT simplification from the vocabulary
    fit's own streaming-honesty constraint, stated here plainly: this only
    reweights which VISUAL WORDS matter, it does not leak any future
    GEOMETRY into which CANDIDATE FRAMES a query is allowed to rank (that
    constraint -- `i <= j - min_gap` -- is enforced separately, per-query, by
    `eligible_candidates`, and is the one that actually matters for recall
    honesty).
(d) direct mutual-NN descriptor match count (the "ceiling oracle"): for a
    query/candidate frame pair, cross-check (mutual nearest-neighbor, cosine
    similarity, no ratio test) match the two frames' raw descriptors and use
    the match COUNT as the similarity score -- the most expensive but most
    information-preserving method (no pooling/aggregation loss at all), an
    upper bound on what any pooled-descriptor method in this script could
    achieve. Because this is O(query_count * candidate_count * N_kp^2), it is
    evaluated over a SUBSAMPLED query/candidate set -- see
    `MutualNnSubsampling` and `build_report`'s own docstring for exactly what
    is (and is not) subsampled and why the (42, 456) pair itself is always
    force-included regardless of subsampling.

# Streaming constraint

Every method ranks candidates for query arrival `j` from ONLY the arrivals
`i` with `i <= j - min_gap` (`min_gap` defaults to
`DpvoLongLoopConfig::min_temporal_gap`'s own default, 150) -- the same
"long-range, not proximity" gap `crate::dpvo_long_loop::DpvoLongLoopIndex::query_candidates`
enforces. Unlike the real Rust index this script does NOT bound `top_k` (the
Rust index only geometrically verifies its own top-3); recall@K here is
computed directly against each method's own FULL ranked candidate list, which
is the more informative offline question ("is the right answer anywhere near
the top", not "would the exact current top_k=3 cutoff have found it").

# Labelling GT revisits

Reuses `scripts/eval_dpvo_long_loop_recall.py`'s own EuRoC GT loading +
`label_revisit_pairs` (position radius, camera optical-axis angle, minimum
arrival gap) by IMPORTING that module directly (dynamic sys.path insert, see
below) rather than re-deriving the same geometry a second time.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import eval_dpvo_long_loop_recall as recall_lib  # noqa: E402  (path setup above)

DEFAULT_MIN_TEMPORAL_GAP = recall_lib.DEFAULT_MIN_TEMPORAL_GAP  # 150
DEFAULT_NEAR_WINDOW = 5
DEFAULT_K_VALUES = [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# Loading the frame-descriptor dump
# ---------------------------------------------------------------------------


@dataclass
class DumpFrame:
    arrival_index: int
    keypoints: np.ndarray  # (N, 2) float32, patch-grid coordinates
    descriptors: np.ndarray  # (N, D) float32


def load_dump(dump_dir: Path) -> dict[int, DumpFrame]:
    """Load `manifest.csv` + every `.npy` pair it names -- see
    `examples/euroc_dpvo_vo_demo.rs`'s `dump_long_loop_frame` for the writer
    and `crates/vision/src/dpvo/npz.rs::write_npy_f32` for the exact `.npy`
    format (bare v1.0, `<f4`, C-order -- `numpy.load` reads it with zero
    extra code, no `.npz`/zipfile involved).
    """
    manifest_path = dump_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.csv under {dump_dir}")
    frames: dict[int, DumpFrame] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            arrival = int(row["arrival_index"])
            keypoint_count = int(row["keypoint_count"])
            descriptor_dim = int(row["descriptor_dim"])
            keypoints = np.load(dump_dir / row["keypoints_file"])
            descriptors = np.load(dump_dir / row["descriptors_file"])
            if keypoints.shape != (keypoint_count, 2):
                raise ValueError(
                    f"arrival {arrival}: manifest says keypoint_count={keypoint_count}, "
                    f"but {row['keypoints_file']} has shape {keypoints.shape}"
                )
            if descriptors.shape != (keypoint_count, descriptor_dim):
                raise ValueError(
                    f"arrival {arrival}: manifest says ({keypoint_count}, {descriptor_dim}), "
                    f"but {row['descriptors_file']} has shape {descriptors.shape}"
                )
            frames[arrival] = DumpFrame(
                arrival_index=arrival,
                keypoints=keypoints.astype(np.float32, copy=False),
                descriptors=descriptors.astype(np.float32, copy=False),
            )
    return frames


# ---------------------------------------------------------------------------
# Streaming candidate eligibility
# ---------------------------------------------------------------------------


def eligible_candidates(
    query_arrival: int, indexed_arrivals_sorted: list[int], min_gap: int
) -> list[int]:
    """Every indexed arrival `i` with `i <= query_arrival - min_gap` -- the
    SAME "long-range, not proximity" filter
    `crate::dpvo_long_loop::DpvoLongLoopIndex::query_candidates` applies,
    mirrored here so this script never lets a query see a candidate the real
    streaming index structurally could not have.
    """
    cutoff = query_arrival - min_gap
    # `indexed_arrivals_sorted` is sorted, so this is a simple prefix --
    # `bisect` would be marginally faster but this list is at most a few
    # thousand long in this script's own regime (hundreds to low thousands of
    # frames per run), so a plain comprehension stays readable without a
    # measurable cost.
    return [i for i in indexed_arrivals_sorted if i <= cutoff]


# ---------------------------------------------------------------------------
# Method (a): mean-pooled descriptor cosine
# ---------------------------------------------------------------------------


def mean_pool(descriptors: np.ndarray) -> np.ndarray:
    """L2-normalized mean of a frame's own descriptors."""
    if descriptors.shape[0] == 0:
        return np.zeros(descriptors.shape[1] if descriptors.ndim == 2 else 0, dtype=np.float32)
    pooled = descriptors.mean(axis=0)
    norm = np.linalg.norm(pooled)
    return pooled / norm if norm > 0 else pooled


# ---------------------------------------------------------------------------
# Method (b): GeM pooling (p=3) + cosine
# ---------------------------------------------------------------------------


def gem_pool(descriptors: np.ndarray, p: float = 3.0, eps: float = 1e-6) -> np.ndarray:
    """Generalized-mean pooling, L2-normalized.

    The textbook GeM formula (Radenovic et al., "Fine-tuning CNN Image
    Retrieval with No Human Annotation") is `(mean(x_i^p))^(1/p)` and
    ASSUMES `x_i >= 0` (it pools post-ReLU convolutional activations).
    SuperPoint descriptors are ordinary L2-normalized float vectors that CAN
    be negative, so raising a negative value to a non-integer power is
    complex/undefined -- this uses the natural SIGNED generalization instead:
    `sign(x) * |x|^p` before pooling, `sign(.) * |.|^(1/p)` after. At `p=3`
    (an odd integer) this is continuous and reduces to the textbook formula
    when every input happens to be non-negative. An explicit, documented
    deviation from the textbook formula, not a silent one.
    """
    if descriptors.shape[0] == 0:
        return np.zeros(descriptors.shape[1] if descriptors.ndim == 2 else 0, dtype=np.float32)
    signed_pow = np.sign(descriptors) * (np.abs(descriptors) + eps) ** p
    pooled = signed_pow.mean(axis=0)
    result = np.sign(pooled) * np.abs(pooled) ** (1.0 / p)
    norm = np.linalg.norm(result)
    return (result / norm if norm > 0 else result).astype(np.float32)


# ---------------------------------------------------------------------------
# Method (c): k-means visual words + TF-IDF cosine
# ---------------------------------------------------------------------------


def kmeans_fit(x: np.ndarray, k: int, iterations: int, seed: int) -> np.ndarray:
    """A from-scratch, numpy-only Lloyd's-algorithm k-means with a
    k-means++ initialization, seeded for reproducibility. `k` must be
    `<= x.shape[0]`.

    Vectorized via matmul (`||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2`, and the
    per-cluster mean update via a one-hot-assignment matmul rather than a
    Python loop over clusters) so this stays fast enough for `k` up to ~1024
    over tens of thousands of 256-d descriptors on plain CPU numpy (no
    scikit-learn dependency -- this task's own "numpy only, keep each simple"
    instruction, and this repo's own no-new-Rust-deps norm extended here to
    "no new Python deps" in spirit).
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if k > n:
        raise ValueError(f"kmeans_fit: k={k} exceeds sample count {n}")
    x64 = x.astype(np.float64, copy=False)

    # k-means++ initialization.
    centroids = np.empty((k, x.shape[1]), dtype=np.float64)
    first = rng.integers(0, n)
    centroids[0] = x64[first]
    closest_sq_dist = np.sum((x64 - centroids[0]) ** 2, axis=1)
    for c in range(1, k):
        total = closest_sq_dist.sum()
        probs = (
            closest_sq_dist / total
            if total > 0
            else np.full(n, 1.0 / n)
        )
        idx = rng.choice(n, p=probs)
        centroids[c] = x64[idx]
        new_dist = np.sum((x64 - centroids[c]) ** 2, axis=1)
        closest_sq_dist = np.minimum(closest_sq_dist, new_dist)

    x_sq = np.sum(x64 ** 2, axis=1, keepdims=True)  # (n, 1)
    for _ in range(iterations):
        c_sq = np.sum(centroids ** 2, axis=1)  # (k,)
        cross = x64 @ centroids.T  # (n, k)
        dist_sq = x_sq - 2.0 * cross + c_sq[None, :]
        assign = np.argmin(dist_sq, axis=1)

        onehot = np.zeros((n, k), dtype=np.float64)
        onehot[np.arange(n), assign] = 1.0
        counts = onehot.sum(axis=0)  # (k,)
        sums = onehot.T @ x64  # (k, d)
        nonzero = counts > 0
        centroids[nonzero] = sums[nonzero] / counts[nonzero, None]
        # Empty clusters (rare, only possible with a pathological/duplicated
        # input) keep their previous centroid rather than being
        # re-seeded -- honestly inert rather than silently reshuffling
        # cluster identities mid-fit.

    return centroids.astype(np.float32)


def nearest_centroid_indices(descriptors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Hard-assign each row of `descriptors` to its nearest `centroids` row."""
    if descriptors.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    d_sq = np.sum(descriptors.astype(np.float64) ** 2, axis=1, keepdims=True)
    c_sq = np.sum(centroids.astype(np.float64) ** 2, axis=1)
    cross = descriptors.astype(np.float64) @ centroids.astype(np.float64).T
    dist_sq = d_sq - 2.0 * cross + c_sq[None, :]
    return np.argmin(dist_sq, axis=1)


def build_tfidf_vectors(
    frames: dict[int, DumpFrame], centroids: np.ndarray
) -> dict[int, np.ndarray]:
    """Bag-of-visual-words TF-IDF vector per frame, L2-normalized.

    IDF is computed from document frequencies over the WHOLE indexed corpus
    (every frame in `frames`, not just the ones the vocabulary was fit on) --
    see this module's own docstring, "k-means visual words + TF-IDF cosine",
    for why this is a deliberate, separately-stated simplification from the
    vocabulary FIT's own streaming-honesty constraint.
    """
    k = centroids.shape[0]
    arrivals = sorted(frames)
    raw_counts: dict[int, np.ndarray] = {}
    doc_freq = np.zeros(k, dtype=np.float64)
    for arrival in arrivals:
        assign = nearest_centroid_indices(frames[arrival].descriptors, centroids)
        counts = np.bincount(assign, minlength=k).astype(np.float64)
        raw_counts[arrival] = counts
        doc_freq += counts > 0

    n_docs = len(arrivals)
    idf = np.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0  # sklearn-style smoothed IDF

    vectors: dict[int, np.ndarray] = {}
    for arrival in arrivals:
        tf = raw_counts[arrival]
        total = tf.sum()
        tf_normalized = tf / total if total > 0 else tf
        vec = tf_normalized * idf
        norm = np.linalg.norm(vec)
        vectors[arrival] = (vec / norm if norm > 0 else vec).astype(np.float32)
    return vectors


# ---------------------------------------------------------------------------
# Method (d): direct mutual-NN descriptor match count (ceiling oracle)
# ---------------------------------------------------------------------------


def mutual_nn_match_count(desc_a: np.ndarray, desc_b: np.ndarray) -> int:
    """Mutual-nearest-neighbor descriptor match count between two frames'
    raw descriptor sets, by cosine similarity, no Lowe ratio test (a plain
    "is `a`'s best match `b`, AND is `b`'s best match `a`" cross-check,
    matching `visloc_vision::matching::CrossCheckMatcher`'s own definition of
    "mutual" minus the ratio test -- the ratio test is an extra precision
    filter, not part of what makes a match "mutual").
    """
    if desc_a.shape[0] == 0 or desc_b.shape[0] == 0:
        return 0
    a = desc_a / np.clip(np.linalg.norm(desc_a, axis=1, keepdims=True), 1e-12, None)
    b = desc_b / np.clip(np.linalg.norm(desc_b, axis=1, keepdims=True), 1e-12, None)
    sim = a @ b.T  # (Na, Nb) cosine similarity
    best_b_for_a = np.argmax(sim, axis=1)
    best_a_for_b = np.argmax(sim, axis=0)
    return int(np.sum(best_a_for_b[best_b_for_a] == np.arange(a.shape[0])))


@dataclass
class MutualNnSubsampling:
    """Runtime-bounding knobs for method (d) -- see `build_report`'s own
    docstring for exactly how these are applied (and how the (42, 456) pair
    itself always bypasses both caps).
    """

    query_cap: int
    candidate_cap: int
    force_query_range: tuple[int, int]
    force_candidate_window: tuple[int, int]


def select_mutual_nn_queries(
    denom_queries: list[int], subsampling: MutualNnSubsampling
) -> list[int]:
    """Evenly-spaced subsample of `denom_queries`, capped at `query_cap`,
    but ALWAYS including every query arrival inside `force_query_range`
    (inclusive) that is itself a labelled query arrival -- the diagnostic
    range `[451, 461]` this task's own report needs is never subsampled
    away.
    """
    lo, hi = subsampling.force_query_range
    forced = [j for j in denom_queries if lo <= j <= hi]
    remainder = [j for j in denom_queries if not (lo <= j <= hi)]
    budget = max(subsampling.query_cap - len(forced), 0)
    if budget >= len(remainder) or budget <= 0:
        sampled_remainder = remainder[:budget] if budget > 0 else []
    else:
        stride = len(remainder) / budget
        sampled_remainder = [remainder[int(i * stride)] for i in range(budget)]
    return sorted(set(forced) | set(sampled_remainder))


def select_mutual_nn_candidates(
    query_arrival: int,
    candidates: list[int],
    subsampling: MutualNnSubsampling,
) -> list[int]:
    """Evenly-spaced subsample of `candidates`, capped at `candidate_cap`,
    but ALWAYS including any candidate inside `force_candidate_window`
    (inclusive) when `query_arrival` falls in `force_query_range` -- so the
    (42, 456)-style report always has arrival 42 present when it is actually
    an eligible candidate for a forced query.
    """
    lo, hi = subsampling.force_candidate_window
    in_forced_query_range = (
        subsampling.force_query_range[0] <= query_arrival <= subsampling.force_query_range[1]
    )
    forced = [c for c in candidates if in_forced_query_range and lo <= c <= hi]
    remainder = [c for c in candidates if c not in forced]
    budget = max(subsampling.candidate_cap - len(forced), 0)
    if budget >= len(remainder) or budget <= 0:
        sampled_remainder = remainder[:budget] if budget > 0 else []
    else:
        stride = len(remainder) / budget
        sampled_remainder = [remainder[int(i * stride)] for i in range(budget)]
    return sorted(set(forced) | set(sampled_remainder))


# ---------------------------------------------------------------------------
# Shared ranking + recall machinery
# ---------------------------------------------------------------------------


def rank_by_frame_vectors(
    frame_vectors: dict[int, np.ndarray],
    query_arrivals: list[int],
    min_gap: int,
) -> dict[int, list[tuple[int, float]]]:
    """For every `query_arrivals` entry present in `frame_vectors`, rank
    every ELIGIBLE candidate (see `eligible_candidates`) by descending cosine
    similarity (`frame_vectors` are assumed already L2-normalized, so a dot
    product IS cosine similarity).
    """
    indexed_sorted = sorted(frame_vectors)
    rankings: dict[int, list[tuple[int, float]]] = {}
    for j in query_arrivals:
        if j not in frame_vectors:
            continue
        candidates = eligible_candidates(j, indexed_sorted, min_gap)
        if not candidates:
            rankings[j] = []
            continue
        cand_matrix = np.stack([frame_vectors[i] for i in candidates])
        scores = cand_matrix @ frame_vectors[j]
        order = np.argsort(-scores, kind="stable")
        rankings[j] = [(candidates[idx], float(scores[idx])) for idx in order]
    return rankings


def rank_by_mutual_nn(
    frames: dict[int, DumpFrame],
    query_arrivals: list[int],
    min_gap: int,
    subsampling: MutualNnSubsampling,
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, list[int]]]:
    """Method (d)'s own ranking, over a SUBSAMPLED query/candidate set (see
    `MutualNnSubsampling`). Returns `(rankings, candidates_considered)` --
    the second dict records exactly which candidate arrivals were actually
    scored per query (the honest "this is the subsample, not the full
    eligible set" record for the JSON report).
    """
    indexed_sorted = sorted(frames)
    full_denom = [j for j in query_arrivals if j in frames]
    selected_queries = select_mutual_nn_queries(full_denom, subsampling)
    rankings: dict[int, list[tuple[int, float]]] = {}
    candidates_considered: dict[int, list[int]] = {}
    for j in selected_queries:
        eligible = eligible_candidates(j, indexed_sorted, min_gap)
        candidates = select_mutual_nn_candidates(j, eligible, subsampling)
        candidates_considered[j] = candidates
        scored = [
            (i, float(mutual_nn_match_count(frames[j].descriptors, frames[i].descriptors)))
            for i in candidates
        ]
        scored.sort(key=lambda pair: -pair[1])
        rankings[j] = scored
    return rankings, candidates_considered


def recall_at_k(
    rankings: dict[int, list[tuple[int, float]]],
    mask: np.ndarray,
    near_window: int,
    k_values: list[int],
) -> tuple[dict[int, dict], list[int]]:
    """`{k: {hits, denominator, recall}}` plus the list of query arrivals
    that formed the denominator (every `j` with >= 1 GT partner AND present
    in `rankings`, i.e. actually evaluated by this method).

    A "hit" at `K` means at least one of the top-`K` ranked candidates is
    within `near_window` arrivals of ANY true GT partner of `j` -- mirrors
    `scripts/eval_dpvo_long_loop_recall.py::evaluate_run`'s own near-window
    matching (a returned candidate need not be the EXACT labelled arrival,
    just close enough that it is unambiguously the same physical revisit).
    """
    n = mask.shape[1]
    partner_map = {j: np.nonzero(mask[:, j])[0] for j in range(n) if mask[:, j].any()}
    denom_queries = sorted(j for j in partner_map if j in rankings and rankings[j])
    results: dict[int, dict] = {}
    for k in k_values:
        hits = 0
        for j in denom_queries:
            top = [arrival for arrival, _ in rankings[j][:k]]
            partners = partner_map[j]
            if any(abs(t - p) <= near_window for t in top for p in partners):
                hits += 1
        denom = len(denom_queries)
        results[k] = {
            "k": k,
            "hits": hits,
            "denominator": denom,
            "recall": hits / denom if denom else None,
        }
    return results, denom_queries


def rank_of_candidate(ranking: list[tuple[int, float]], candidate: int) -> int | None:
    """0-indexed rank of `candidate` in `ranking` (best = 0), or `None` if
    `candidate` was never scored for this query (either genuinely ineligible
    under the streaming gap, or -- for method (d) -- subsampled away).
    """
    for idx, (arrival, _score) in enumerate(ranking):
        if arrival == candidate:
            return idx
    return None


def similarity_of_candidate(ranking: list[tuple[int, float]], candidate: int) -> float | None:
    for arrival, score in ranking:
        if arrival == candidate:
            return score
    return None


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


@dataclass
class MethodResult:
    name: str
    elapsed_s: float
    recall: dict[int, dict]
    denom_queries: list[int]
    pair_rank: int | None
    pair_similarity: float | None
    near_target_ranks: dict[int, int | None] = field(default_factory=dict)


def build_report(
    dump_dir: Path,
    gt_dir: Path,
    stride: int,
    max_frames: int,
    min_gap: int,
    radius_m: float,
    radius_secondary_m: float,
    max_angle_deg: float,
    near_window: int,
    gt_tol_ns: int,
    k_values: list[int],
    vocab_fit_max_arrival: int,
    kmeans_ks: list[int],
    kmeans_iterations: int,
    kmeans_seed: int,
    kmeans_fit_sample_cap: int,
    mutual_nn_query_cap: int,
    mutual_nn_candidate_cap: int,
    diagnostic_i: int,
    diagnostic_j: int,
    diagnostic_near_window: int,
    diagnostic_query_range: tuple[int, int],
) -> dict:
    frames = load_dump(dump_dir)
    indexed_arrivals = sorted(frames)

    gt_csv = gt_dir / "state_groundtruth_estimate0" / "data.csv"
    cam0_csv = gt_dir / "cam0" / "data.csv"
    cam0_sensor_yaml = gt_dir / "cam0" / "sensor.yaml"
    gt = recall_lib.load_ground_truth(gt_csv)
    cam0_timestamps = recall_lib.load_cam0_timestamps(cam0_csv)
    t_bs = recall_lib.parse_cam0_t_bs(cam0_sensor_yaml)
    r_bc = t_bs[:3, :3]

    n_arrivals = recall_lib.num_arrivals(len(cam0_timestamps), stride, max_frames)
    positions, axes = recall_lib.build_arrival_poses(
        n_arrivals, stride, cam0_timestamps, gt, r_bc, gt_tol_ns
    )

    masks = {
        "primary": recall_lib.label_revisit_pairs(
            positions, axes, min_gap, radius_m, max_angle_deg
        ),
        "secondary": recall_lib.label_revisit_pairs(
            positions, axes, min_gap, radius_secondary_m, max_angle_deg
        ),
    }

    all_query_arrivals = indexed_arrivals  # every ingested frame is a candidate query

    method_results: dict[str, dict[str, MethodResult]] = {"primary": {}, "secondary": {}}

    def run_vector_method(name: str, frame_vectors: dict[int, np.ndarray]) -> None:
        start = time.monotonic()
        rankings = rank_by_frame_vectors(frame_vectors, all_query_arrivals, min_gap)
        elapsed = time.monotonic() - start
        pair_ranking = rankings.get(diagnostic_j, [])
        near_target_ranks = {}
        for j in range(diagnostic_query_range[0], diagnostic_query_range[1] + 1):
            if j not in rankings:
                continue
            ranking = rankings[j]
            near = [
                idx
                for idx, (arrival, _score) in enumerate(ranking)
                if abs(arrival - diagnostic_i) <= diagnostic_near_window
            ]
            near_target_ranks[j] = min(near) if near else None
        for radius_key, mask in masks.items():
            recall, denom = recall_at_k(rankings, mask, near_window, k_values)
            method_results[radius_key][name] = MethodResult(
                name=name,
                elapsed_s=elapsed,
                recall=recall,
                denom_queries=denom,
                pair_rank=rank_of_candidate(pair_ranking, diagnostic_i),
                pair_similarity=similarity_of_candidate(pair_ranking, diagnostic_i),
                near_target_ranks=near_target_ranks,
            )

    # --- (a) mean-pool ---
    mean_vectors = {arrival: mean_pool(frame.descriptors) for arrival, frame in frames.items()}
    run_vector_method("mean_pool_cosine", mean_vectors)

    # --- (b) GeM (p=3) ---
    gem_vectors = {arrival: gem_pool(frame.descriptors, p=3.0) for arrival, frame in frames.items()}
    run_vector_method("gem_p3_cosine", gem_vectors)

    # --- (c) k-means + TF-IDF, for each requested k ---
    fit_arrivals = [a for a in indexed_arrivals if a < vocab_fit_max_arrival]
    fit_descriptors = np.concatenate(
        [frames[a].descriptors for a in fit_arrivals], axis=0
    ) if fit_arrivals else np.zeros((0, 0), dtype=np.float32)
    kmeans_notes = {
        "fit_arrivals_count": len(fit_arrivals),
        "fit_descriptors_available": int(fit_descriptors.shape[0]),
        "fit_sample_cap": kmeans_fit_sample_cap,
    }
    if fit_descriptors.shape[0] > kmeans_fit_sample_cap:
        rng = np.random.default_rng(kmeans_seed)
        sample_idx = rng.choice(fit_descriptors.shape[0], size=kmeans_fit_sample_cap, replace=False)
        fit_sample = fit_descriptors[sample_idx]
    else:
        fit_sample = fit_descriptors
    kmeans_notes["fit_descriptors_used"] = int(fit_sample.shape[0])

    for k in kmeans_ks:
        start = time.monotonic()
        centroids = kmeans_fit(fit_sample, k, kmeans_iterations, kmeans_seed)
        tfidf_vectors = build_tfidf_vectors(frames, centroids)
        fit_elapsed = time.monotonic() - start
        method_name = f"kmeans_k{k}_tfidf_cosine"
        run_vector_method(method_name, tfidf_vectors)
        for radius_key in method_results:
            method_results[radius_key][method_name].elapsed_s += fit_elapsed

    # --- (d) mutual-NN oracle ---
    subsampling = MutualNnSubsampling(
        query_cap=mutual_nn_query_cap,
        candidate_cap=mutual_nn_candidate_cap,
        force_query_range=diagnostic_query_range,
        force_candidate_window=(
            diagnostic_i - diagnostic_near_window,
            diagnostic_i + diagnostic_near_window,
        ),
    )
    start = time.monotonic()
    mnn_rankings, mnn_candidates_considered = rank_by_mutual_nn(
        frames, all_query_arrivals, min_gap, subsampling
    )
    mnn_elapsed = time.monotonic() - start
    pair_ranking = mnn_rankings.get(diagnostic_j, [])
    near_target_ranks = {}
    for j in range(diagnostic_query_range[0], diagnostic_query_range[1] + 1):
        if j not in mnn_rankings:
            continue
        ranking = mnn_rankings[j]
        near = [
            idx
            for idx, (arrival, _score) in enumerate(ranking)
            if abs(arrival - diagnostic_i) <= diagnostic_near_window
        ]
        near_target_ranks[j] = min(near) if near else None
    for radius_key, mask in masks.items():
        recall, denom = recall_at_k(mnn_rankings, mask, near_window, k_values)
        method_results[radius_key]["mutual_nn_oracle"] = MethodResult(
            name="mutual_nn_oracle",
            elapsed_s=mnn_elapsed,
            recall=recall,
            denom_queries=denom,
            pair_rank=rank_of_candidate(pair_ranking, diagnostic_i),
            pair_similarity=similarity_of_candidate(pair_ranking, diagnostic_i),
            near_target_ranks=near_target_ranks,
        )

    # --- the specific (42, 456) pair, guaranteed present regardless of any
    # subsampling above (a direct, uncapped computation) ---
    pair_report: dict[str, object] = {
        "target_i": diagnostic_i,
        "target_j": diagnostic_j,
    }
    if diagnostic_i in frames and diagnostic_j in frames:
        pair_report["mutual_nn_match_count"] = mutual_nn_match_count(
            frames[diagnostic_j].descriptors, frames[diagnostic_i].descriptors
        )
        pair_report["keypoint_count_i"] = int(frames[diagnostic_i].keypoints.shape[0])
        pair_report["keypoint_count_j"] = int(frames[diagnostic_j].keypoints.shape[0])
    else:
        pair_report["mutual_nn_match_count"] = None
        pair_report["note"] = "target_i or target_j missing from the dump"

    def method_to_dict(result: MethodResult) -> dict:
        return {
            "name": result.name,
            "elapsed_s": result.elapsed_s,
            "recall_at_k": list(result.recall.values()),
            "denominator_query_count": len(result.denom_queries),
            "pair_42_456_rank": result.pair_rank,
            "pair_42_456_similarity": result.pair_similarity,
            "near_target_ranks_by_query": result.near_target_ranks,
        }

    report = {
        "dump_dir": str(dump_dir),
        "gt_dir": str(gt_dir),
        "config": {
            "stride": stride,
            "max_frames": max_frames,
            "n_arrivals": n_arrivals,
            "n_indexed_frames": len(indexed_arrivals),
            "min_gap": min_gap,
            "radius_m_primary": radius_m,
            "radius_m_secondary": radius_secondary_m,
            "max_angle_deg": max_angle_deg,
            "near_window": near_window,
            "vocab_fit_max_arrival": vocab_fit_max_arrival,
            "kmeans_ks": kmeans_ks,
            "kmeans_iterations": kmeans_iterations,
            "kmeans_seed": kmeans_seed,
            "mutual_nn_query_cap": mutual_nn_query_cap,
            "mutual_nn_candidate_cap": mutual_nn_candidate_cap,
            "diagnostic_i": diagnostic_i,
            "diagnostic_j": diagnostic_j,
            "diagnostic_query_range": list(diagnostic_query_range),
        },
        "kmeans_fit_notes": kmeans_notes,
        "mutual_nn_subsampling_notes": {
            "queries_selected": sorted(mnn_rankings),
            "candidates_considered_per_query_sample": {
                str(j): mnn_candidates_considered[j]
                for j in sorted(mnn_candidates_considered)
                if diagnostic_query_range[0] <= j <= diagnostic_query_range[1]
            },
        },
        "methods_by_radius": {
            radius_key: {name: method_to_dict(result) for name, result in methods.items()}
            for radius_key, methods in method_results.items()
        },
        "pair_42_456": pair_report,
        "notes": {
            "rust_index_actual_mechanism": (
                "pipelines/slam/src/dpvo_long_loop.rs uses VLAD "
                "(visloc_vision::place_recognition::vlad) over a Vocabulary(k=32) "
                "built once from the first 40 committed frames, scored by cosine "
                "similarity -- NOT the TF-IDF vocab-tree index "
                "(visloc_vision::vocab_tree) method (c) here approximates. "
                "Method (a)/(b) (plain pooled-descriptor cosine) is the closer "
                "structural analog to the real mechanism; (c) is included because "
                "the task brief asked for it as an explicit comparison point."
            ),
            "tfidf_idf_not_causal": (
                "TF-IDF's IDF weights are computed from document frequency over "
                "the WHOLE indexed corpus, not causally per-query -- see "
                "build_tfidf_vectors's own doc. Only the k-means FIT and the "
                "per-query candidate eligibility (i <= j - min_gap) respect "
                "streaming honesty strictly."
            ),
            "mutual_nn_oracle_is_subsampled": (
                "method (d) is evaluated over a capped, evenly-spaced subsample "
                "of (query, candidate) pairs (see mutual_nn_subsampling_notes and "
                "MutualNnSubsampling) -- its recall@K is a subsample-based "
                "estimate, not exhaustive, except for the (42, 456) pair itself "
                "(pair_42_456), which is always computed directly."
            ),
        },
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_int_list(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def format_text_report(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"dump_dir: {report['dump_dir']}")
    lines.append(f"gt_dir: {report['gt_dir']}")
    cfg = report["config"]
    lines.append(
        f"n_arrivals={cfg['n_arrivals']} n_indexed_frames={cfg['n_indexed_frames']} "
        f"min_gap={cfg['min_gap']} kmeans_ks={cfg['kmeans_ks']}"
    )
    lines.append("")
    pair_label = f"pair({cfg['diagnostic_i']},{cfg['diagnostic_j']})"
    for radius_key in ("primary", "secondary"):
        methods = report["methods_by_radius"][radius_key]
        radius_m = cfg["radius_m_primary"] if radius_key == "primary" else cfg["radius_m_secondary"]
        lines.append(f"=== radius={radius_m} m ===")
        for name, result in methods.items():
            recall_str = ", ".join(
                f"R@{r['k']}={r['recall']:.4f}" if r["recall"] is not None else f"R@{r['k']}=n/a"
                for r in result["recall_at_k"]
            )
            lines.append(
                f"  {name:28s} denom={result['denominator_query_count']:4d} "
                f"elapsed={result['elapsed_s']:6.2f}s  {recall_str}  "
                f"{pair_label}: rank={result['pair_42_456_rank']} "
                f"sim={result['pair_42_456_similarity']}"
            )
        lines.append("")
    pair = report["pair_42_456"]
    lines.append(
        f"pair ({pair['target_i']}, {pair['target_j']}): "
        f"mutual_nn_match_count={pair.get('mutual_nn_match_count')}"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--min-gap", type=int, default=DEFAULT_MIN_TEMPORAL_GAP)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--radius-secondary", type=float, default=0.5)
    parser.add_argument("--max-angle-deg", type=float, default=30.0)
    parser.add_argument("--near-window", type=int, default=DEFAULT_NEAR_WINDOW)
    parser.add_argument("--gt-tol-ns", type=int, default=5_000_000)
    parser.add_argument("--k-values", type=parse_int_list, default=DEFAULT_K_VALUES)
    parser.add_argument("--vocab-fit-max-arrival", type=int, default=150)
    parser.add_argument("--kmeans-ks", type=parse_int_list, default=[256, 1024])
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--kmeans-seed", type=int, default=0)
    parser.add_argument("--kmeans-fit-sample-cap", type=int, default=20_000)
    parser.add_argument("--mutual-nn-query-cap", type=int, default=60)
    parser.add_argument("--mutual-nn-candidate-cap", type=int, default=150)
    parser.add_argument("--diagnostic-i", type=int, default=42)
    parser.add_argument("--diagnostic-j", type=int, default=456)
    parser.add_argument("--diagnostic-near-window", type=int, default=5)
    parser.add_argument(
        "--diagnostic-query-range",
        type=str,
        default="451,461",
        help="inclusive 'lo,hi' range of query arrivals for the near-target-rank report",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    lo_str, hi_str = args.diagnostic_query_range.split(",")
    diagnostic_query_range = (int(lo_str), int(hi_str))

    report = build_report(
        dump_dir=args.dump_dir,
        gt_dir=args.gt_dir,
        stride=args.stride,
        max_frames=args.max_frames,
        min_gap=args.min_gap,
        radius_m=args.radius,
        radius_secondary_m=args.radius_secondary,
        max_angle_deg=args.max_angle_deg,
        near_window=args.near_window,
        gt_tol_ns=args.gt_tol_ns,
        k_values=args.k_values,
        vocab_fit_max_arrival=args.vocab_fit_max_arrival,
        kmeans_ks=args.kmeans_ks,
        kmeans_iterations=args.kmeans_iterations,
        kmeans_seed=args.kmeans_seed,
        kmeans_fit_sample_cap=args.kmeans_fit_sample_cap,
        mutual_nn_query_cap=args.mutual_nn_query_cap,
        mutual_nn_candidate_cap=args.mutual_nn_candidate_cap,
        diagnostic_i=args.diagnostic_i,
        diagnostic_j=args.diagnostic_j,
        diagnostic_near_window=args.diagnostic_near_window,
        diagnostic_query_range=diagnostic_query_range,
    )

    print(format_text_report(report))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
