"""Tests for `scripts/eval_dpvo_retrieval_ranking_offline.py`.

Follows `tests/test_eval_dpvo_long_loop_recall.py`'s own style: load the
script by path (`importlib`) rather than requiring `scripts/` on `sys.path`
for THIS test file, plain pytest functions + `tmp_path`, synthetic small
arrays. Per this task's own brief, these tests cover the labelling/
simulation CORE (streaming-eligibility gate, pooling, k-means, TF-IDF,
mutual-NN matching, recall@K, subsampling) -- not a real dump/GT fixture
(those live on `E:` and are out of scope for a fast, hermetic test suite).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_dpvo_retrieval_ranking_offline.py"
# The module under test itself dynamically inserts `scripts/` onto
# `sys.path` (so it can `import eval_dpvo_long_loop_recall`) -- do the same
# here BEFORE loading it by path, so that import resolves identically to how
# `python scripts/eval_dpvo_retrieval_ranking_offline.py` would run it.
SCRIPTS_DIR = SCRIPT.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
SPEC = importlib.util.spec_from_file_location("eval_dpvo_retrieval_ranking_offline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE  # dataclasses' typing introspection needs this registered
SPEC.loader.exec_module(MODULE)


# ---------------------------------------------------------------------------
# eligible_candidates (the streaming constraint)
# ---------------------------------------------------------------------------


def test_eligible_candidates_respects_min_gap() -> None:
    indexed = list(range(0, 500, 1))
    result = MODULE.eligible_candidates(query_arrival=456, indexed_arrivals_sorted=indexed, min_gap=150)
    assert result == list(range(0, 307))  # 456 - 150 = 306, inclusive


def test_eligible_candidates_excludes_future_and_too_recent() -> None:
    indexed = [0, 10, 50, 100, 149, 150, 200, 300]
    result = MODULE.eligible_candidates(query_arrival=200, indexed_arrivals_sorted=indexed, min_gap=150)
    assert result == [0, 10, 50]  # 200 - 150 = 50 is the cutoff (inclusive); 100/149/150/200/300 excluded


def test_eligible_candidates_empty_when_nothing_old_enough() -> None:
    indexed = [0, 5, 10]
    result = MODULE.eligible_candidates(query_arrival=20, indexed_arrivals_sorted=indexed, min_gap=150)
    assert result == []


# ---------------------------------------------------------------------------
# mean_pool / gem_pool
# ---------------------------------------------------------------------------


def test_mean_pool_is_l2_normalized() -> None:
    descriptors = np.asarray([[3.0, 4.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    pooled = MODULE.mean_pool(descriptors)
    np.testing.assert_allclose(np.linalg.norm(pooled), 1.0, atol=1e-6)
    np.testing.assert_allclose(pooled, [0.6, 0.8, 0.0], atol=1e-6)


def test_mean_pool_empty_descriptors_returns_zero_vector() -> None:
    descriptors = np.zeros((0, 8), dtype=np.float32)
    pooled = MODULE.mean_pool(descriptors)
    assert pooled.shape == (8,)
    assert np.all(pooled == 0.0)


def test_gem_pool_is_l2_normalized_and_finite() -> None:
    rng = np.random.default_rng(0)
    descriptors = rng.normal(size=(10, 16)).astype(np.float32)
    pooled = MODULE.gem_pool(descriptors, p=3.0)
    assert np.all(np.isfinite(pooled))
    np.testing.assert_allclose(np.linalg.norm(pooled), 1.0, atol=1e-5)


def test_gem_pool_matches_textbook_formula_on_nonnegative_input() -> None:
    # On strictly non-negative input, the signed generalization used here
    # must reduce exactly to the textbook (mean(x^p))^(1/p) formula (before
    # L2 normalization) -- verifies the "sign" bookkeeping is a genuine no-op
    # in the case the textbook formula was defined for.
    descriptors = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pooled = MODULE.gem_pool(descriptors, p=3.0, eps=0.0)
    expected_unnormalized = np.mean(descriptors ** 3, axis=0) ** (1.0 / 3.0)
    expected = expected_unnormalized / np.linalg.norm(expected_unnormalized)
    np.testing.assert_allclose(pooled, expected, atol=1e-5)


def test_gem_pool_handles_negative_values_without_nan() -> None:
    descriptors = np.asarray([[-1.0, 2.0], [1.0, -2.0]], dtype=np.float32)
    pooled = MODULE.gem_pool(descriptors, p=3.0)
    assert np.all(np.isfinite(pooled))


# ---------------------------------------------------------------------------
# kmeans_fit / nearest_centroid_indices
# ---------------------------------------------------------------------------


def test_kmeans_fit_recovers_well_separated_clusters() -> None:
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0.0, scale=0.01, size=(20, 4)).astype(np.float32)
    cluster_b = rng.normal(loc=10.0, scale=0.01, size=(20, 4)).astype(np.float32)
    x = np.concatenate([cluster_a, cluster_b], axis=0)
    centroids = MODULE.kmeans_fit(x, k=2, iterations=10, seed=0)
    assert centroids.shape == (2, 4)
    # One centroid should land near each cluster's true mean (order not
    # guaranteed -- check as a set of distances).
    dist_to_a = np.linalg.norm(centroids - 0.0, axis=1)
    dist_to_b = np.linalg.norm(centroids - 10.0, axis=1)
    assert min(dist_to_a) < 0.5
    assert min(dist_to_b) < 0.5


def test_kmeans_fit_is_deterministic_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 6)).astype(np.float32)
    c1 = MODULE.kmeans_fit(x, k=3, iterations=5, seed=42)
    c2 = MODULE.kmeans_fit(x, k=3, iterations=5, seed=42)
    np.testing.assert_array_equal(c1, c2)


def test_kmeans_fit_rejects_k_larger_than_sample_count() -> None:
    x = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        MODULE.kmeans_fit(x, k=5, iterations=1, seed=0)


def test_nearest_centroid_indices_picks_the_closest_row() -> None:
    centroids = np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    descriptors = np.asarray([[0.1, -0.1], [9.9, 10.1], [0.0, 0.0]], dtype=np.float32)
    assign = MODULE.nearest_centroid_indices(descriptors, centroids)
    np.testing.assert_array_equal(assign, [0, 1, 0])


def test_nearest_centroid_indices_empty_descriptors() -> None:
    centroids = np.zeros((4, 8), dtype=np.float32)
    assign = MODULE.nearest_centroid_indices(np.zeros((0, 8), dtype=np.float32), centroids)
    assert assign.shape == (0,)


# ---------------------------------------------------------------------------
# build_tfidf_vectors
# ---------------------------------------------------------------------------


def test_build_tfidf_vectors_are_l2_normalized() -> None:
    centroids = np.asarray([[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]], dtype=np.float32)
    frames = {
        0: MODULE.DumpFrame(0, np.zeros((2, 2)), np.asarray([[0.0, 0.0], [0.1, -0.1]], dtype=np.float32)),
        1: MODULE.DumpFrame(1, np.zeros((1, 2)), np.asarray([[9.9, 10.1]], dtype=np.float32)),
    }
    vectors = MODULE.build_tfidf_vectors(frames, centroids)
    assert set(vectors) == {0, 1}
    for vec in vectors.values():
        norm = np.linalg.norm(vec)
        assert norm == pytest.approx(1.0, abs=1e-6) or norm == pytest.approx(0.0, abs=1e-6)


def test_build_tfidf_vectors_word_used_by_every_frame_gets_low_idf_weight() -> None:
    # Word 0 and word 2 appear in EVERY frame (uninformative, minimum
    # possible smoothed IDF); word 1 appears in only frame 2 (informative,
    # higher IDF). Frame 2 has ONE raw occurrence of each of its three
    # words (equal TF), so any difference between its own word-0 and
    # word-1 weight is attributable ONLY to IDF -- a same-frame comparison,
    # avoiding the (single-distinct-word-per-frame) degenerate case where
    # L2 normalization alone would erase any IDF effect.
    centroids = np.asarray([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    frames = {
        0: MODULE.DumpFrame(
            0,
            np.zeros((4, 2)),
            np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [20.0, 20.0]], dtype=np.float32),
        ),
        1: MODULE.DumpFrame(
            1,
            np.zeros((4, 2)),
            np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [20.0, 20.0]], dtype=np.float32),
        ),
        2: MODULE.DumpFrame(
            2,
            np.zeros((3, 2)),
            np.asarray([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]], dtype=np.float32),
        ),
    }
    vectors = MODULE.build_tfidf_vectors(frames, centroids)
    # Within frame 2: word 1 (df=1, rare) must outweigh word 0 (df=3,
    # ubiquitous) despite both having the same raw count (1) in this frame.
    assert vectors[2][1] > vectors[2][0]


# ---------------------------------------------------------------------------
# mutual_nn_match_count
# ---------------------------------------------------------------------------


def test_mutual_nn_match_count_identical_frames_matches_everything() -> None:
    rng = np.random.default_rng(0)
    descriptors = rng.normal(size=(12, 16)).astype(np.float32)
    count = MODULE.mutual_nn_match_count(descriptors, descriptors.copy())
    assert count == 12


def test_mutual_nn_match_count_orthogonal_random_sets_rarely_all_match() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 32)).astype(np.float32)
    b = rng.normal(size=(30, 32)).astype(np.float32)
    count = MODULE.mutual_nn_match_count(a, b)
    assert 0 <= count <= 30
    assert count < 30  # independent random descriptors should not ALL mutually match


def test_mutual_nn_match_count_empty_inputs_returns_zero() -> None:
    assert MODULE.mutual_nn_match_count(np.zeros((0, 8), dtype=np.float32), np.zeros((5, 8), dtype=np.float32)) == 0
    assert MODULE.mutual_nn_match_count(np.zeros((5, 8), dtype=np.float32), np.zeros((0, 8), dtype=np.float32)) == 0


def test_mutual_nn_match_count_one_clear_pair() -> None:
    # Two descriptors each; (a0, b0) are near-identical and clearly closer to
    # each other than to (a1, b1), and vice versa -- exactly one mutual
    # match expected for EACH pair, so 2 total.
    a = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b = np.asarray([[0.99, 0.01], [0.01, 0.99]], dtype=np.float32)
    assert MODULE.mutual_nn_match_count(a, b) == 2


# ---------------------------------------------------------------------------
# rank_by_frame_vectors / recall_at_k / rank_of_candidate
# ---------------------------------------------------------------------------


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_rank_by_frame_vectors_orders_by_descending_cosine_and_respects_gap() -> None:
    frame_vectors = {
        0: _unit([1.0, 0.0]),  # least similar to the query direction
        100: _unit([0.9, 0.1]),  # exactly the query's own direction -- must rank first
        200: _unit([0.0, 1.0]),  # too recent relative to query 300 under min_gap=150
        300: _unit([0.9, 0.1]),  # the query itself
    }
    rankings = MODULE.rank_by_frame_vectors(frame_vectors, query_arrivals=[300], min_gap=150)
    ranking = rankings[300]
    ranked_arrivals = [arrival for arrival, _ in ranking]
    assert 200 not in ranked_arrivals  # gap 300-200=100 < 150, must be excluded
    assert ranked_arrivals[0] == 100  # exact-direction match must outrank the off-axis candidate 0
    scores = [score for _, score in ranking]
    assert scores == sorted(scores, reverse=True)  # descending by cosine similarity


def test_rank_by_frame_vectors_no_eligible_candidates_gives_empty_ranking() -> None:
    frame_vectors = {0: _unit([1.0, 0.0]), 50: _unit([0.0, 1.0])}
    rankings = MODULE.rank_by_frame_vectors(frame_vectors, query_arrivals=[50], min_gap=150)
    assert rankings[50] == []


def test_rank_of_candidate_and_similarity_of_candidate() -> None:
    ranking = [(10, 0.9), (5, 0.5), (42, 0.1)]
    assert MODULE.rank_of_candidate(ranking, 42) == 2
    assert MODULE.rank_of_candidate(ranking, 999) is None
    assert MODULE.similarity_of_candidate(ranking, 10) == pytest.approx(0.9)
    assert MODULE.similarity_of_candidate(ranking, 999) is None


def test_recall_at_k_basic_hit_and_miss() -> None:
    # 4 arrivals; mask[i, j] true iff (i, j) is a labelled revisit.
    n = 4
    mask = np.zeros((n, n), dtype=bool)
    mask[0, 3] = True  # arrival 0 is the true partner of query 3
    mask[1, 2] = True  # arrival 1 is the true partner of query 2, but query 2 has no ranking below

    rankings = {
        3: [(2, 0.9), (0, 0.8), (1, 0.1)],  # true partner (0) at rank 1
        # query 2 intentionally has no ranking entry -> excluded from the denominator
    }
    results, denom = recall_at_k_wrapper(rankings, mask, near_window=0, k_values=[1, 2, 3])
    assert denom == [3]
    assert results[1]["hits"] == 0  # top-1 is arrival 2, not near 0
    assert results[2]["hits"] == 1  # top-2 includes arrival 0
    assert results[3]["hits"] == 1


def recall_at_k_wrapper(rankings, mask, near_window, k_values):
    return MODULE.recall_at_k(rankings, mask, near_window, k_values)


def test_recall_at_k_near_window_counts_a_close_but_not_exact_arrival() -> None:
    n = 10
    mask = np.zeros((n, n), dtype=bool)
    mask[2, 9] = True  # true partner of query 9 is arrival 2
    rankings = {9: [(4, 0.9)]}  # returned candidate is 4, not 2, but within near_window=5
    results, denom = MODULE.recall_at_k(rankings, mask, near_window=5, k_values=[1])
    assert denom == [9]
    assert results[1]["hits"] == 1
    results_tight, _ = MODULE.recall_at_k(rankings, mask, near_window=1, k_values=[1])
    assert results_tight[1]["hits"] == 0


def test_recall_at_k_empty_denominator_reports_none_recall() -> None:
    mask = np.zeros((5, 5), dtype=bool)  # no labelled pairs at all
    results, denom = MODULE.recall_at_k({}, mask, near_window=5, k_values=[1, 3])
    assert denom == []
    assert results[1]["recall"] is None
    assert results[1]["denominator"] == 0


# ---------------------------------------------------------------------------
# MutualNnSubsampling / select_mutual_nn_queries / select_mutual_nn_candidates
# ---------------------------------------------------------------------------


def test_select_mutual_nn_queries_always_includes_forced_range() -> None:
    subsampling = MODULE.MutualNnSubsampling(
        query_cap=3,
        candidate_cap=100,
        force_query_range=(451, 461),
        force_candidate_window=(37, 47),
    )
    denom_queries = [100, 200, 300, 455, 458, 700, 800]
    selected = MODULE.select_mutual_nn_queries(denom_queries, subsampling)
    assert 455 in selected
    assert 458 in selected


def test_select_mutual_nn_queries_respects_cap_outside_forced_range() -> None:
    subsampling = MODULE.MutualNnSubsampling(
        query_cap=2,
        candidate_cap=100,
        force_query_range=(1_000_000, 1_000_001),  # no queries fall in this range
        force_candidate_window=(0, 0),
    )
    denom_queries = list(range(0, 1000, 10))  # 100 candidates, none forced
    selected = MODULE.select_mutual_nn_queries(denom_queries, subsampling)
    assert len(selected) <= 2


def test_select_mutual_nn_candidates_always_includes_forced_window_for_forced_query() -> None:
    subsampling = MODULE.MutualNnSubsampling(
        query_cap=100,
        candidate_cap=2,
        force_query_range=(451, 461),
        force_candidate_window=(37, 47),
    )
    candidates = list(range(0, 300))
    selected = MODULE.select_mutual_nn_candidates(456, candidates, subsampling)
    assert 42 in selected  # 456 is inside the forced query range, 42 inside the forced window


def test_select_mutual_nn_candidates_does_not_force_outside_query_range() -> None:
    subsampling = MODULE.MutualNnSubsampling(
        query_cap=100,
        candidate_cap=2,
        force_query_range=(451, 461),
        force_candidate_window=(37, 47),
    )
    candidates = list(range(0, 300))
    selected = MODULE.select_mutual_nn_candidates(999, candidates, subsampling)  # not in forced range
    assert len(selected) <= 2


# ---------------------------------------------------------------------------
# load_dump (manifest.csv + .npy loading, round-tripped against a tiny
# hand-written fixture -- exercises the exact contract
# `examples/euroc_dpvo_vo_demo.rs`'s `dump_long_loop_frame` writes to)
# ---------------------------------------------------------------------------


def test_load_dump_round_trips_a_hand_written_fixture(tmp_path: Path) -> None:
    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    keypoints = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    descriptors = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    np.save(dump_dir / "000007_keypoints.npy", keypoints)
    np.save(dump_dir / "000007_descriptors.npy", descriptors)
    manifest = (
        "arrival_index,keypoint_count,descriptor_dim,keypoints_file,descriptors_file\n"
        "7,2,3,000007_keypoints.npy,000007_descriptors.npy\n"
    )
    (dump_dir / "manifest.csv").write_text(manifest, encoding="utf-8")

    frames = MODULE.load_dump(dump_dir)
    assert set(frames) == {7}
    np.testing.assert_allclose(frames[7].keypoints, keypoints)
    np.testing.assert_allclose(frames[7].descriptors, descriptors)


def test_load_dump_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MODULE.load_dump(tmp_path)


def test_load_dump_shape_mismatch_raises(tmp_path: Path) -> None:
    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    np.save(dump_dir / "000000_keypoints.npy", np.zeros((3, 2), dtype=np.float32))
    np.save(dump_dir / "000000_descriptors.npy", np.zeros((3, 4), dtype=np.float32))
    manifest = (
        "arrival_index,keypoint_count,descriptor_dim,keypoints_file,descriptors_file\n"
        "0,5,4,000000_keypoints.npy,000000_descriptors.npy\n"  # claims 5, file has 3
    )
    (dump_dir / "manifest.csv").write_text(manifest, encoding="utf-8")
    with pytest.raises(ValueError):
        MODULE.load_dump(dump_dir)
