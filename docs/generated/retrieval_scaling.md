# Retrieval scaling: vocab-tree vs flat-VLAD pair generation

Evidence for `docs/colmap_port_plan.md`'s M4 acceptance criterion (sub-linear
retrieval at thousands-of-images scale). Registry manifest:
`benchmarks/registry/runs/retrieval/retrieval-scale-synthetic-20260826.json`.
Reproduce with:

```sh
cargo run --release --example retrieval_scale_benchmark -- --sizes 250,500,1000,2000
```

## Protocol

- Deterministic synthetic "places" corpus: 200 clusters in 256-dim descriptor
  space, 64 noisy descriptors per image; every image belongs to one place.
- Fixed vocabularies trained once on a separate 512-image pool (never re-trained
  per size): flat arm = k-means VLAD (256 words); tree arm = hierarchical
  k-means (branching 16, depth 3 → up to 4096 leaf words) + Hamming-embedding
  inverted index (`visloc_vision::vocab_tree::VocabTree`).
- Both arms propose top-K=10 pairs for every image. Reported per corpus size:
  wall clock, exact comparison counts (flat) or measured work counters
  (`QueryWorkStats`, tree), and same-place recall@10.

## Results (Linux x86_64, CPU-only release build, 2026-08-26)

| images | flat scan (s) | flat comparisons | flat recall@10 | tree query (s) | tree entries visited | tree recall@10 |
|---|---|---|---|---|---|---|
| 250 | 14.006 | 62,500 | 1.000 | 28.453 | 1,292,589 (+39,360,000 leaf scans) | 0.960 |
| 500 | 54.312 | 250,000 | 1.000 | 56.571 | 4,898,092 (+78,720,000 leaf scans) | 0.962 |
| 1000 | 208.881 | 1,000,000 | 1.000 | 99.540 | 18,853,327 (+157,440,000 leaf scans) | 0.957 |
| 2000 | 828.280 | 4,000,000 | 1.000 | 196.923 | 76,011,190 (+314,880,000 leaf scans) | 0.957 |

## Reading

- **Flat scan is quadratic**: comparisons grow exactly N(N−K) and wall clock
  ~3.9x per doubling.
- **Tree is near-linear**: measured inverted-file entries visited grow
  ~3.9x per doubling — i.e. linearly with the corpus — while its
  corpus-independent leaf-scan cost stays fixed at the fixed-vocabulary width;
  wall clock grows ~1.9x per doubling here because leaf assignment dominates
  at these sizes.
- **Crossover**: the tree wins from ≥1000 images and is **4.2x faster at
  2000** (196.9 s vs 828.3 s) with near-parity same-place recall (0.957 vs
  1.000). At 250–500 images the fixed 4096-word exact leaf scan dominates and
  the flat arm is faster — consistent with COLMAP scaling vocabulary size
  with corpus scale.
- **Machine-independent invariant**: the deterministic unit test
  `query_work_grows_linearly_while_flat_pairwise_is_quadratic`
  (crates/vision/src/vocab_tree/index.rs) pins entries-visited growth <10x
  across an 8x corpus growth where the replaced flat pairwise scan is exactly
  64x.

Honest scope: synthetic descriptors measure *cost scaling* cleanly but are not
a real-photo quality benchmark; recall here is same-place retrieval of
clustered synthetic descriptors, not geometric verification outcomes on a real
collection.
