# Public Data Demo

The README localization image uses public COLMAP South Building data rather than generated artwork.

## Source

- Dataset: COLMAP South Building example dataset
- Download used for the demo: `https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip`
- Input images used for the small reconstruction: `P1180141.JPG` through `P1180149.JPG`

The original archive is not committed because it is about 400 MiB. The repository only stores the derived README assets.

## Reconstruction

The demo was built from a small subset of the public images:

1. Download `south-building.zip`.
2. Extract the first 9 South Building images.
3. Resize the images to 1024x768 for a compact README-oriented reconstruction.
4. Run `pycolmap.extract_features`, `pycolmap.match_exhaustive`, and `pycolmap.incremental_mapping`.
5. Export the sparse model to COLMAP text format.

The generated model registered 9 cameras and reconstructed 1,428 sparse 3D points.

## README Assets

- `docs/assets/south-building-query.jpg`: real query image from the public dataset subset.
- `docs/assets/south-building-localization.png`: real query image with observed 2D points next to the sparse SfM point cloud and recovered camera frustum.
- `docs/assets/south-building-localization.gif`: animated README version of the same public-data localization visualization.

The visualization is intentionally a lightweight README artifact. It is not a benchmark and does not claim full SLAM; it demonstrates the library direction: Visual Localization first, using a reusable visual map and a query image pose.
