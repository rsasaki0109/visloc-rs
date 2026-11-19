# Outdoor building changes and visual seasonal bridges

## ROVER RGB-only submap integration

The whole-traverse experiment supplied 114 summer and 116 winter images, sampled every 3 seconds. Its image-only reconstruction split into 15 single-season submaps. This experiment attempts to connect them without using the total-station trajectories.

The direct variant associates photometrically matched pixels to existing 3D tracks. It verifies 3D-to-3D similarity transforms on held-out landmarks and also tries PnP localization into the opposite map with separate reprojection and pose checks. Those direct attempts found no accepted summer-to-winter transform. The short-window variant independently bundle-adjusts both seasons in each of four selected windows, then ties those local reconstructions to the original submaps by shared RGB exposures. Candidate offsets come from the image-matching evidence. The similarity estimate uses camera positions and orientations from three or more shared views and checks an unused split of the shared exposures.

| Window | Summer and winter images jointly registered | Connected to both broader submaps |
| --- | ---: | --- |
| House | 7 + 7 | Yes |
| Shed | 1 + 5 | Insufficient views |
| North | 1 + 1 | Insufficient views |
| Lawn | 1 + 5 | Insufficient views |

The house bridge passed held-out pose checks on 2/2 exposures for each season. It joins two broader submaps with **22 summer and 21 winter poses** and saves 713 sparse points in the combined visual gauge. The public prism trajectory is used only after joining to set a scoring scale from the summer path and to assess distances on the winter path. The winter prism-distance RMSE is 0.106 m on 14 synchronized frames outside the bridge window, over a 6.71 m reference diameter. This checks local metric consistency after alignment. An absolute cross-season pose residual requires common surveyed control or independent 3D landmark labels.

Image-only integration now covers the house sector, with repeated observations across several laps. Extending that component across the rest of the site requires further verified visual bridges. [Local bridge results](../benchmarks/rover/visual_bridge_results.json) · [Direct-link diagnostics](../benchmarks/rover/visual_direct_submaps_results.json)

## Rotterdam building changes

| Chronological event | Raw height difference precision / recall | Saved neighborhood update precision / recall |
| --- | ---: | ---: |
| Building additions | 84.6% / 98.6% | 84.8% / 98.6% |
| Roof removal | 75.0% / 100.0% | 100.0% / 100.0% |

The saved update found 417 of 423 added columns and all 60 removed columns. False changes on the 946 labeled stable columns fell from 96 to 75. The remaining 75 are concentrated on one stable roof with an elevation offset near the 3 m threshold. The sample covers two additions and one demolition patch.

The evaluator reads public AHN airborne LiDAR in the Dutch RD coordinate system. It builds a 1 m surface grid from the 90th percentile of four or more returns per column. When both epochs have support, a height difference of at least 3 m votes for a signed change. The guarded update also requires six same-sign columns in a 3 × 3 neighborhood. It writes the new elevations and change mask and reopens each snapshot for scoring. Missing observations in a labeled changed column count as missed detections.

The manual reference patches cover the added Depot Boijmans Van Beuningen and Zalmhaven building footprints, a removed roof along Dr Molewaterplein at Erasmus MC, and stable controls. PDOK aerial photographs from 2016 and 2023 support visual labeling, with BAG footprints for additions and an independently acquired 5 m AHN4 surface for the removed roof. The AHN4 surface is used for label review. The additions compare AHN2 to AHN5. The removed roof compares AHN3 to AHN5 because its footprint in the earlier AHN2 scan did not show the 2016 roof. The masks cover selected interiors. These numbers measure real outdoor height-map changes with registered airborne scans and do not measure ground-robot pose error or full 3D occupancy. [Labels](../benchmarks/ahn/rotterdam_changes.json) · [Metrics](../benchmarks/ahn/rotterdam_results.json) · [AHN data](https://www.ahn.nl/open-data) · [GeoTiles sub-tiles and attribution](https://geotiles.citg.tudelft.nl/tiles/html/37HN1.html) · [PDOK aerial imagery](https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?service=WMS&request=GetCapabilities)

## Reproduce

Use the ROVER samples, pretrained weights, site results and COLMAP submaps from the [site evaluation](rover_site_and_clearance_revisits.md). Build the short-window models once and reuse them by omitting the `--efficient-loftr` option on subsequent runs.

```bash
export OPENBLAS_NUM_THREADS=1
python scripts/eval_rover_visual_submaps.py --summer "$summer" --winter "$winter" \
  --calibration "$calibration" --efficient-loftr "$loftr" --max-side 448 \
  --site-results runs/site_alignment.json --site-models runs/site_models --out runs/rover_direct_links.json
python scripts/eval_rover_local_bridges.py --summer "$summer" --winter "$winter" \
  --calibration "$calibration" --efficient-loftr "$loftr" \
  --site-results runs/site_alignment.json --site-models runs/site_models \
  --windows benchmarks/rover/local_bridge_windows.json --local-models runs/rover_local_models \
  --map-out runs/rover_visual_joined.npz --out runs/rover_visual_bridges.json

python -m pip install laspy==2.7.0 lazrs==0.9.0
mkdir -p runs/rotterdam
for item in 2:02 2:03 2:08 3:02 5:02 5:03 5:08; do
  epoch=${item%:*}
  tile=37HN1_${item#*:}
  curl -fL "https://geotiles.citg.tudelft.nl/AHN${epoch}_T/${tile}.LAZ" \
    -o "runs/rotterdam/ahn${epoch}-${tile}.laz"
done
(cd runs/rotterdam && sha256sum -c ../../benchmarks/ahn/rotterdam_inputs.sha256)
python scripts/eval_ahn_building_change.py --data runs/rotterdam \
  --annotations benchmarks/ahn/rotterdam_changes.json \
  --maps-out runs/rotterdam_updated --out runs/rotterdam_results.json
```

The follow-up scores full 3D occupancy updates with KITTI ground LiDAR and measures whether denser image sampling can extend the ROVER visual bridge. [Ground LiDAR and dense visual windows](ground_lidar_and_dense_visual_windows.md)
