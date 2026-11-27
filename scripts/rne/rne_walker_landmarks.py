#!/usr/bin/env python3
"""Landmarks of a saved map lying near a walker at an image that observes them.

  RNE_WORK=DIR python3 scripts/rne/rne_walker_landmarks.py MAP EPISODE [people.csv]

In DIR: MAP_slam/trajectory_online.tum and MAP_map/ (from
basalt_euroc_online_slam_demo --out-dir MAP_slam --save-map MAP_map) and
EPISODE/ (an RNE example 113 export with mav0/state_groundtruth_estimate0,
and people.csv unless given). Counts the landmarks within 0.45 m
(horizontally, 0-2 m high) of a walker at any image that observes them.
Background points near a walker's path count too: compare with a map built
without walkers against the same people.csv.
"""
import sys, csv, bisect
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate_map_localization import load_tum_seconds, load_euroc_ground_truth, fit_to_ground_truth
W=__import__('os').environ['RNE_WORK']; m=sys.argv[1]; ep=sys.argv[2]
R,t=fit_to_ground_truth(load_tum_seconds(f'{W}/{m}_slam/trajectory_online.tum'), load_euroc_ground_truth(f'{W}/{ep}/mav0/state_groundtruth_estimate0/data.csv'), 10_000_000)
img_ts={}
for r in csv.DictReader(open(f'{W}/{m}_map/images.csv')):
    img_ts[int(r['image_id'])]=int(r['timestamp_ns'])
people={}
for r in csv.DictReader(open(sys.argv[3] if len(sys.argv)>3 else f'{W}/{ep}/people.csv')):
    people.setdefault(int(r['timestamp_ns']),[]).append((float(r['x']),float(r['z'])))
pk=sorted(people)
on=n=0; near=0
for l in open(f'{W}/{m}_map/sparse/points3D.txt'):
    if l.startswith('#'): continue
    f=l.split(); n+=1
    p=R@np.array(list(map(float,f[1:4])))+t
    imgs=set(int(x) for x in f[8::2])
    hit=False
    for i in imgs:
        ts=img_ts.get(i)
        if ts is None: continue
        k=pk[min(bisect.bisect_left(pk,ts),len(pk)-1)]
        for (x,z) in people[k]:
            if np.hypot(p[0]-x,p[2]-z)<0.45 and -0.1<p[1]<2.0: hit=True
    on+=hit
print(f'{m}: {n} landmarks, {on} on a walker ({100*on/n:.1f}%) at an observing image')
