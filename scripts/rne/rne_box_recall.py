#!/usr/bin/env python3
"""Detector boxes vs RNE's true walker boxes.

  python3 scripts/rne/rne_box_recall.py GT.csv DET.csv [MIN_AREA_PX]

GT.csv from rne_walker_boxes.py, DET.csv from yolox_track_demo --boxes-csv.
GT boxes ignore occlusion (a walker behind a wall still has one), so recall
is counted on GT boxes at least MIN_AREA px (2000 by default), each found
if a person detection overlaps it with IoU >= 0.3 (one detection can find
two walkers that overlap). A detection overlapping no GT box (IoU < 0.1)
counts as false."""
import csv, sys
from collections import defaultdict
def load(p, cls=None):
    d=defaultdict(list)
    for r in csv.DictReader(open(p)):
        if cls is not None and r['class_id']!=cls: continue
        d[int(r['timestamp_ns'])].append(tuple(float(r[k]) for k in ('x_min','y_min','x_max','y_max')))
    return d
def iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0])); h=max(0,min(a[3],b[3])-max(a[1],b[1]))
    i=w*h; return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i+1e-9)
gt=load(sys.argv[1]); det=load(sys.argv[2], '0')
MIN=float(sys.argv[3]) if len(sys.argv)>3 else 2000
n=hit=0; fp=ndet=0
for t,boxes in gt.items():
    for g in boxes:
        if (g[2]-g[0])*(g[3]-g[1])<MIN: continue
        n+=1
        if any(iou(g,d)>=0.3 for d in det.get(t,[])): hit+=1
for t,ds in det.items():
    for d in ds:
        ndet+=1
        if not any(iou(g,d)>=0.1 for g in gt.get(t,[])): fp+=1
print(f'GT boxes >= {MIN:.0f} px: {n}, found (IoU>=0.3) {hit} = {100*hit/max(n,1):.0f}%; person detections {ndet}, matching no GT box {fp}')
