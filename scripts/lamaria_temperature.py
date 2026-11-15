#!/usr/bin/env python3
"""The SLAM cameras' temperature over a LaMAria recording, from a sparse copy of its VRS file.

    python scripts/lamaria_vrs_fetch.py URL work/seq.vrs --sample-streams 1201-1,1201-2 --every 200
    python scripts/lamaria_temperature.py work/seq.vrs work/seq_temperature.csv --every 200

Each SLAM camera's image record carries the sensor's temperature (whole degrees Celsius; the IMU's is NaN). The
sparse copy holds every --every-th record of each camera (10 s at 20 Hz for 200) and the first and the last; this
writes `timestamp_ns,cam0_c,cam1_c` (capture time on the device clock, as the extracted images), the two
cameras' records paired by order. Needs projectaria_tools.
"""
import argparse
import csv


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("vrs")
    p.add_argument("out")
    p.add_argument("--every", type=int, default=200, help="as given to lamaria_vrs_fetch.py --every")
    a = p.parse_args()
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.stream_id import StreamId
    provider = data_provider.create_vrs_data_provider(a.vrs)
    rows = {}
    for c, sid in enumerate(("1201-1", "1201-2")):
        stream = StreamId(sid)
        n = provider.get_num_data(stream)
        idx = sorted({0, n - 1, *range(a.every - 1, n, a.every)})
        for k, i in enumerate(idx):
            _, rec = provider.get_image_data_by_index(stream, i)
            row = rows.setdefault(k, [rec.capture_timestamp_ns, None, None])
            row[1 + c] = rec.temperature
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", "cam0_c", "cam1_c"])
        for k in sorted(rows):
            w.writerow(rows[k])
    temps = [r[1] for r in rows.values()]
    print(f"wrote {a.out}: {len(rows)} rows, cam0 {min(temps)}..{max(temps)} C")


if __name__ == "__main__":
    main()
