#!/usr/bin/env python3
"""The GPS fixes of an Aria VRS file as CSV: t_ns (device clock), lat, lon, alt, accuracy (m), speed, provider.

    python scripts/lamaria_gps_extract.py work/sequence_5_5/sequence_5_5.vrs work/gps/sequence_5_5.csv

The file can be the whole recording or the sparse copy of lamaria_vrs_fetch.py. Fixes without a position (accuracy
above 1 km, or latitude and longitude both zero) are left out. Needs projectaria_tools.
"""
import sys

from projectaria_tools.core import data_provider


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    p = data_provider.create_vrs_data_provider(sys.argv[1])
    sid = p.get_stream_id_from_label("gps")
    n = p.get_num_data(sid)
    kept = 0
    with open(sys.argv[2], "w") as f:
        f.write("t_ns,lat,lon,alt,accuracy,speed,provider\n")
        for i in range(n):
            d = p.get_gps_data_by_index(sid, i)
            if d.accuracy > 1000 or (d.latitude == 0 and d.longitude == 0):
                continue
            f.write(f"{d.capture_timestamp_ns},{d.latitude:.9f},{d.longitude:.9f},{d.altitude:.3f},"
                    f"{d.accuracy:.2f},{d.speed:.3f},{d.provider}\n")
            kept += 1
    print(f"{sys.argv[1]}: {kept} of {n} fixes")


if __name__ == "__main__":
    main()
