"""Read public KITTI Tracking cuboids, calibration, and synchronized OXTS poses."""

from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.spatial.transform import Rotation


def rigid(rows):
    matrix = np.eye(4)
    matrix[:len(rows), :rows.shape[1]] = rows
    return matrix


def transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


class TrackingSequence:
    def __init__(self, root: Path, identifier: str):
        self.root, self.identifier = root, identifier

        def read(archive, directory):
            with ZipFile(root / f"{archive}.zip") as source:
                return source.read(f"training/{directory}/{identifier}.txt").decode()

        calibration = {row.split()[0].rstrip(":"): np.array(row.split()[1:], dtype=float)
                       for row in read("calib", "calib").splitlines() if row.strip()}
        velo_to_camera = rigid(calibration["R_rect"].reshape(3, 3)) @ rigid(calibration["Tr_velo_cam"].reshape(3, 4))
        self.imu_from_velo = np.linalg.inv(rigid(calibration["Tr_imu_velo"].reshape(3, 4)))
        self.camera_from_velo = velo_to_camera
        odometry = np.array([row.split()[:6] for row in read("oxts", "oxts").splitlines() if row.strip()],
                            dtype=float)
        earth_radius = 6378137.0
        scale = np.cos(np.deg2rad(odometry[0, 0]))
        latitude, longitude = np.deg2rad(odometry[:, :2]).T
        positions = np.column_stack([scale * earth_radius * longitude,
                                     scale * earth_radius * np.log(np.tan(np.pi / 4 + latitude / 2)),
                                     odometry[:, 2]])
        world_from_imu = np.repeat(np.eye(4)[None], len(odometry), axis=0)
        world_from_imu[:, :3, :3] = Rotation.from_euler("xyz", odometry[:, 3:6]).as_matrix()
        world_from_imu[:, :3, 3] = positions - positions[0]
        self.poses = world_from_imu @ self.imu_from_velo
        self.labels = defaultdict(dict)
        for row in read("label_2", "label_02").splitlines():
            fields = row.split()
            if fields and fields[2] != "DontCare":
                frame, track = map(int, fields[:2])
                self.labels[frame][track] = (fields[2], int(fields[4]), np.array(fields[10:17], dtype=float))

    def center(self, frame, track):
        box = self.labels[frame][track][2]
        center = box[3:6].copy()
        center[1] -= box[0] / 2
        return transform(center, self.poses[frame] @ np.linalg.inv(self.camera_from_velo))

    def in_box(self, world, frame, track, *, inset_m=0.0):
        height, width, length, x, y, z, yaw = self.labels[frame][track][2]
        camera = transform(world, self.camera_from_velo @ np.linalg.inv(self.poses[frame]))
        local = Rotation.from_rotvec([0, yaw, 0]).inv().apply(camera - [x, y, z])
        return ((abs(local[:, 0]) <= length / 2) & (-height <= local[:, 1]) & (local[:, 1] <= -inset_m)
                & (abs(local[:, 2]) <= width / 2))

    def scan(self, frame):
        local = np.fromfile(self.root / "training" / "velodyne" / self.identifier / f"{frame:06d}.bin", dtype="<f4").reshape(-1, 4)[:, :3]
        return local, self.poses[frame]
