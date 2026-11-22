#!/usr/bin/env python3
"""Generate the Gazebo (Harmonic) apartment world for docs/navigation.md.

    python3 ros2/visloc-ros2/gz/make_world.py --out runs/gz_nav/world

Writes `apartment.sdf`, `textures/*.png` and `calibration.json` under --out.

The floor plan is `visloc_navigation::sim::World::apartment`'s: a 10 m x 7 m
apartment, five rooms joined by 1 m doorways, a sofa, a table, a shelf and a
bed, one hallway wall left bare. Walls and furniture carry procedurally
generated textures (shapes, strokes, noise; seeded), so a camera finds
features, as on real walls. `--extra-box x0,y0,x1,y1,h` adds a box that is
not in the map, with a texture of its own (`--extra-box-plain`: a plain one).

The robot (`robot`): a 0.2 m radius differential drive (`gz-sim-diff-drive-
system` on `/cmd_vel`), with on `imu_link` (on the robot's axis, 0.5 m up)
a stereo pair (640 x 480 mono, 90 deg, 12 cm baseline, 20 Hz, 0.1 m ahead),
an IMU (200 Hz, with noise and bias), and, on `base_link`, a 2D lidar for
Nav2's obstacle layers. `gz-sim-odometry-publisher-system` gives the ground
truth. `calibration.json` is the Basalt calibration of that rig: Double
Sphere with xi = alpha = 0, an ideal pinhole.
"""

import argparse
import json
import math
import os
import random

from PIL import Image, ImageDraw, ImageFilter

WALL_H = 2.5
THICK = 0.1


def apartment():
    """Walls (a, b, height, textured) and boxes (min, max, height), as World::apartment."""
    w = []
    t, bare = True, False
    w += [((0, 0), (10, 0), t), ((10, 0), (10, 7), t), ((10, 7), (0, 7), t), ((0, 7), (0, 0), t)]
    w += [((6, 0), (6, 1.5), t), ((6, 2.5), (6, 4), t)]
    w += [((0, 4), (2, 4), t), ((3, 4), (7, 4), t), ((8, 4), (10, 4), t)]
    w += [((0, 5), (1, 5), bare), ((2, 5), (8, 5), bare), ((9, 5), (10, 5), t)]
    w += [((5, 5), (5, 7), t)]
    boxes = [((1.0, 0.4), (3.0, 1.2), 0.8), ((3.4, 2.2), (4.6, 3.0), 0.75),
             ((8.8, 0.3), (9.7, 1.6), 1.8), ((5.2, 6.0), (6.6, 7.0), 0.5)]
    return w, boxes


def texture(path, seed, width_m, height_m, px_per_m=160):
    """A feature-rich texture: coloured patches, circles, strokes and noise."""
    rng = random.Random(seed)
    w, h = max(64, int(width_m * px_per_m)), max(64, int(height_m * px_per_m))
    base = tuple(rng.randint(120, 200) for _ in range(3))
    img = Image.new('RGB', (w, h), base)
    d = ImageDraw.Draw(img)
    area = width_m * height_m
    for _ in range(int(40 * area)):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        s = rng.uniform(0.03, 0.25) * px_per_m
        colour = tuple(rng.randint(0, 255) for _ in range(3))
        kind = rng.random()
        if kind < 0.4:
            d.rectangle([x, y, x + s * rng.uniform(0.5, 2), y + s * rng.uniform(0.5, 2)], fill=colour)
        elif kind < 0.7:
            d.ellipse([x, y, x + s, y + s], fill=colour)
        else:
            d.line([x, y, x + rng.uniform(-2, 2) * s, y + rng.uniform(-2, 2) * s], fill=colour,
                   width=max(2, int(s / 6)))
    noise = Image.effect_noise((w, h), 40).convert('RGB')
    img = Image.blend(img, noise, 0.12).filter(ImageFilter.GaussianBlur(0.6))
    img.save(path)


def plain_texture(path, colour):
    Image.new('RGB', (64, 64), colour).save(path)


def material(tex_rel):
    return (f'<material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><specular>0.05 0.05 0.05 1</specular>'
            f'<pbr><metal><albedo_map>{tex_rel}</albedo_map><roughness>0.9</roughness>'
            f'<metalness>0.0</metalness></metal></pbr></material>')


def box_model(name, cx, cy, cz, sx, sy, sz, yaw, tex_rel):
    return f'''
    <model name="{name}"><static>true</static><pose>{cx} {cy} {cz} 0 0 {yaw}</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>{material(tex_rel)}</visual>
      </link>
    </model>'''


def imu_noise(stddev, bias_mean, bias_stddev):
    axis = (f'<noise type="gaussian"><mean>0</mean><stddev>{stddev}</stddev>'
            f'<bias_mean>{bias_mean}</bias_mean><bias_stddev>{bias_stddev}</bias_stddev></noise>')
    return ''.join(f'<{a}>{axis}</{a}>' for a in 'xyz')


def camera(name, y):
    return f'''
        <sensor name="{name}" type="camera">
          <pose>0.1 {y} 0 0 0 0</pose><always_on>1</always_on><update_rate>20</update_rate>
          <topic>{name}/image_raw</topic><gz_frame_id>{name}_optical</gz_frame_id>
          <camera><horizontal_fov>{math.pi / 2}</horizontal_fov>
            <image><width>640</width><height>480</height><format>L8</format></image>
            <clip><near>0.05</near><far>30</far></clip>
            <noise><type>gaussian</type><mean>0</mean><stddev>0.004</stddev></noise>
          </camera>
        </sensor>'''


def robot(x, y, yaw):
    return f'''
    <model name="robot"><pose>{x} {y} 0 0 0 {yaw}</pose>
      <link name="base_link">
        <pose>0 0 0.06 0 0 0</pose>
        <inertial><mass>8</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.16</izz></inertia></inertial>
        <collision name="c"><pose>0 0 0.12 0 0 0</pose><geometry><cylinder><radius>0.2</radius><length>0.2</length></cylinder></geometry></collision>
        <visual name="v"><pose>0 0 0.12 0 0 0</pose><geometry><cylinder><radius>0.2</radius><length>0.2</length></cylinder></geometry>
          <material><diffuse>0.2 0.3 0.7 1</diffuse></material></visual>
        <collision name="caster_f"><pose>0.15 0 -0.03 0 0 0</pose><geometry><sphere><radius>0.03</radius></sphere></geometry>
          <surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface></collision>
        <collision name="caster_b"><pose>-0.15 0 -0.03 0 0 0</pose><geometry><sphere><radius>0.03</radius></sphere></geometry>
          <surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface></collision>
        <sensor name="lidar" type="gpu_lidar">
          <pose>0 0 0.2 0 0 0</pose><update_rate>10</update_rate><topic>scan</topic>
          <gz_frame_id>base_link</gz_frame_id>
          <lidar><scan><horizontal><samples>180</samples><resolution>1</resolution>
            <min_angle>-1.5708</min_angle><max_angle>1.5708</max_angle></horizontal></scan>
            <range><min>0.1</min><max>5.0</max><resolution>0.01</resolution></range>
            <noise><type>gaussian</type><mean>0</mean><stddev>0.01</stddev></noise></lidar>
        </sensor>
      </link>
      <link name="left_wheel"><pose>0 0.175 0.05 -1.5708 0 0</pose>
        <inertial><mass>0.5</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz></inertia></inertial>
        <collision name="c"><geometry><cylinder><radius>0.05</radius><length>0.03</length></cylinder></geometry></collision>
        <visual name="v"><geometry><cylinder><radius>0.05</radius><length>0.03</length></cylinder></geometry></visual></link>
      <link name="right_wheel"><pose>0 -0.175 0.05 -1.5708 0 0</pose>
        <inertial><mass>0.5</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz></inertia></inertial>
        <collision name="c"><geometry><cylinder><radius>0.05</radius><length>0.03</length></cylinder></geometry></collision>
        <visual name="v"><geometry><cylinder><radius>0.05</radius><length>0.03</length></cylinder></geometry></visual></link>
      <joint name="left_wheel_joint" type="revolute"><parent>base_link</parent><child>left_wheel</child>
        <axis><xyz>0 0 1</xyz></axis></joint>
      <joint name="right_wheel_joint" type="revolute"><parent>base_link</parent><child>right_wheel</child>
        <axis><xyz>0 0 1</xyz></axis></joint>
      <link name="imu_link"><pose>0 0 0.5 0 0 0</pose>
        <inertial><mass>0.1</mass><inertia><ixx>0.0001</ixx><iyy>0.0001</iyy><izz>0.0001</izz></inertia></inertial>
        <sensor name="imu" type="imu"><always_on>1</always_on><update_rate>200</update_rate><topic>imu</topic>
          <gz_frame_id>imu_link</gz_frame_id>
          <imu><angular_velocity>{imu_noise(0.002, 0.0, 0.0001)}</angular_velocity>
            <linear_acceleration>{imu_noise(0.02, 0.0, 0.001)}</linear_acceleration></imu>
        </sensor>{camera('cam0', 0.06)}{camera('cam1', -0.06)}
      </link>
      <joint name="imu_joint" type="fixed"><parent>base_link</parent><child>imu_link</child></joint>
      <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
        <left_joint>left_wheel_joint</left_joint><right_joint>right_wheel_joint</right_joint>
        <wheel_separation>0.35</wheel_separation><wheel_radius>0.05</wheel_radius>
        <max_linear_acceleration>1.0</max_linear_acceleration><max_angular_acceleration>3.0</max_angular_acceleration>
        <topic>cmd_vel</topic><odom_topic>wheel_odom</odom_topic><tf_topic>wheel_tf</tf_topic>
        <frame_id>odom</frame_id><child_frame_id>base_link</child_frame_id><odom_publish_frequency>20</odom_publish_frequency>
      </plugin>
      <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
        <odom_frame>map</odom_frame><robot_base_frame>base_link</robot_base_frame>
        <odom_topic>ground_truth</odom_topic><odom_publish_frequency>20</odom_publish_frequency>
        <dimensions>3</dimensions><tf_topic>ground_truth_tf</tf_topic>
      </plugin>
      <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher"/>
    </model>'''


def calibration():
    """Basalt calibration: IMU frame = imu_link (x fwd, y left, z up)."""
    # Optical frame: z forward (= imu x), x right (= -imu y), y down (= -imu z).
    # Rotation imu <- cam with columns (cam x, cam y, cam z) in imu axes:
    # [[0, 0, 1], [-1, 0, 0], [0, -1, 0]] -> quaternion (x, y, z, w):
    q = (-0.5, 0.5, -0.5, 0.5)
    f = 320.0 / math.tan(math.pi / 4)
    cams = [{"px": 0.1, "py": y, "pz": 0.0, "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3]}
            for y in (0.06, -0.06)]
    intr = {"camera_type": "ds", "intrinsics": {"fx": f, "fy": f, "cx": 320.0, "cy": 240.0,
                                                "xi": 0.0, "alpha": 0.0}}
    ident = {"px": 0.0, "py": 0.0, "pz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    return {"value0": {
        "T_imu_cam": cams, "intrinsics": [intr, intr], "resolution": [[640, 480], [640, 480]],
        "vignette": [], "calib_accel_bias": [0.0] * 9, "calib_gyro_bias": [0.0] * 12,
        "imu_update_rate": 200.0,
        "accel_noise_std": [0.02, 0.02, 0.02], "gyro_noise_std": [0.002, 0.002, 0.002],
        "accel_bias_std": [0.001, 0.001, 0.001], "gyro_bias_std": [0.0001, 0.0001, 0.0001],
        "T_mocap_world": ident, "T_imu_marker": ident,
        "mocap_time_offset_ns": 0, "mocap_to_imu_offset_ns": 0, "cam_time_offset_ns": 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--start', default='1.5,2.5,0', help='robot start x,y,yaw')
    parser.add_argument('--extra-box', default='', help='x0,y0,x1,y1,h: an obstacle not in the map')
    parser.add_argument('--extra-box-plain', action='store_true',
                        help='give the extra box no texture (a camera sees no features on it)')
    parser.add_argument('--real-time-factor', type=float, default=1.0,
                        help='target speed of the simulation (below 1 slows it down)')
    args = parser.parse_args()
    tex_dir = os.path.join(args.out, 'textures')
    os.makedirs(tex_dir, exist_ok=True)
    walls, boxes = apartment()
    models = []
    for i, (a, b, textured) in enumerate(walls):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        name = f'wall{i}.png'
        if textured:
            texture(os.path.join(tex_dir, name), args.seed * 1000 + i, length, WALL_H)
        else:
            plain_texture(os.path.join(tex_dir, name), (205, 200, 190))
        models.append(box_model(f'wall{i}', (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, WALL_H / 2,
                                length + THICK, THICK, WALL_H, yaw, f'textures/{name}'))
    for i, ((x0, y0), (x1, y1), h) in enumerate(boxes):
        name = f'box{i}.png'
        texture(os.path.join(tex_dir, name), args.seed * 1000 + 100 + i, max(x1 - x0, y1 - y0), h)
        models.append(box_model(f'furniture{i}', (x0 + x1) / 2, (y0 + y1) / 2, h / 2,
                                x1 - x0, y1 - y0, h, 0, f'textures/{name}'))
    if args.extra_box:
        x0, y0, x1, y1, h = (float(v) for v in args.extra_box.split(','))
        if args.extra_box_plain:
            plain_texture(os.path.join(tex_dir, 'extra.png'), (170, 60, 50))
        else:
            texture(os.path.join(tex_dir, 'extra.png'), args.seed * 1000 + 500,
                    max(x1 - x0, y1 - y0), h)
        models.append(box_model('unmapped_box', (x0 + x1) / 2, (y0 + y1) / 2, h / 2,
                                x1 - x0, y1 - y0, h, 0, 'textures/extra.png'))
    floor = 'floor.png'
    texture(os.path.join(tex_dir, floor), args.seed * 1000 + 999, 10, 7, px_per_m=60)
    models.append(f'''
    <model name="floor"><static>true</static><pose>5 3.5 -0.05 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>12 9 0.1</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>10.2 7.2 0.1</size></box></geometry>{material('textures/' + floor)}</visual>
      </link>
    </model>''')
    sx, sy, syaw = (float(v) for v in args.start.split(','))
    models.append(robot(sx, sy, syaw))
    sdf = f'''<?xml version="1.0"?>
<!-- Generated by ros2/visloc-ros2/gz/make_world.py; see docs/navigation.md. -->
<sdf version="1.9">
  <world name="apartment">
    <physics name="1ms" type="ignored"><max_step_size>0.001</max_step_size>
      <real_time_factor>{args.real_time_factor}</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <scene><ambient>0.9 0.9 0.9 1</ambient><background>0.8 0.8 0.8 1</background><shadows>false</shadows></scene>
    <light type="directional" name="sun"><cast_shadows>false</cast_shadows><direction>-0.4 0.2 -0.9</direction>
      <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.1 0.1 0.1 1</specular></light>
    <light type="point" name="lamp"><pose>5 3.5 2.3 0 0 0</pose><diffuse>0.5 0.5 0.5 1</diffuse>
      <attenuation><range>20</range><linear>0.05</linear><constant>0.5</constant><quadratic>0.0</quadratic></attenuation></light>
{''.join(models)}
  </world>
</sdf>
'''
    with open(os.path.join(args.out, 'apartment.sdf'), 'w', encoding='utf-8') as f:
        f.write(sdf)
    with open(os.path.join(args.out, 'calibration.json'), 'w', encoding='utf-8') as f:
        json.dump(calibration(), f, indent=2)
    print(f"wrote {args.out}/apartment.sdf ({len(walls)} walls, {len(boxes)} boxes)")


if __name__ == '__main__':
    main()
