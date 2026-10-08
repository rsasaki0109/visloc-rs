use super::*;

#[test]
#[ignore = "frame39 native current-translation schedule probe"]
fn m11_frame39_track1827_camera_translation_schedule_probe() {
    fn pose(words: [u32; 7]) -> F32Pose {
        F32Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(words[3]),
                f32::from_bits(words[0]),
                f32::from_bits(words[1]),
                f32::from_bits(words[2]),
            )),
            translation: Vector3::new(
                f32::from_bits(words[4]),
                f32::from_bits(words[5]),
                f32::from_bits(words[6]),
            ),
        }
    }

    let target_camera_from_imu = pose([
        0x3b19188b, 0xbc55012e, 0xbf33d4ed, 0x3f362af6, 0xbd27e3b1, 0xbc8044de, 0xbb7a8e6e,
    ]);
    let target_imu_from_anchor_imu = pose([
        0xbae89e11, 0x3cacb89c, 0xbb36a8c7, 0x3f7ff113, 0x3de32d30, 0xbb8d4bec, 0xbd09edaa,
    ]);
    let anchor_t_imu_cam = pose([
        0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e, 0xbc896b48, 0xbd8d2fdc, 0x3ba86617,
    ]);
    let captured_prefix = pose([
        0x3c826a39, 0x3b559009, 0xbf344ab5, 0x3f35b244, 0xbd30c0af, 0xbe022c22, 0xbd130180,
    ]);
    let expected = [0xbde5ffa4, 0xbde359ce, 0xbd013192];

    let exact = |value: Vector3<f32>| {
        [value.x.to_bits(), value.y.to_bits(), value.z.to_bits()]
            .iter()
            .zip(expected)
            .filter(|(left, right)| **left == *right)
            .count()
    };
    let bits = |value: Vector3<f32>| {
        format!(
            "{:08x} {:08x} {:08x}",
            value.x.to_bits(),
            value.y.to_bits(),
            value.z.to_bits()
        )
    };

    for (prefix_name, prefix_action) in [
        (
            "generic",
            sophus_rotate_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu.translation,
            ),
        ),
        (
            "step_packet",
            sophus_rotate_step_packet_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu.translation,
            ),
        ),
    ] {
        let prefix = F32Pose {
            rotation: sophus_quat_product_camera_prefix_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu.rotation,
            ),
            translation: prefix_action + target_camera_from_imu.translation,
        };
        for (suffix_name, suffix_action) in [
            (
                "generic",
                sophus_rotate_f32(prefix.rotation, anchor_t_imu_cam.translation),
            ),
            (
                "step_packet",
                sophus_rotate_step_packet_f32(prefix.rotation, anchor_t_imu_cam.translation),
            ),
        ] {
            let result = suffix_action + prefix.translation;
            println!(
                "m11_frame39_translation prefix={prefix_name} suffix={suffix_name} prefix_t={} result={} exact={}/3",
                bits(prefix.translation), bits(result), exact(result)
            );
        }
    }

    for (suffix_name, suffix_action) in [
        (
            "generic",
            sophus_rotate_f32(captured_prefix.rotation, anchor_t_imu_cam.translation),
        ),
        (
            "step_packet",
            sophus_rotate_step_packet_f32(captured_prefix.rotation, anchor_t_imu_cam.translation),
        ),
    ] {
        let result = suffix_action + captured_prefix.translation;
        println!(
            "m11_frame39_translation captured_prefix suffix={suffix_name} result={} exact={}/3",
            bits(result),
            exact(result)
        );
    }
}
