// Deterministic COLMAP GR6P oracle fixture generator.
//
// This helper only calls COLMAP's public GR6PEstimator API; it does not copy
// solver implementation code.  The COLMAP headers/library and the PoseLib
// implementation used by that API are BSD-3-Clause software.  Keep the
// upstream attribution and disclaimer when redistributing generated fixtures:
// https://github.com/colmap/colmap and https://github.com/PoseLib/PoseLib.

#include "colmap/estimators/solvers/generalized_relative_pose.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

// The pinned COLMAP runtime image ships libPoseLib.a but not PoseLib headers.
// This declaration mirrors PoseLib's BSD-3-Clause CameraPose ABI so that the
// COLMAP conversion symbol can be provided locally without importing any
// additional source or build dependency.  The upstream PoseLib declaration
// (PoseLib/camera_pose.h) is `alignas(32)` with `Eigen::Vector4d q` followed by
// `Eigen::Vector3d t`; q is ordered (w, x, y, z).  The pinned COLMAP
// poselib_utils conversion reads exactly q(0..3) and t, and the pinned archive
// exports the matching gen_relpose_6pt/CameraPose ABI.  COLMAP's GR6P entry
// point returns this type only through ConvertPoseLibPoseToRigid3d.
namespace poselib {
struct alignas(32) CameraPose {
  Eigen::Vector4d q;
  Eigen::Vector3d t;
};
}  // namespace poselib

namespace colmap {

Rigid3d ConvertPoseLibPoseToRigid3d(const poselib::CameraPose& pose) {
  // PoseLib stores quaternions as (w, x, y, z); Eigen::Quaterniond has the
  // same constructor ordering.
  return Rigid3d(Eigen::Quaterniond(pose.q(0), pose.q(1), pose.q(2), pose.q(3)),
                 pose.t);
}

// GR6P's Residuals method references this geometry helper.  Defining the
// compact equivalent here keeps the oracle link limited to the solver and
// PoseLib archives instead of pulling all of COLMAP's geometry library.
Eigen::Matrix3d EssentialMatrixFromPose(const Rigid3d& cam2_from_cam1) {
  const Eigen::Vector3d t = cam2_from_cam1.translation();
  Eigen::Matrix3d tx;
  tx << 0.0, -t.z(), t.y(), t.z(), 0.0, -t.x(), -t.y(), t.x(), 0.0;
  return tx * cam2_from_cam1.rotation().toRotationMatrix();
}

// The solver object contains a fatal-check cold path.  The fixture supplies
// the tiny utility symbol so that linking does not require COLMAP's full
// util archive; the normal path never calls it.
const char* __GetConstFileBaseName(const char* path) {
  const char* slash = std::strrchr(path, '/');
  return slash == nullptr ? path : slash + 1;
}

}  // namespace colmap

namespace {

using colmap::GR6PEstimator;
using colmap::GRNPObservation;
using colmap::Rigid3d;

struct Correspondence {
  int source_sensor;
  int target_sensor;
  Eigen::Vector3d point_in_rig1;
  Eigen::Vector3d point_in_rig2;
  GRNPObservation source;
  GRNPObservation target;
};

bool IsFinite(const Eigen::Vector3d& value) {
  return value.array().isFinite().all();
}

bool IsFinite(const Rigid3d& value) {
  return IsFinite(value.translation()) &&
         value.rotation().coeffs().array().isFinite().all();
}

double RotationErrorRad(const Rigid3d& estimate, const Rigid3d& truth) {
  const Eigen::Matrix3d relative =
      estimate.rotation().toRotationMatrix().transpose() *
      truth.rotation().toRotationMatrix();
  const double cosine =
      std::clamp((relative.trace() - 1.0) * 0.5, -1.0, 1.0);
  return std::acos(cosine);
}

void WriteVector(std::ostream& output, const Eigen::Vector3d& value) {
  output << '[' << value.x() << ',' << value.y() << ',' << value.z() << ']';
}

void WriteRigid(std::ostream& output, const Rigid3d& value) {
  const Eigen::Quaterniond rotation = value.rotation();
  output << "{\"rotation_wxyz\":[" << rotation.w() << ',' << rotation.x()
         << ',' << rotation.y() << ',' << rotation.z()
         << "],\"translation\":";
  WriteVector(output, value.translation());
  output << '}';
}

Rigid3d CameraFromCenter(const Eigen::Vector3d& center) {
  return Rigid3d(Eigen::Quaterniond::Identity(), -center);
}

Correspondence MakeCorrespondence(
    const Rigid3d& rig2_from_rig1,
    const std::vector<Rigid3d>& cameras,
    const int source_sensor,
    const int target_sensor,
    const Eigen::Vector3d& point_in_rig1) {
  Correspondence correspondence{source_sensor,
                                target_sensor,
                                point_in_rig1,
                                rig2_from_rig1 * point_in_rig1,
                                {},
                                {}};
  correspondence.source.cam_from_rig = cameras[source_sensor];
  correspondence.target.cam_from_rig = cameras[target_sensor];
  correspondence.source.ray_in_cam =
      (cameras[source_sensor] * correspondence.point_in_rig1).normalized();
  correspondence.target.ray_in_cam =
      (cameras[target_sensor] * correspondence.point_in_rig2).normalized();
  return correspondence;
}

void WriteObservation(std::ostream& output,
                      const int sensor,
                      const GRNPObservation& observation) {
  output << "{\"sensor_id\":" << sensor << ",\"cam_from_rig\":";
  WriteRigid(output, observation.cam_from_rig);
  output << ",\"ray_in_cam\":";
  WriteVector(output, observation.ray_in_cam);
  output << '}';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 2) {
    std::cerr << "usage: " << argv[0] << " [output.json]\n";
    return 2;
  }

  std::ostream* output = &std::cout;
  std::ofstream output_file;
  if (argc == 2) {
    output_file.open(argv[1]);
    if (!output_file) {
      std::cerr << "cannot open output: " << argv[1] << '\n';
      return 2;
    }
    output = &output_file;
  }

  // A non-pure-translation metric motion and three distinct camera centers
  // make the fixture a genuine generalized relative-pose problem.
  const Eigen::Vector3d motion_axis(0.31, -0.47, 0.19);
  const Eigen::Quaterniond motion_rotation(
      Eigen::AngleAxisd(0.31, motion_axis.normalized()));
  const Rigid3d rig2_from_rig1(motion_rotation,
                               Eigen::Vector3d(0.72, -0.21, 0.28));
  const std::vector<Rigid3d> cameras = {
      CameraFromCenter(Eigen::Vector3d(0.00, 0.00, 0.00)),
      CameraFromCenter(Eigen::Vector3d(0.46, 0.08, 0.03)),
      CameraFromCenter(Eigen::Vector3d(-0.22, 0.39, 0.11)),
  };

  const std::vector<int> source_sensors = {0, 1, 2, 0, 1, 2};
  const std::vector<int> target_sensors = {1, 2, 0, 2, 0, 1};
  const std::vector<Eigen::Vector3d> points = {
      {4.10, 0.20, 7.30},  {3.15, -1.20, 8.70}, {5.60, 1.65, 10.20},
      {2.15, 2.30, 6.80},  {6.35, -1.55, 9.40}, {1.65, -2.05, 7.90},
  };

  std::vector<Correspondence> correspondences;
  correspondences.reserve(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    correspondences.push_back(MakeCorrespondence(
        rig2_from_rig1, cameras, source_sensors[i], target_sensors[i],
        points[i]));
  }

  std::vector<GRNPObservation> source;
  std::vector<GRNPObservation> target;
  source.reserve(correspondences.size());
  target.reserve(correspondences.size());
  for (const Correspondence& correspondence : correspondences) {
    source.push_back(correspondence.source);
    target.push_back(correspondence.target);
  }

  std::vector<Rigid3d> candidates;
  GR6PEstimator::Estimate(source, target, &candidates);

  struct CandidateMetrics {
    double rotation_error_rad;
    double translation_error;
    double residual_sum;
    double residual_max;
    double ranking_error;
  };
  std::vector<CandidateMetrics> metrics;
  metrics.reserve(candidates.size());
  size_t best_index = std::numeric_limits<size_t>::max();
  double best_ranking_error = std::numeric_limits<double>::infinity();
  const double translation_scale = rig2_from_rig1.translation().norm();
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (!IsFinite(candidates[i])) {
      std::cerr << "solver returned a non-finite candidate\n";
      return 1;
    }
    std::vector<double> residuals;
    GR6PEstimator::Residuals(source, target, candidates[i], &residuals);
    const double residual_sum =
        std::accumulate(residuals.begin(), residuals.end(), 0.0);
    const double residual_max =
        residuals.empty() ? 0.0
                          : *std::max_element(residuals.begin(), residuals.end());
    const double rotation_error =
        RotationErrorRad(candidates[i], rig2_from_rig1);
    const double translation_error =
        (candidates[i].translation() - rig2_from_rig1.translation()).norm();
    const double ranking_error =
        rotation_error + translation_error / translation_scale;
    metrics.push_back({rotation_error, translation_error, residual_sum,
                       residual_max, ranking_error});
    if (ranking_error < best_ranking_error) {
      best_ranking_error = ranking_error;
      best_index = i;
    }
  }

  *output << std::setprecision(17);
  *output << "{\n"
          << "  \"schema\":\"colmap_gr6p_fixture_v1\",\n"
          << "  \"solver\":\"colmap::GR6PEstimator::Estimate\",\n"
          << "  \"colmap_image\":\"colmap/colmap@sha256:b809882552887b6471094dcadd2f2eb01656b010663564c43a5e7f04c0a08f2f\",\n"
          << "  \"colmap_version\":\"4.2.0.dev0\",\n"
          << "  \"num_correspondences\":" << correspondences.size() << ",\n"
          << "  \"truth_rig2_from_rig1\":";
  WriteRigid(*output, rig2_from_rig1);
  *output << ",\n  \"correspondences\":[\n";
  for (size_t i = 0; i < correspondences.size(); ++i) {
    const Correspondence& correspondence = correspondences[i];
    *output << "    {\"point_in_rig1\":";
    WriteVector(*output, correspondence.point_in_rig1);
    *output << ",\"point_in_rig2\":";
    WriteVector(*output, correspondence.point_in_rig2);
    *output << ",\"source\":";
    WriteObservation(*output, correspondence.source_sensor,
                     correspondence.source);
    *output << ",\"target\":";
    WriteObservation(*output, correspondence.target_sensor,
                     correspondence.target);
    *output << '}';
    if (i + 1 != correspondences.size()) {
      *output << ',';
    }
    *output << '\n';
  }
  *output << "  ],\n  \"candidate_count\":" << candidates.size()
          << ",\n  \"best_candidate_index\":";
  if (best_index == std::numeric_limits<size_t>::max()) {
    *output << "null";
  } else {
    *output << best_index;
  }
  *output << ",\n  \"candidates\":[\n";
  for (size_t i = 0; i < candidates.size(); ++i) {
    *output << "    {\"pose\":";
    WriteRigid(*output, candidates[i]);
    *output << ",\"rotation_error_rad\":"
            << metrics[i].rotation_error_rad
            << ",\"translation_error\":" << metrics[i].translation_error
            << ",\"residual_sum\":" << metrics[i].residual_sum
            << ",\"residual_max\":" << metrics[i].residual_max
            << ",\"ranking_error\":" << metrics[i].ranking_error << '}';
    if (i + 1 != candidates.size()) {
      *output << ',';
    }
    *output << '\n';
  }
  *output << "  ]\n}\n";
  return 0;
}
