// Bounded standalone Ceres reference for the frozen visloc rig BA fixture.
//
// The first implementation stage intentionally provides only strict fixture
// parsing and evaluate-only geometry.  It does not touch the Rust BA path,
// convert a COLMAP model, or run a solve.  The solve entry point can reuse
// the parsed state after the evaluate-only contract has been independently
// checked.

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/version.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace ceres_rig_reference {

constexpr std::size_t kMaxCameras = 16;
constexpr std::size_t kMaxPoses = 512;
constexpr std::size_t kMaxLandmarks = 8192;
constexpr std::size_t kMaxObservations = 262144;
constexpr double kQuaternionNormTolerance = 1e-9;
constexpr double kRigBaselineTolerance = 1e-12;

class FixtureError final : public std::runtime_error {
 public:
  explicit FixtureError(const std::string& message)
      : std::runtime_error(message) {}
};

struct Camera {
  std::uint64_t id = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::array<double, 4> intrinsics{};
};

struct Pose {
  std::uint64_t id = 0;
  // The fixture stores world-to-rig as wxyz followed by translation.
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
};

struct Landmark {
  std::uint64_t id = 0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
};

struct Observation {
  std::uint64_t frame_id = 0;
  std::uint64_t landmark_id = 0;
  Eigen::Vector2d xy = Eigen::Vector2d::Zero();
  std::uint64_t camera_id = 0;
  // Fixed sensor-from-rig transform, stored as wxyz and translation.
  Eigen::Quaterniond sensor_from_rig_rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d sensor_from_rig_translation = Eigen::Vector3d::Zero();
};

struct Fixture {
  std::string source_sha256;
  std::string source_sha256_cameras;
  std::string source_sha256_images;
  std::string source_sha256_points;
  std::string source_sha256_manifest;
  double declared_initial_cost = 0.0;
  std::uint64_t declared_initial_cost_bits = 0;
  std::uint64_t fixed_pose_id = 0;
  std::vector<Camera> cameras;
  std::vector<Pose> poses;
  std::vector<Landmark> landmarks;
  std::vector<Observation> observations;
  std::map<std::uint64_t, std::size_t> camera_index;
  std::map<std::uint64_t, std::size_t> pose_index;
  std::map<std::uint64_t, std::size_t> landmark_index;
  // One fixed sensor transform per camera is required.  The fixture repeats
  // it on every observation so the observation row remains self-contained.
  std::map<std::uint64_t,
           std::pair<Eigen::Quaterniond, Eigen::Vector3d>>
      sensor_from_rig_by_camera;
};

struct EvaluationSummary {
  double squared_cost = 0.0;
  double eigen_squared_cost = 0.0;
  double max_ceres_eigen_residual_abs_diff = 0.0;
  std::size_t observation_count = 0;
  std::size_t positive_depth_count = 0;
  double minimum_depth = std::numeric_limits<double>::infinity();
  double maximum_depth = -std::numeric_limits<double>::infinity();
};

struct Cli {
  bool self_test = false;
  bool evaluate_only = false;
  fs::path fixture;
  fs::path dump;
};

// This is the single residual functor shared by evaluate-only and the future
// Ceres solve path.  Pose parameters are [qw, qx, qy, qz, tx, ty, tz], matching
// the fixture and the Rust exporter.  The two fixed sensor parameters are
// captured by value and are never optimization blocks.
struct RigReprojectionCost {
  std::array<double, 4> intrinsics{};
  Eigen::Vector2d observed = Eigen::Vector2d::Zero();
  Eigen::Quaterniond sensor_from_rig_rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d sensor_from_rig_translation = Eigen::Vector3d::Zero();

  template <typename T>
  bool operator()(const T* const pose,
                  const T* const point_world,
                  T* residuals) const {
    T point_rig[3];
    ceres::QuaternionRotatePoint(pose, point_world, point_rig);
    point_rig[0] += T(pose[4]);
    point_rig[1] += T(pose[5]);
    point_rig[2] += T(pose[6]);

    const T sensor_rotation[4] = {
        T(sensor_from_rig_rotation.w()),
        T(sensor_from_rig_rotation.x()),
        T(sensor_from_rig_rotation.y()),
        T(sensor_from_rig_rotation.z()),
    };
    T point_sensor[3];
    ceres::QuaternionRotatePoint(sensor_rotation, point_rig, point_sensor);
    point_sensor[0] += T(sensor_from_rig_translation.x());
    point_sensor[1] += T(sensor_from_rig_translation.y());
    point_sensor[2] += T(sensor_from_rig_translation.z());
    if (!ceres::IsFinite(point_sensor[0]) ||
        !ceres::IsFinite(point_sensor[1]) ||
        !ceres::IsFinite(point_sensor[2]) || !(point_sensor[2] > T(0))) {
      return false;
    }

    residuals[0] = T(intrinsics[0]) * point_sensor[0] / point_sensor[2] +
                   T(intrinsics[2]) - T(observed.x());
    residuals[1] = T(intrinsics[1]) * point_sensor[1] / point_sensor[2] +
                   T(intrinsics[3]) - T(observed.y());
    return ceres::IsFinite(residuals[0]) && ceres::IsFinite(residuals[1]);
  }
};

using RigReprojectionCostFunction =
    ceres::AutoDiffCostFunction<RigReprojectionCost, 2, 7, 3>;

Eigen::Vector3d TransformPoint(const Pose& pose,
                               const Observation& observation,
                               const Landmark& landmark);

std::unique_ptr<ceres::CostFunction> MakeRigReprojectionCost(
    const Camera& camera,
    const Observation& observation) {
  auto* functor = new RigReprojectionCost;
  functor->intrinsics = camera.intrinsics;
  functor->observed = observation.xy;
  functor->sensor_from_rig_rotation = observation.sensor_from_rig_rotation;
  functor->sensor_from_rig_translation = observation.sensor_from_rig_translation;
  return std::unique_ptr<ceres::CostFunction>(
      new RigReprojectionCostFunction(functor));
}

struct CeresObservationEvaluation {
  Eigen::Vector2d residual = Eigen::Vector2d::Zero();
  double depth = 0.0;
};

CeresObservationEvaluation EvaluateWithCeres(const Pose& pose,
                                             const Observation& observation,
                                             const Landmark& landmark,
                                             const Camera& camera) {
  const std::unique_ptr<ceres::CostFunction> cost =
      MakeRigReprojectionCost(camera, observation);
  const Eigen::Quaterniond& rotation = pose.rotation;
  const double pose_parameters[7] = {
      rotation.w(), rotation.x(), rotation.y(), rotation.z(),
      pose.translation.x(), pose.translation.y(), pose.translation.z(),
  };
  const double point_parameters[3] = {
      landmark.position.x(), landmark.position.y(), landmark.position.z(),
  };
  const double* parameters[] = {pose_parameters, point_parameters};
  double residual[2] = {0.0, 0.0};
  if (!cost->Evaluate(parameters, residual, nullptr) ||
      !std::isfinite(residual[0]) || !std::isfinite(residual[1])) {
    throw FixtureError("Ceres AutoDiff rejected a nonfinite or nonpositive-depth "
                       "observation");
  }
  CeresObservationEvaluation result;
  result.residual = Eigen::Vector2d(residual[0], residual[1]);
  result.depth = TransformPoint(pose, observation, landmark).z();
  if (!std::isfinite(result.depth) || !(result.depth > 0.0)) {
    throw FixtureError("Ceres AutoDiff accepted an invalid depth");
  }
  return result;
}

std::string Trim(const std::string& input) {
  const std::size_t first = input.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const std::size_t last = input.find_last_not_of(" \t\r\n");
  return input.substr(first, last - first + 1);
}

std::vector<std::string> Split(const std::string& line) {
  std::istringstream stream(line);
  std::vector<std::string> fields;
  std::string field;
  while (stream >> field) {
    fields.push_back(field);
  }
  return fields;
}

[[noreturn]] void Fail(std::size_t line, const std::string& message) {
  throw FixtureError("fixture line " + std::to_string(line) + ": " + message);
}

std::uint64_t ParseUint64(const std::string& token,
                          std::size_t line,
                          const char* field) {
  if (token.empty() || token[0] == '-') {
    Fail(line, std::string("invalid ") + field + " " + token);
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(token.c_str(), &end, 10);
  if (errno == ERANGE || end == token.c_str() || *end != '\0') {
    Fail(line, std::string("invalid ") + field + " " + token);
  }
  return static_cast<std::uint64_t>(value);
}

double ParseDouble(const std::string& token,
                   std::size_t line,
                   const char* field) {
  if (token.empty()) {
    Fail(line, std::string("empty ") + field);
  }
  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(token.c_str(), &end);
  if (errno == ERANGE || end == token.c_str() || *end != '\0' ||
      !std::isfinite(value)) {
    Fail(line, std::string("invalid finite ") + field + " " + token);
  }
  return value;
}

std::size_t ParseCount(const std::string& token,
                       std::size_t line,
                       const char* field,
                       std::size_t cap) {
  const std::uint64_t value = ParseUint64(token, line, field);
  if (value > cap) {
    Fail(line, std::string(field) + " exceeds bounded cap");
  }
  return static_cast<std::size_t>(value);
}

void RequireFieldCount(const std::vector<std::string>& fields,
                       std::size_t expected,
                       std::size_t line,
                       const char* record) {
  if (fields.size() != expected) {
    Fail(line, std::string(record) + " requires " + std::to_string(expected) +
                  " fields, got " + std::to_string(fields.size()));
  }
}

void RequireSha256(const std::string& value,
                   std::size_t line,
                   const char* field) {
  if (value.size() != 64 ||
      !std::all_of(value.begin(), value.end(), [](const char character) {
        return std::isxdigit(static_cast<unsigned char>(character)) != 0;
      })) {
    Fail(line, std::string(field) + " must be a 64-character hex digest");
  }
}

std::uint64_t DoubleBits(double value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "unexpected double size");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void ValidateQuaternion(const Eigen::Quaterniond& quaternion,
                        std::size_t line,
                        const char* field) {
  const double norm = quaternion.norm();
  if (!std::isfinite(norm) || norm <= 0.0 ||
      std::abs(norm - 1.0) > kQuaternionNormTolerance) {
    Fail(line, std::string(field) + " is not a unit quaternion");
  }
}

Eigen::Vector3d CameraCenterInRig(const Eigen::Quaterniond& rotation,
                                  const Eigen::Vector3d& translation) {
  return rotation.conjugate() * (-translation);
}

void ReadRequiredHeader(std::istream& input,
                        std::size_t* line_number,
                        const char* expected_key,
                        std::string* value) {
  std::string line;
  if (!std::getline(input, line)) {
    throw FixtureError(std::string("missing header ") + expected_key);
  }
  ++*line_number;
  const std::vector<std::string> fields = Split(line);
  if (fields.size() != 2 || fields[0] != expected_key) {
    Fail(*line_number,
         std::string("expected header ") + expected_key);
  }
  *value = fields[1];
}

std::size_t ReadRequiredCount(std::istream& input,
                              std::size_t* line_number,
                              const char* key,
                              std::size_t cap) {
  std::string value;
  ReadRequiredHeader(input, line_number, key, &value);
  return ParseCount(value, *line_number, key, cap);
}

Fixture ParseFixture(std::istream& input, const std::string& source_name) {
  Fixture fixture;
  std::size_t line_number = 0;
  std::string line;
  if (!std::getline(input, line)) {
    throw FixtureError("fixture is empty: " + source_name);
  }
  ++line_number;
  if (Trim(line) != "VISLOC_BA_ORACLE_FIXTURE 1") {
    Fail(line_number, "unsupported fixture magic/version");
  }

  ReadRequiredHeader(input, &line_number, "SOURCE_SHA256",
                     &fixture.source_sha256);
  RequireSha256(fixture.source_sha256, line_number, "SOURCE_SHA256");
  ReadRequiredHeader(input, &line_number, "SOURCE_SHA256_CAMERAS",
                     &fixture.source_sha256_cameras);
  RequireSha256(fixture.source_sha256_cameras, line_number,
                "SOURCE_SHA256_CAMERAS");
  ReadRequiredHeader(input, &line_number, "SOURCE_SHA256_IMAGES",
                     &fixture.source_sha256_images);
  RequireSha256(fixture.source_sha256_images, line_number,
                "SOURCE_SHA256_IMAGES");
  ReadRequiredHeader(input, &line_number, "SOURCE_SHA256_POINTS",
                     &fixture.source_sha256_points);
  RequireSha256(fixture.source_sha256_points, line_number,
                "SOURCE_SHA256_POINTS");
  ReadRequiredHeader(input, &line_number, "SOURCE_SHA256_MANIFEST",
                     &fixture.source_sha256_manifest);

  std::string value;
  ReadRequiredHeader(input, &line_number, "INITIAL_COST", &value);
  fixture.declared_initial_cost = ParseDouble(value, line_number, "INITIAL_COST");
  if (fixture.declared_initial_cost < 0.0) {
    Fail(line_number, "INITIAL_COST must be nonnegative");
  }
  ReadRequiredHeader(input, &line_number, "INITIAL_COST_BITS", &value);
  fixture.declared_initial_cost_bits =
      ParseUint64(value, line_number, "INITIAL_COST_BITS");
  if (DoubleBits(fixture.declared_initial_cost) !=
      fixture.declared_initial_cost_bits) {
    Fail(line_number, "INITIAL_COST_BITS does not match INITIAL_COST");
  }

  const std::size_t camera_count =
      ReadRequiredCount(input, &line_number, "CAMERA_COUNT", kMaxCameras);
  const std::size_t pose_count =
      ReadRequiredCount(input, &line_number, "POSE_COUNT", kMaxPoses);
  const std::size_t landmark_count =
      ReadRequiredCount(input, &line_number, "LANDMARK_COUNT", kMaxLandmarks);
  const std::size_t observation_count = ReadRequiredCount(
      input, &line_number, "OBSERVATION_COUNT", kMaxObservations);
  if (camera_count == 0 || pose_count == 0 || landmark_count == 0 ||
      observation_count == 0) {
    throw FixtureError("fixture counts must all be positive");
  }

  fixture.cameras.reserve(camera_count);
  fixture.poses.reserve(pose_count);
  fixture.landmarks.reserve(landmark_count);
  fixture.observations.reserve(observation_count);

  for (std::size_t index = 0; index < camera_count; ++index) {
    if (!std::getline(input, line)) {
      throw FixtureError("truncated CAMERA section");
    }
    ++line_number;
    const std::vector<std::string> fields = Split(line);
    RequireFieldCount(fields, 10, line_number, "CAMERA");
    if (fields[0] != "CAMERA" || fields[2] != "PINHOLE" || fields[5] != "4") {
      Fail(line_number, "only PINHOLE cameras with four parameters are accepted");
    }
    Camera camera;
    camera.id = ParseUint64(fields[1], line_number, "camera id");
    camera.width = static_cast<std::uint32_t>(
        ParseUint64(fields[3], line_number, "camera width"));
    camera.height = static_cast<std::uint32_t>(
        ParseUint64(fields[4], line_number, "camera height"));
    if (camera.width == 0 || camera.height == 0) {
      Fail(line_number, "camera dimensions must be positive");
    }
    for (std::size_t parameter = 0; parameter < camera.intrinsics.size();
         ++parameter) {
      camera.intrinsics[parameter] =
          ParseDouble(fields[6 + parameter], line_number, "camera parameter");
    }
    if (camera.intrinsics[0] <= 0.0 || camera.intrinsics[1] <= 0.0) {
      Fail(line_number, "camera focal lengths must be positive");
    }
    if (!fixture.camera_index.emplace(camera.id, fixture.cameras.size()).second) {
      Fail(line_number, "duplicate camera id");
    }
    fixture.cameras.push_back(camera);
  }

  for (std::size_t index = 0; index < pose_count; ++index) {
    if (!std::getline(input, line)) {
      throw FixtureError("truncated POSE section");
    }
    ++line_number;
    const std::vector<std::string> fields = Split(line);
    RequireFieldCount(fields, 9, line_number, "POSE");
    if (fields[0] != "POSE") {
      Fail(line_number, "expected POSE record");
    }
    Pose pose;
    pose.id = ParseUint64(fields[1], line_number, "pose id");
    pose.rotation = Eigen::Quaterniond(
        ParseDouble(fields[2], line_number, "pose qw"),
        ParseDouble(fields[3], line_number, "pose qx"),
        ParseDouble(fields[4], line_number, "pose qy"),
        ParseDouble(fields[5], line_number, "pose qz"));
    pose.translation = Eigen::Vector3d(
        ParseDouble(fields[6], line_number, "pose tx"),
        ParseDouble(fields[7], line_number, "pose ty"),
        ParseDouble(fields[8], line_number, "pose tz"));
    ValidateQuaternion(pose.rotation, line_number, "pose quaternion");
    if (!fixture.pose_index.emplace(pose.id, fixture.poses.size()).second) {
      Fail(line_number, "duplicate pose id");
    }
    fixture.poses.push_back(pose);
  }

  if (!std::getline(input, line)) {
    throw FixtureError("missing FIXED_POSE record");
  }
  ++line_number;
  {
    const std::vector<std::string> fields = Split(line);
    RequireFieldCount(fields, 2, line_number, "FIXED_POSE");
    if (fields[0] != "FIXED_POSE") {
      Fail(line_number, "expected FIXED_POSE record");
    }
    fixture.fixed_pose_id = ParseUint64(fields[1], line_number, "fixed pose id");
    if (fixture.fixed_pose_id != 0 ||
        fixture.pose_index.find(fixture.fixed_pose_id) ==
            fixture.pose_index.end()) {
      Fail(line_number, "the bounded reference requires existing FIXED_POSE 0");
    }
  }

  for (std::size_t index = 0; index < landmark_count; ++index) {
    if (!std::getline(input, line)) {
      throw FixtureError("truncated LANDMARK section");
    }
    ++line_number;
    const std::vector<std::string> fields = Split(line);
    RequireFieldCount(fields, 5, line_number, "LANDMARK");
    if (fields[0] != "LANDMARK") {
      Fail(line_number, "expected LANDMARK record");
    }
    Landmark landmark;
    landmark.id = ParseUint64(fields[1], line_number, "landmark id");
    landmark.position = Eigen::Vector3d(
        ParseDouble(fields[2], line_number, "landmark x"),
        ParseDouble(fields[3], line_number, "landmark y"),
        ParseDouble(fields[4], line_number, "landmark z"));
    if (!fixture.landmark_index.emplace(landmark.id,
                                        fixture.landmarks.size())
             .second) {
      Fail(line_number, "duplicate landmark id");
    }
    fixture.landmarks.push_back(landmark);
  }

  std::vector<std::size_t> landmark_observation_counts(landmark_count, 0);
  for (std::size_t index = 0; index < observation_count; ++index) {
    if (!std::getline(input, line)) {
      throw FixtureError("truncated RIG_OBSERVATION section");
    }
    ++line_number;
    const std::vector<std::string> fields = Split(line);
    RequireFieldCount(fields, 13, line_number, "RIG_OBSERVATION");
    if (fields[0] != "RIG_OBSERVATION") {
      Fail(line_number, "expected RIG_OBSERVATION record");
    }
    Observation observation;
    observation.frame_id = ParseUint64(fields[1], line_number, "observation frame id");
    observation.landmark_id =
        ParseUint64(fields[2], line_number, "observation landmark id");
    observation.xy = Eigen::Vector2d(
        ParseDouble(fields[3], line_number, "observation x"),
        ParseDouble(fields[4], line_number, "observation y"));
    observation.camera_id =
        ParseUint64(fields[5], line_number, "observation camera id");
    observation.sensor_from_rig_rotation = Eigen::Quaterniond(
        ParseDouble(fields[6], line_number, "sensor qw"),
        ParseDouble(fields[7], line_number, "sensor qx"),
        ParseDouble(fields[8], line_number, "sensor qy"),
        ParseDouble(fields[9], line_number, "sensor qz"));
    observation.sensor_from_rig_translation = Eigen::Vector3d(
        ParseDouble(fields[10], line_number, "sensor tx"),
        ParseDouble(fields[11], line_number, "sensor ty"),
        ParseDouble(fields[12], line_number, "sensor tz"));
    ValidateQuaternion(observation.sensor_from_rig_rotation, line_number,
                       "sensor quaternion");
    if (fixture.pose_index.find(observation.frame_id) == fixture.pose_index.end()) {
      Fail(line_number, "observation references unknown frame");
    }
    const auto landmark = fixture.landmark_index.find(observation.landmark_id);
    if (landmark == fixture.landmark_index.end()) {
      Fail(line_number, "observation references unknown landmark");
    }
    if (fixture.camera_index.find(observation.camera_id) ==
        fixture.camera_index.end()) {
      Fail(line_number, "observation references unknown camera");
    }
    ++landmark_observation_counts[landmark->second];

    const auto sensor = fixture.sensor_from_rig_by_camera.find(observation.camera_id);
    if (sensor == fixture.sensor_from_rig_by_camera.end()) {
      fixture.sensor_from_rig_by_camera.emplace(
          observation.camera_id,
          std::make_pair(observation.sensor_from_rig_rotation,
                         observation.sensor_from_rig_translation));
    } else {
      const double rotation_error =
          (sensor->second.first.coeffs() -
           observation.sensor_from_rig_rotation.coeffs())
              .cwiseAbs()
              .maxCoeff();
      const double translation_error =
          (sensor->second.second - observation.sensor_from_rig_translation)
              .cwiseAbs()
              .maxCoeff();
      if (!std::isfinite(rotation_error) || !std::isfinite(translation_error) ||
          rotation_error > 1e-12 || translation_error > 1e-12) {
        Fail(line_number,
             "sensor-from-rig transform changes within one camera id");
      }
    }
    fixture.observations.push_back(observation);
  }

  if (std::any_of(landmark_observation_counts.begin(),
                  landmark_observation_counts.end(),
                  [](std::size_t count) { return count == 0; })) {
    throw FixtureError("every variable landmark must have at least one observation");
  }

  if (!std::getline(input, line)) {
    throw FixtureError("missing END record");
  }
  ++line_number;
  if (Trim(line) != "END") {
    Fail(line_number, "expected END record");
  }
  while (std::getline(input, line)) {
    ++line_number;
    if (!Trim(line).empty()) {
      Fail(line_number, "records after END are not accepted");
    }
  }

  if (fixture.sensor_from_rig_by_camera.size() != fixture.cameras.size()) {
    throw FixtureError("every camera must occur in an observation with a fixed sensor pose");
  }
  double maximum_baseline = 0.0;
  for (const auto& left : fixture.sensor_from_rig_by_camera) {
    const Eigen::Vector3d left_center =
        CameraCenterInRig(left.second.first, left.second.second);
    for (const auto& right : fixture.sensor_from_rig_by_camera) {
      const Eigen::Vector3d right_center =
          CameraCenterInRig(right.second.first, right.second.second);
      maximum_baseline = std::max(maximum_baseline,
                                  (left_center - right_center).norm());
    }
  }
  if (!std::isfinite(maximum_baseline) ||
      maximum_baseline <= kRigBaselineTolerance) {
    throw FixtureError("rig must contain a nonzero sensor baseline");
  }
  return fixture;
}

Eigen::Vector3d TransformPoint(const Pose& pose,
                               const Observation& observation,
                               const Landmark& landmark) {
  const Eigen::Vector3d point_rig = pose.rotation * landmark.position +
                                    pose.translation;
  return observation.sensor_from_rig_rotation * point_rig +
         observation.sensor_from_rig_translation;
}

EvaluationSummary Evaluate(const Fixture& fixture, std::ostream& dump) {
  dump << "VISLOC_BA_EVALUATION 1\n"
       << "SOURCE_SHA256 " << fixture.source_sha256 << "\n"
       << "SOURCE_SHA256_CAMERAS " << fixture.source_sha256_cameras << "\n"
       << "SOURCE_SHA256_IMAGES " << fixture.source_sha256_images << "\n"
       << "SOURCE_SHA256_POINTS " << fixture.source_sha256_points << "\n"
       << "SOURCE_SHA256_MANIFEST " << fixture.source_sha256_manifest << "\n"
       << "CERES_VERSION " << CERES_VERSION_STRING << "\n"
       << "FIXED_POSE " << fixture.fixed_pose_id << "\n"
       << "DECLARED_INITIAL_COST " << std::setprecision(17)
       << fixture.declared_initial_cost << "\n"
       << "DECLARED_INITIAL_COST_BITS " << fixture.declared_initial_cost_bits
       << "\n"
       << "OBSERVATION_INDEX FRAME_ID LANDMARK_ID CAMERA_ID RESIDUAL_X "
          "RESIDUAL_Y DEPTH SQUARED_COST\n";

  EvaluationSummary summary;
  summary.observation_count = fixture.observations.size();
  dump << std::setprecision(17);
  for (std::size_t index = 0; index < fixture.observations.size(); ++index) {
    const Observation& observation = fixture.observations[index];
    const Pose& pose = fixture.poses.at(fixture.pose_index.at(observation.frame_id));
    const Landmark& landmark =
        fixture.landmarks.at(fixture.landmark_index.at(observation.landmark_id));
    const Camera& camera = fixture.cameras.at(fixture.camera_index.at(observation.camera_id));
    const Eigen::Vector3d point_sensor =
        TransformPoint(pose, observation, landmark);
    if (!point_sensor.allFinite() || !(point_sensor.z() > 0.0)) {
      throw FixtureError("evaluate-only rejected nonfinite or nonpositive depth at "
                         "observation " + std::to_string(index));
    }
    const double eigen_predicted_x = camera.intrinsics[0] * point_sensor.x() /
                                         point_sensor.z() + camera.intrinsics[2];
    const double eigen_predicted_y = camera.intrinsics[1] * point_sensor.y() /
                                         point_sensor.z() + camera.intrinsics[3];
    const Eigen::Vector2d eigen_residual(
        eigen_predicted_x - observation.xy.x(),
        eigen_predicted_y - observation.xy.y());
    const double eigen_squared_cost = eigen_residual.squaredNorm();
    if (!eigen_residual.allFinite() || !std::isfinite(eigen_squared_cost)) {
      throw FixtureError("evaluate-only rejected nonfinite projection at observation " +
                         std::to_string(index));
    }
    const CeresObservationEvaluation ceres =
        EvaluateWithCeres(pose, observation, landmark, camera);
    const Eigen::Vector2d residual_difference = ceres.residual - eigen_residual;
    const double max_difference = residual_difference.cwiseAbs().maxCoeff();
    if (!std::isfinite(max_difference)) {
      throw FixtureError("evaluate-only residual parity is nonfinite at observation " +
                         std::to_string(index));
    }
    summary.max_ceres_eigen_residual_abs_diff = std::max(
        summary.max_ceres_eigen_residual_abs_diff, max_difference);
    const double squared_cost = ceres.residual.squaredNorm();
    if (!std::isfinite(squared_cost)) {
      throw FixtureError("evaluate-only Ceres squared cost overflowed at observation " +
                         std::to_string(index));
    }
    summary.squared_cost += squared_cost;
    summary.eigen_squared_cost += eigen_squared_cost;
    if (!std::isfinite(summary.squared_cost)) {
      throw FixtureError("evaluate-only squared cost overflowed");
    }
    if (!std::isfinite(summary.eigen_squared_cost)) {
      throw FixtureError("evaluate-only Eigen squared cost overflowed");
    }
    ++summary.positive_depth_count;
    summary.minimum_depth = std::min(summary.minimum_depth, ceres.depth);
    summary.maximum_depth = std::max(summary.maximum_depth, ceres.depth);
    dump << "OBSERVATION " << index << ' ' << observation.frame_id << ' '
         << observation.landmark_id << ' ' << observation.camera_id << ' '
         << ceres.residual.x() << ' ' << ceres.residual.y() << ' ' << ceres.depth << ' '
         << squared_cost << '\n';
  }
  dump << "SUMMARY_OBSERVATIONS " << summary.observation_count << '\n'
       << "SUMMARY_POSITIVE_DEPTH " << summary.positive_depth_count << '\n'
       << "SUMMARY_SQUARED_COST " << std::setprecision(17)
       << summary.squared_cost << '\n'
       << "SUMMARY_SQUARED_COST_BITS " << DoubleBits(summary.squared_cost) << '\n'
       << "SUMMARY_EIGEN_SQUARED_COST " << summary.eigen_squared_cost << '\n'
       << "SUMMARY_CERES_EIGEN_MAX_RESIDUAL_ABS_DIFF "
       << summary.max_ceres_eigen_residual_abs_diff << '\n'
       << "SUMMARY_MIN_DEPTH " << summary.minimum_depth << '\n'
       << "SUMMARY_MAX_DEPTH " << summary.maximum_depth << '\n';
  return summary;
}

bool IsPathPrefix(const fs::path& ancestor, const fs::path& candidate) {
  std::error_code error;
  const fs::path relative = fs::relative(candidate, ancestor, error);
  if (error) {
    return false;
  }
  if (relative.empty() || relative == ".") {
    return true;
  }
  const auto first = relative.begin();
  return first != relative.end() && *first != ".." &&
         first->string().rfind("..", 0) != 0;
}

void ValidateOutputPath(const fs::path& fixture_path, const fs::path& output_path) {
  std::error_code error;
  const fs::file_status input_status = fs::symlink_status(fixture_path, error);
  if (error || !fs::is_regular_file(input_status) ||
      fs::is_symlink(input_status)) {
    throw FixtureError("fixture input must be a regular non-symlink file");
  }
  const fs::path input = fs::canonical(fixture_path, error);
  if (error) {
    throw FixtureError("cannot canonicalize fixture input: " + error.message());
  }
  const fs::file_status output_status = fs::symlink_status(output_path, error);
  if (!error && output_status.type() != fs::file_type::not_found) {
    throw FixtureError("output path already exists, including symlinks: " +
                       output_path.string());
  }
  error.clear();
  const fs::path parent = output_path.has_parent_path()
                              ? output_path.parent_path()
                              : fs::path(".");
  if (!fs::is_directory(fs::symlink_status(parent, error)) || error) {
    throw FixtureError("output parent must be an existing directory");
  }
  const fs::path output_parent = fs::canonical(parent, error);
  if (error) {
    throw FixtureError("cannot canonicalize output parent: " + error.message());
  }
  const fs::path output = output_parent / output_path.filename();
  if (IsPathPrefix(input, output) || IsPathPrefix(output, input)) {
    throw FixtureError("output path overlaps fixture input");
  }
}

fs::path MakeStagingPath(const fs::path& output) {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  return output.parent_path() /
         ("." + output.filename().string() + ".staging-" +
          std::to_string(static_cast<long long>(stamp)));
}

class ExclusiveFileBuffer final : public std::streambuf {
 public:
  explicit ExclusiveFileBuffer(int file_descriptor) : file_descriptor_(file_descriptor) {
    setp(buffer_, buffer_ + sizeof(buffer_));
  }

  ExclusiveFileBuffer(const ExclusiveFileBuffer&) = delete;
  ExclusiveFileBuffer& operator=(const ExclusiveFileBuffer&) = delete;

  ~ExclusiveFileBuffer() override {
    sync();
    if (file_descriptor_ >= 0) {
      ::close(file_descriptor_);
      file_descriptor_ = -1;
    }
  }

 protected:
  int_type overflow(int_type character = traits_type::eof()) override {
    if (!FlushBuffer()) {
      return traits_type::eof();
    }
    if (!traits_type::eq_int_type(character, traits_type::eof())) {
      *pptr() = traits_type::to_char_type(character);
      pbump(1);
    }
    return traits_type::not_eof(character);
  }

  int sync() override { return FlushBuffer() ? 0 : -1; }

 private:
  bool FlushBuffer() {
    const char* data = pbase();
    std::ptrdiff_t remaining = pptr() - pbase();
    while (remaining > 0) {
      const ssize_t written = ::write(file_descriptor_, data,
                                      static_cast<std::size_t>(remaining));
      if (written < 0 && errno == EINTR) {
        continue;
      }
      if (written <= 0) {
        return false;
      }
      data += written;
      remaining -= written;
    }
    setp(buffer_, buffer_ + sizeof(buffer_));
    return true;
  }

  int file_descriptor_ = -1;
  char buffer_[64 * 1024]{};
};

EvaluationSummary EvaluateToFile(const Fixture& fixture,
                                 const fs::path& fixture_path,
                                 const fs::path& output_path) {
  ValidateOutputPath(fixture_path, output_path);
  const fs::path staging = MakeStagingPath(output_path);
  std::error_code error;
  if (fs::symlink_status(staging, error).type() != fs::file_type::not_found ||
      error) {
    throw FixtureError("evaluation staging path already exists");
  }
  bool owns_staging = false;
  std::optional<EvaluationSummary> summary;
  try {
    const int file_descriptor =
        ::open(staging.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (file_descriptor < 0) {
      throw FixtureError("cannot exclusively create evaluation staging file: " +
                         std::string(std::strerror(errno)));
    }
    owns_staging = true;
    {
      ExclusiveFileBuffer buffer(file_descriptor);
      std::ostream output(&buffer);
      summary = Evaluate(fixture, output);
      output.flush();
      if (!output) {
        throw FixtureError("cannot flush evaluation staging file");
      }
    }
    // A hard link publishes without replacing a path that appeared after the
    // preflight.  The fixture and dump are required to share a filesystem.
    fs::create_hard_link(staging, output_path, error);
    if (error) {
      throw FixtureError("cannot publish evaluation output without overwrite: " +
                         error.message());
    }
    fs::remove(staging, error);
    if (error) {
      throw FixtureError("cannot remove owned evaluation staging file: " +
                         error.message());
    }
    owns_staging = false;
    std::ifstream summary_input(output_path, std::ios::binary);
    if (!summary_input) {
      throw FixtureError("published evaluation output cannot be reopened");
    }
    return *summary;
  } catch (...) {
    if (owns_staging) {
      std::error_code cleanup_error;
      fs::remove(staging, cleanup_error);
    }
    throw;
  }
}

std::string SyntheticFixture() {
  constexpr const char* kDigest =
      "0000000000000000000000000000000000000000000000000000000000000000";
  std::ostringstream fixture;
  fixture << "VISLOC_BA_ORACLE_FIXTURE 1\n"
          << "SOURCE_SHA256 " << kDigest << "\n"
          << "SOURCE_SHA256_CAMERAS " << kDigest << "\n"
          << "SOURCE_SHA256_IMAGES " << kDigest << "\n"
          << "SOURCE_SHA256_POINTS " << kDigest << "\n"
          << "SOURCE_SHA256_MANIFEST " << kDigest << "\n"
          << "INITIAL_COST 0\n"
          << "INITIAL_COST_BITS 0\n"
          << "CAMERA_COUNT 2\n"
          << "POSE_COUNT 1\n"
          << "LANDMARK_COUNT 1\n"
          << "OBSERVATION_COUNT 2\n"
          << "CAMERA 1 PINHOLE 848 800 4 10 10 0 0\n"
          << "CAMERA 2 PINHOLE 848 800 4 10 10 0 0\n"
          << "POSE 0 1 0 0 0 0 0 0\n"
          << "FIXED_POSE 0\n"
          << "LANDMARK 1 0 0 2\n"
          << "RIG_OBSERVATION 0 1 0 0 1 1 0 0 0 0 0 0\n"
          << "RIG_OBSERVATION 0 1 -2.5 0 2 1 0 0 0 -0.5 0 0\n"
          << "END\n";
  return fixture.str();
}

void ExpectFailure(const std::string& text, const std::string& expected) {
  std::istringstream input(text);
  try {
    (void)ParseFixture(input, "synthetic");
  } catch (const FixtureError& error) {
    if (std::string(error.what()).find(expected) == std::string::npos) {
      throw FixtureError("self-test expected error containing " + expected +
                         ", got " + error.what());
    }
    return;
  }
  throw FixtureError("self-test expected fixture rejection: " + expected);
}

void RunSelfTests() {
  const std::string valid = SyntheticFixture();
  std::istringstream input(valid);
  const Fixture fixture = ParseFixture(input, "synthetic");
  std::ostringstream dump;
  const EvaluationSummary summary = Evaluate(fixture, dump);
  if (summary.observation_count != 2 || summary.positive_depth_count != 2 ||
      summary.squared_cost != 0.0 ||
      dump.str().find("OBSERVATION 1 0 1 2") == std::string::npos) {
    throw FixtureError("self-test valid fixture evaluation mismatch");
  }

  std::string bad_depth = valid;
  const std::string old_landmark = "LANDMARK 1 0 0 2";
  const std::string new_landmark = "LANDMARK 1 0 0 -2";
  const std::size_t landmark_position = bad_depth.find(old_landmark);
  bad_depth.replace(landmark_position, old_landmark.size(), new_landmark);
  {
    std::istringstream bad_input(bad_depth);
    const Fixture bad_fixture = ParseFixture(bad_input, "bad-depth");
    std::ostringstream bad_dump;
    bool rejected = false;
    try {
      (void)Evaluate(bad_fixture, bad_dump);
    } catch (const FixtureError& error) {
      if (std::string(error.what()).find("nonfinite or nonpositive depth") ==
          std::string::npos) {
        throw FixtureError("self-test bad-depth error mismatch: " +
                           std::string(error.what()));
      }
      rejected = true;
    }
    if (!rejected) {
      throw FixtureError("self-test expected negative depth rejection");
    }
  }

  std::string duplicate_camera = valid;
  const std::string camera_2 = "CAMERA 2 PINHOLE 848 800 4 10 10 0 0";
  const std::size_t camera_position = duplicate_camera.find(camera_2);
  duplicate_camera.replace(camera_position, camera_2.size(),
                           "CAMERA 1 PINHOLE 848 800 4 10 10 0 0");
  ExpectFailure(duplicate_camera, "duplicate camera id");

  std::string unknown_landmark = valid;
  const std::string old_observation =
      "RIG_OBSERVATION 0 1 0 0 1 1 0 0 0 0 0 0";
  const std::size_t observation_position = unknown_landmark.find(old_observation);
  unknown_landmark.replace(observation_position, old_observation.size(),
                           "RIG_OBSERVATION 0 99 0 0 1 1 0 0 0 0 0 0");
  ExpectFailure(unknown_landmark, "unknown landmark");

  std::string nonfinite = valid;
  const std::size_t finite_position = nonfinite.find("LANDMARK 1 0 0 2");
  nonfinite.replace(finite_position, old_landmark.size(),
                    "LANDMARK 1 nan 0 2");
  ExpectFailure(nonfinite, "invalid finite landmark x");

  std::string trailing = valid + "RIG_OBSERVATION 0 1 0 0 1 1 0 0 0 0 0 0\n";
  ExpectFailure(trailing, "records after END");
}

std::string Usage() {
  return "Usage:\n"
         "  ceres_rig_reference --self-test\n"
         "  ceres_rig_reference --evaluate-only --fixture PATH --dump PATH\n"
         "\n"
         "The solve/publication mode is intentionally not enabled in this\n"
         "parser/evaluate-only checkpoint.\n";
}

Cli ParseCli(int argc, char** argv) {
  Cli cli;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto require_value = [&](const char* flag) -> fs::path {
      if (index + 1 >= argc) {
        throw FixtureError(std::string(flag) + " requires a path\n" + Usage());
      }
      ++index;
      return fs::path(argv[index]);
    };
    if (argument == "--self-test") {
      if (cli.self_test) {
        throw FixtureError("duplicate --self-test");
      }
      cli.self_test = true;
    } else if (argument == "--evaluate-only") {
      if (cli.evaluate_only) {
        throw FixtureError("duplicate --evaluate-only");
      }
      cli.evaluate_only = true;
    } else if (argument == "--fixture") {
      if (!cli.fixture.empty()) {
        throw FixtureError("duplicate --fixture");
      }
      cli.fixture = require_value("--fixture");
    } else if (argument == "--dump") {
      if (!cli.dump.empty()) {
        throw FixtureError("duplicate --dump");
      }
      cli.dump = require_value("--dump");
    } else if (argument == "--help" || argument == "-h") {
      throw FixtureError(Usage());
    } else {
      throw FixtureError("unknown argument " + argument + "\n" + Usage());
    }
  }
  if (cli.self_test) {
    if (cli.evaluate_only || !cli.fixture.empty() || !cli.dump.empty()) {
      throw FixtureError("--self-test is standalone\n" + Usage());
    }
    return cli;
  }
  if (!cli.evaluate_only || cli.fixture.empty() || cli.dump.empty()) {
    throw FixtureError("evaluate-only requires --fixture and --dump\n" + Usage());
  }
  return cli;
}

}  // namespace ceres_rig_reference

int main(int argc, char** argv) {
  using namespace ceres_rig_reference;
  try {
    const Cli cli = ParseCli(argc, argv);
    if (cli.self_test) {
      RunSelfTests();
      std::cout << "ceres_rig_reference self-test: PASS\n";
      return 0;
    }
    std::error_code error;
    const fs::file_status status = fs::symlink_status(cli.fixture, error);
    if (error || !fs::is_regular_file(status) || fs::is_symlink(status)) {
      throw FixtureError("--fixture must be a regular non-symlink file");
    }
    std::ifstream input(cli.fixture, std::ios::binary);
    if (!input) {
      throw FixtureError("cannot open fixture " + cli.fixture.string());
    }
    const Fixture fixture = ParseFixture(input, cli.fixture.string());
    const EvaluationSummary summary =
        EvaluateToFile(fixture, cli.fixture, cli.dump);
    const double absolute_delta =
        summary.squared_cost - fixture.declared_initial_cost;
    std::cout << std::setprecision(17)
              << "mode=evaluate-only ceres_version=" << CERES_VERSION_STRING
              << " source_sha256=" << fixture.source_sha256
              << " fixed_pose=" << fixture.fixed_pose_id
              << " cameras=" << fixture.cameras.size()
              << " poses=" << fixture.poses.size()
              << " landmarks=" << fixture.landmarks.size()
              << " observations=" << summary.observation_count
              << " positive_depth=" << summary.positive_depth_count
              << " declared_initial_cost=" << fixture.declared_initial_cost
              << " evaluated_squared_cost=" << summary.squared_cost
              << " evaluated_cost_bits=" << DoubleBits(summary.squared_cost)
              << " eigen_squared_cost=" << summary.eigen_squared_cost
              << " ceres_eigen_max_residual_abs_diff="
              << summary.max_ceres_eigen_residual_abs_diff
              << " cost_delta=" << absolute_delta
              << " min_depth=" << summary.minimum_depth
              << " max_depth=" << summary.maximum_depth
              << " dump=" << cli.dump << '\n';
    return 0;
  } catch (const FixtureError& error) {
    std::cerr << "ceres_rig_reference: " << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "ceres_rig_reference: unexpected error: " << error.what()
              << '\n';
    return 3;
  }
}
