// Bounded standalone Ceres reference for the frozen visloc rig BA fixture.
//
// The first implementation stage intentionally provides only strict fixture
// parsing and evaluate-only geometry.  It does not touch the Rust BA path,
// convert a COLMAP model, or run a solve.  The solve entry point can reuse
// the parsed state after the evaluate-only contract has been independently
// checked.

#include <ceres/ceres.h>
#include <ceres/product_manifold.h>
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

static_assert(CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR == 2 &&
                  CERES_VERSION_REVISION == 0,
              "the frozen reference requires Ceres 2.2.0");

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
  double max_ceres_eigen_depth_abs_diff = 0.0;
  std::size_t observation_count = 0;
  std::size_t positive_depth_count = 0;
  double minimum_depth = std::numeric_limits<double>::infinity();
  double maximum_depth = -std::numeric_limits<double>::infinity();
};

struct Cli {
  bool self_test = false;
  bool evaluate_only = false;
  bool solve = false;
  fs::path fixture;
  fs::path dump;
  fs::path state;
};

struct SolveParameters {
  std::vector<std::array<double, 7>> poses;
  std::vector<std::array<double, 3>> landmarks;
};

struct SolveResult {
  SolveParameters parameters;
  EvaluationSummary initial_evaluation;
  EvaluationSummary final_evaluation;
  ceres::Solver::Options solver_options;
  ceres::Solver::Summary solver_summary;
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
  bool TransformPoint(const T* const pose,
                      const T* const point_world,
                      T* point_sensor) const {
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
    ceres::QuaternionRotatePoint(sensor_rotation, point_rig, point_sensor);
    point_sensor[0] += T(sensor_from_rig_translation.x());
    point_sensor[1] += T(sensor_from_rig_translation.y());
    point_sensor[2] += T(sensor_from_rig_translation.z());
    return ceres::isfinite(point_sensor[0]) &&
           ceres::isfinite(point_sensor[1]) &&
           ceres::isfinite(point_sensor[2]) && point_sensor[2] > T(0);
  }

  template <typename T>
  bool operator()(const T* const pose,
                  const T* const point_world,
                  T* residuals) const {
    T point_sensor[3];
    if (!TransformPoint(pose, point_world, point_sensor)) {
      return false;
    }

    residuals[0] = T(intrinsics[0]) * point_sensor[0] / point_sensor[2] +
                   T(intrinsics[2]) - T(observed.x());
    residuals[1] = T(intrinsics[1]) * point_sensor[1] / point_sensor[2] +
                   T(intrinsics[3]) - T(observed.y());
    return ceres::isfinite(residuals[0]) && ceres::isfinite(residuals[1]);
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
  double eigen_depth = 0.0;
};

Eigen::Vector3d EigenTransformPoint(const double* pose_parameters,
                                    const Observation& observation,
                                    const double* point_parameters) {
  const Eigen::Quaterniond pose_rotation(
      pose_parameters[0], pose_parameters[1], pose_parameters[2],
      pose_parameters[3]);
  const Eigen::Vector3d pose_translation(
      pose_parameters[4], pose_parameters[5], pose_parameters[6]);
  const Eigen::Vector3d point_world(
      point_parameters[0], point_parameters[1], point_parameters[2]);
  return observation.sensor_from_rig_rotation *
             (pose_rotation * point_world + pose_translation) +
         observation.sensor_from_rig_translation;
}

CeresObservationEvaluation EvaluateWithCeresParameters(
    const double* pose_parameters,
    const double* point_parameters,
    const Observation& observation,
    const Camera& camera) {
  const std::unique_ptr<ceres::CostFunction> cost =
      MakeRigReprojectionCost(camera, observation);
  RigReprojectionCost transform_functor;
  transform_functor.intrinsics = camera.intrinsics;
  transform_functor.observed = observation.xy;
  transform_functor.sensor_from_rig_rotation =
      observation.sensor_from_rig_rotation;
  transform_functor.sensor_from_rig_translation =
      observation.sensor_from_rig_translation;
  double point_sensor[3] = {0.0, 0.0, 0.0};
  if (!transform_functor.TransformPoint(pose_parameters, point_parameters,
                                        point_sensor)) {
    throw FixtureError("Ceres transform rejected a nonfinite or nonpositive-depth "
                       "observation");
  }
  const double* parameters[] = {pose_parameters, point_parameters};
  double residual[2] = {0.0, 0.0};
  if (!cost->Evaluate(parameters, residual, nullptr) ||
      !std::isfinite(residual[0]) || !std::isfinite(residual[1])) {
    throw FixtureError("Ceres AutoDiff rejected a nonfinite or nonpositive-depth "
                       "observation");
  }
  CeresObservationEvaluation result;
  result.residual = Eigen::Vector2d(residual[0], residual[1]);
  result.depth = point_sensor[2];
  result.eigen_depth =
      EigenTransformPoint(pose_parameters, observation, point_parameters).z();
  if (!std::isfinite(result.depth) || !(result.depth > 0.0)) {
    throw FixtureError("Ceres AutoDiff accepted an invalid depth");
  }
  return result;
}

CeresObservationEvaluation EvaluateWithCeres(const Pose& pose,
                                             const Observation& observation,
                                             const Landmark& landmark,
                                             const Camera& camera) {
  const Eigen::Quaterniond& rotation = pose.rotation;
  const double pose_parameters[7] = {
      rotation.w(), rotation.x(), rotation.y(), rotation.z(),
      pose.translation.x(), pose.translation.y(), pose.translation.z(),
  };
  const double point_parameters[3] = {
      landmark.position.x(), landmark.position.y(), landmark.position.z(),
  };
  return EvaluateWithCeresParameters(pose_parameters, point_parameters,
                                     observation, camera);
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
    const CeresObservationEvaluation ceres =
        EvaluateWithCeres(pose, observation, landmark, camera);
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
    const double depth_difference = std::abs(ceres.depth - ceres.eigen_depth);
    if (!std::isfinite(depth_difference)) {
      throw FixtureError("evaluate-only depth parity is nonfinite at observation " +
                         std::to_string(index));
    }
    summary.max_ceres_eigen_depth_abs_diff = std::max(
        summary.max_ceres_eigen_depth_abs_diff, depth_difference);
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
       << "SUMMARY_CERES_EIGEN_MAX_DEPTH_ABS_DIFF "
       << summary.max_ceres_eigen_depth_abs_diff << '\n'
       << "SUMMARY_MIN_DEPTH " << summary.minimum_depth << '\n'
       << "SUMMARY_MAX_DEPTH " << summary.maximum_depth << '\n';
  return summary;
}

class NullStreamBuffer final : public std::streambuf {
 protected:
  int_type overflow(int_type character = traits_type::eof()) override {
    return traits_type::not_eof(character);
  }
};

SolveParameters MakeSolveParameters(const Fixture& fixture) {
  SolveParameters parameters;
  parameters.poses.reserve(fixture.poses.size());
  for (const Pose& pose : fixture.poses) {
    const Eigen::Quaterniond& rotation = pose.rotation;
    parameters.poses.push_back({rotation.w(), rotation.x(), rotation.y(),
                                rotation.z(), pose.translation.x(),
                                pose.translation.y(), pose.translation.z()});
  }
  parameters.landmarks.reserve(fixture.landmarks.size());
  for (const Landmark& landmark : fixture.landmarks) {
    parameters.landmarks.push_back(
        {landmark.position.x(), landmark.position.y(), landmark.position.z()});
  }
  return parameters;
}

EvaluationSummary EvaluateParameterState(const Fixture& fixture,
                                         const SolveParameters& parameters,
                                         std::ostream* dump,
                                         const char* observation_prefix) {
  if (parameters.poses.size() != fixture.poses.size() ||
      parameters.landmarks.size() != fixture.landmarks.size()) {
    throw FixtureError("parameter state dimensions do not match fixture");
  }
  EvaluationSummary summary;
  summary.observation_count = fixture.observations.size();
  if (dump != nullptr) {
    *dump << std::setprecision(17);
  }
  for (std::size_t index = 0; index < fixture.observations.size(); ++index) {
    const Observation& observation = fixture.observations[index];
    const std::size_t pose_index = fixture.pose_index.at(observation.frame_id);
    const std::size_t landmark_index =
        fixture.landmark_index.at(observation.landmark_id);
    const Camera& camera = fixture.cameras.at(
        fixture.camera_index.at(observation.camera_id));
    const CeresObservationEvaluation ceres = EvaluateWithCeresParameters(
        parameters.poses[pose_index].data(),
        parameters.landmarks[landmark_index].data(), observation, camera);
    const Eigen::Vector3d point_sensor = EigenTransformPoint(
        parameters.poses[pose_index].data(), observation,
        parameters.landmarks[landmark_index].data());
    if (!point_sensor.allFinite() || !(point_sensor.z() > 0.0)) {
      throw FixtureError("solve state has nonfinite or nonpositive depth at " +
                         std::string(observation_prefix) + " observation " +
                         std::to_string(index));
    }
    const Eigen::Vector2d eigen_residual(
        camera.intrinsics[0] * point_sensor.x() / point_sensor.z() +
            camera.intrinsics[2] - observation.xy.x(),
        camera.intrinsics[1] * point_sensor.y() / point_sensor.z() +
            camera.intrinsics[3] - observation.xy.y());
    const double squared_cost = ceres.residual.squaredNorm();
    const double eigen_squared_cost = eigen_residual.squaredNorm();
    if (!ceres.residual.allFinite() || !eigen_residual.allFinite() ||
        !std::isfinite(squared_cost) || !std::isfinite(eigen_squared_cost)) {
      throw FixtureError("solve state has nonfinite residual at " +
                         std::string(observation_prefix) + " observation " +
                         std::to_string(index));
    }
    summary.squared_cost += squared_cost;
    summary.eigen_squared_cost += eigen_squared_cost;
    if (!std::isfinite(summary.squared_cost) ||
        !std::isfinite(summary.eigen_squared_cost)) {
      throw FixtureError("solve state squared cost overflowed");
    }
    const double depth_difference = std::abs(ceres.depth - point_sensor.z());
    if (!std::isfinite(depth_difference)) {
      throw FixtureError("solve state depth parity is nonfinite at " +
                         std::string(observation_prefix) + " observation " +
                         std::to_string(index));
    }
    summary.max_ceres_eigen_depth_abs_diff = std::max(
        summary.max_ceres_eigen_depth_abs_diff, depth_difference);
    const double max_residual_difference =
        (ceres.residual - eigen_residual).cwiseAbs().maxCoeff();
    if (!std::isfinite(max_residual_difference)) {
      throw FixtureError("solve state residual parity is nonfinite at " +
                         std::string(observation_prefix) + " observation " +
                         std::to_string(index));
    }
    summary.max_ceres_eigen_residual_abs_diff = std::max(
        summary.max_ceres_eigen_residual_abs_diff, max_residual_difference);
    ++summary.positive_depth_count;
    summary.minimum_depth = std::min(summary.minimum_depth, ceres.depth);
    summary.maximum_depth = std::max(summary.maximum_depth, ceres.depth);
    if (dump != nullptr) {
      *dump << observation_prefix << ' ' << index << ' ' << observation.frame_id
            << ' ' << observation.landmark_id << ' ' << observation.camera_id
            << ' ' << ceres.residual.x() << ' ' << ceres.residual.y() << ' '
            << ceres.depth << ' ' << squared_cost << '\n';
    }
  }
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

ceres::Solver::Options MakeReferenceSolverOptions(
    const std::shared_ptr<ceres::ParameterBlockOrdering>& ordering) {
  ceres::Solver::Options options;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.linear_solver_ordering = ordering;
  options.max_num_iterations = 20;
  options.num_threads = 1;
  options.initial_trust_region_radius = 1e4;
  options.max_trust_region_radius = 1e16;
  options.min_trust_region_radius = 1e-32;
  options.min_relative_decrease = 1e-3;
  options.min_lm_diagonal = 1e-6;
  options.max_lm_diagonal = 1e32;
  options.max_num_consecutive_invalid_steps = 5;
  options.function_tolerance = 1e-6;
  options.gradient_tolerance = 1e-10;
  options.parameter_tolerance = 1e-8;
  options.use_nonmonotonic_steps = false;
  options.use_inner_iterations = false;
  options.dynamic_sparsity = false;
  options.use_mixed_precision_solves = false;
  options.max_num_refinement_iterations = 0;
  options.min_linear_solver_iterations = 0;
  options.max_linear_solver_iterations = 500;
  options.max_num_spse_iterations = 5;
  options.use_spse_initialization = false;
  options.spse_tolerance = 0.1;
  options.eta = 0.1;
  options.jacobi_scaling = true;
  options.preconditioner_type = ceres::JACOBI;
  options.minimizer_progress_to_stdout = false;
  options.logging_type = ceres::SILENT;
  options.check_gradients = false;
  options.gradient_check_relative_precision = 1e-8;
  options.gradient_check_numeric_derivative_relative_step_size = 1e-6;
  options.max_solver_time_in_seconds = 1e9;
  return options;
}

void WriteReferenceOptions(std::ostream& output,
                           const ceres::Solver::Options& options) {
  const auto bool_value = [](bool value) { return value ? 1 : 0; };
  output << "OPTIONS_BEGIN\n"
         << "OPTION_TRUST_REGION_STRATEGY LEVENBERG_MARQUARDT\n"
         << "OPTION_LINEAR_SOLVER SPARSE_SCHUR\n"
         << "OPTION_LINEAR_SOLVER_ORDERING points_group_0_poses_group_1\n"
         << std::setprecision(17)
         << "OPTION_MAX_NUM_ITERATIONS " << options.max_num_iterations << '\n'
         << "OPTION_NUM_THREADS " << options.num_threads << '\n'
         << "OPTION_INITIAL_TRUST_REGION_RADIUS "
         << options.initial_trust_region_radius << '\n'
         << "OPTION_MAX_TRUST_REGION_RADIUS "
         << options.max_trust_region_radius << '\n'
         << "OPTION_MIN_TRUST_REGION_RADIUS "
         << options.min_trust_region_radius << '\n'
         << "OPTION_MIN_RELATIVE_DECREASE " << options.min_relative_decrease
         << '\n'
         << "OPTION_MIN_LM_DIAGONAL " << options.min_lm_diagonal << '\n'
         << "OPTION_MAX_LM_DIAGONAL " << options.max_lm_diagonal << '\n'
         << "OPTION_MAX_CONSECUTIVE_INVALID_STEPS "
         << options.max_num_consecutive_invalid_steps << '\n'
         << "OPTION_FUNCTION_TOLERANCE " << options.function_tolerance << '\n'
         << "OPTION_GRADIENT_TOLERANCE " << options.gradient_tolerance << '\n'
         << "OPTION_PARAMETER_TOLERANCE " << options.parameter_tolerance << '\n'
         << "OPTION_USE_NONMONOTONIC_STEPS "
         << bool_value(options.use_nonmonotonic_steps) << '\n'
         << "OPTION_USE_INNER_ITERATIONS "
         << bool_value(options.use_inner_iterations) << '\n'
         << "OPTION_DYNAMIC_SPARSITY " << bool_value(options.dynamic_sparsity)
         << '\n'
         << "OPTION_USE_MIXED_PRECISION_SOLVES "
         << bool_value(options.use_mixed_precision_solves) << '\n'
         << "OPTION_MAX_REFINEMENT_ITERATIONS "
         << options.max_num_refinement_iterations << '\n'
         << "OPTION_MIN_LINEAR_SOLVER_ITERATIONS "
         << options.min_linear_solver_iterations << '\n'
         << "OPTION_MAX_LINEAR_SOLVER_ITERATIONS "
         << options.max_linear_solver_iterations << '\n'
         << "OPTION_MAX_SPSE_ITERATIONS " << options.max_num_spse_iterations
         << '\n'
         << "OPTION_USE_SPSE_INITIALIZATION "
         << bool_value(options.use_spse_initialization) << '\n'
         << "OPTION_SPSE_TOLERANCE " << options.spse_tolerance << '\n'
         << "OPTION_ETA " << options.eta << '\n'
         << "OPTION_JACOBI_SCALING " << bool_value(options.jacobi_scaling) << '\n'
         << "OPTION_PRECONDITIONER JACOBI\n"
         << "OPTION_MINIMIZER_PROGRESS_TO_STDOUT "
         << bool_value(options.minimizer_progress_to_stdout) << '\n'
         << "OPTION_LOGGING SILENT\n"
         << "OPTION_MAX_SOLVER_TIME_IN_SECONDS "
         << options.max_solver_time_in_seconds << '\n'
         << "OPTION_CHECK_GRADIENTS " << bool_value(options.check_gradients)
         << '\n'
         << "OPTION_GRADIENT_CHECK_RELATIVE_PRECISION "
         << options.gradient_check_relative_precision << '\n'
         << "OPTION_GRADIENT_CHECK_NUMERIC_STEP "
         << options.gradient_check_numeric_derivative_relative_step_size << '\n'
         << "OPTIONS_END\n";
}

SolveResult SolveInMemory(const Fixture& fixture) {
  NullStreamBuffer initial_buffer;
  std::ostream initial_output(&initial_buffer);
  const EvaluationSummary initial_evaluation = Evaluate(fixture, initial_output);
  const double initial_tolerance =
      1e-6 + 1e-10 * std::abs(fixture.declared_initial_cost);
  if (std::abs(initial_evaluation.squared_cost -
               fixture.declared_initial_cost) > initial_tolerance) {
    throw FixtureError("solve initial full squared cost does not match fixture "
                       "declaration");
  }

  SolveParameters parameters = MakeSolveParameters(fixture);
  ceres::Problem problem;
  for (std::array<double, 7>& pose : parameters.poses) {
    auto* manifold =
        new ceres::ProductManifold<ceres::QuaternionManifold,
                                   ceres::EuclideanManifold<3>>();
    problem.AddParameterBlock(pose.data(), static_cast<int>(pose.size()),
                              manifold);
  }
  for (std::array<double, 3>& landmark : parameters.landmarks) {
    problem.AddParameterBlock(landmark.data(),
                              static_cast<int>(landmark.size()));
  }
  for (const Observation& observation : fixture.observations) {
    const std::size_t pose_index = fixture.pose_index.at(observation.frame_id);
    const std::size_t landmark_index =
        fixture.landmark_index.at(observation.landmark_id);
    const Camera& camera = fixture.cameras.at(
        fixture.camera_index.at(observation.camera_id));
    std::unique_ptr<ceres::CostFunction> cost =
        MakeRigReprojectionCost(camera, observation);
    problem.AddResidualBlock(cost.release(), nullptr,
                             parameters.poses[pose_index].data(),
                             parameters.landmarks[landmark_index].data());
  }
  const std::size_t fixed_pose_index =
      fixture.pose_index.at(fixture.fixed_pose_id);
  problem.SetParameterBlockConstant(parameters.poses[fixed_pose_index].data());

  auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
  for (std::array<double, 3>& landmark : parameters.landmarks) {
    ordering->AddElementToGroup(landmark.data(), 0);
  }
  for (std::array<double, 7>& pose : parameters.poses) {
    ordering->AddElementToGroup(pose.data(), 1);
  }
  ceres::Solver::Options options = MakeReferenceSolverOptions(ordering);
  ceres::Solver::Summary solver_summary;
  ceres::Solve(options, &problem, &solver_summary);
  if (!solver_summary.IsSolutionUsable()) {
    std::ostringstream failure_report;
    failure_report << "Ceres solve did not produce a usable state: "
                   << solver_summary.BriefReport() << '\n'
                   << solver_summary.FullReport() << "ITERATIONS\n";
    for (const ceres::IterationSummary& iteration :
         solver_summary.iterations) {
      failure_report << iteration.iteration << ' '
                     << (iteration.step_is_valid ? 1 : 0) << ' '
                     << (iteration.step_is_successful ? 1 : 0) << ' '
                     << iteration.cost << ' ' << iteration.cost_change << ' '
                     << iteration.gradient_norm << ' ' << iteration.step_norm
                     << ' ' << iteration.relative_decrease << '\n';
    }
    throw FixtureError(failure_report.str());
  }
  if (!std::isfinite(solver_summary.initial_cost) ||
      !std::isfinite(solver_summary.final_cost)) {
    throw FixtureError("Ceres solve reported nonfinite half squared cost");
  }
  for (std::size_t index = 0; index < parameters.poses.size(); ++index) {
    const std::array<double, 7>& pose = parameters.poses[index];
    const double norm = std::sqrt(pose[0] * pose[0] + pose[1] * pose[1] +
                                  pose[2] * pose[2] + pose[3] * pose[3]);
    if (!std::isfinite(norm) || std::abs(norm - 1.0) > 1e-8) {
      throw FixtureError("Ceres solve produced a non-unit pose quaternion");
    }
    if (index == fixed_pose_index) {
      const Pose& original = fixture.poses[index];
      const double expected[7] = {original.rotation.w(), original.rotation.x(),
                                  original.rotation.y(), original.rotation.z(),
                                  original.translation.x(), original.translation.y(),
                                  original.translation.z()};
      for (std::size_t coordinate = 0; coordinate < 7; ++coordinate) {
        if (DoubleBits(pose[coordinate]) != DoubleBits(expected[coordinate])) {
          throw FixtureError("Ceres solve changed fixed anchor pose");
        }
      }
    }
  }
  const EvaluationSummary final_evaluation =
      EvaluateParameterState(fixture, parameters, nullptr, "FINAL");
  const double half_cost_tolerance =
      1e-6 + 1e-10 * std::max(initial_evaluation.squared_cost,
                              final_evaluation.squared_cost);
  if (std::abs(2.0 * solver_summary.initial_cost -
               initial_evaluation.squared_cost) > half_cost_tolerance ||
      std::abs(2.0 * solver_summary.final_cost -
               final_evaluation.squared_cost) > half_cost_tolerance) {
    throw FixtureError("Ceres half-cost and independent full-cost evaluations "
                       "disagree");
  }
  const double nonincrease_tolerance =
      1e-9 + 1e-12 * std::abs(initial_evaluation.squared_cost);
  if (final_evaluation.squared_cost >
      initial_evaluation.squared_cost + nonincrease_tolerance) {
    throw FixtureError("Ceres solve increased the full squared cost");
  }
  options.linear_solver_ordering.reset();
  return SolveResult{std::move(parameters), initial_evaluation, final_evaluation,
                     std::move(options), std::move(solver_summary)};
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

void WriteExclusiveTextFile(const fs::path& path, std::string_view contents) {
  const int file_descriptor =
      ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (file_descriptor < 0) {
    throw FixtureError("cannot exclusively create self-test file " +
                       path.string() + ": " + std::strerror(errno));
  }
  std::size_t written_total = 0;
  while (written_total < contents.size()) {
    const ssize_t written = ::write(
        file_descriptor, contents.data() + written_total,
        contents.size() - written_total);
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      const int saved_errno = errno;
      ::close(file_descriptor);
      std::error_code cleanup_error;
      fs::remove(path, cleanup_error);
      throw FixtureError("cannot write self-test file " + path.string() +
                         ": " + std::strerror(saved_errno));
    }
    written_total += static_cast<std::size_t>(written);
  }
  if (::close(file_descriptor) != 0) {
    const int saved_errno = errno;
    std::error_code cleanup_error;
    fs::remove(path, cleanup_error);
    throw FixtureError("cannot close self-test file " + path.string() +
                       ": " + std::strerror(saved_errno));
  }
}

EvaluationSummary EvaluateToFile(const Fixture& fixture,
                                 const fs::path& fixture_path,
                                 const fs::path& output_path) {
  ValidateOutputPath(fixture_path, output_path);
  const fs::path staging = MakeStagingPath(output_path);
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
    // preflight.  The staging file and output must share a filesystem.
    std::error_code error;
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

void WriteSolveState(std::ostream& output,
                     const Fixture& fixture,
                     const SolveResult& result) {
  output << std::setprecision(17)
         << "VISLOC_BA_CERES_SOLVE_STATE 1\n"
         << "SOURCE_SHA256 " << fixture.source_sha256 << '\n'
         << "SOURCE_SHA256_CAMERAS " << fixture.source_sha256_cameras << '\n'
         << "SOURCE_SHA256_IMAGES " << fixture.source_sha256_images << '\n'
         << "SOURCE_SHA256_POINTS " << fixture.source_sha256_points << '\n'
         << "SOURCE_SHA256_MANIFEST " << fixture.source_sha256_manifest << '\n'
         << "CERES_VERSION " << CERES_VERSION_STRING << '\n'
         << "FIXED_POSE " << fixture.fixed_pose_id << '\n'
         << "CAMERA_COUNT " << fixture.cameras.size() << '\n'
         << "POSE_COUNT " << fixture.poses.size() << '\n'
         << "LANDMARK_COUNT " << fixture.landmarks.size() << '\n'
         << "OBSERVATION_COUNT " << fixture.observations.size() << '\n'
         << "DECLARED_INITIAL_COST " << fixture.declared_initial_cost << '\n'
         << "DECLARED_INITIAL_COST_BITS "
         << fixture.declared_initial_cost_bits << '\n'
         << "SOLVER_MODE CERES_STANDALONE_REFERENCE\n"
         << "COST_CONVENTION CERES_HALF_SQUARED_INTERNAL_FULL_SQUARED_REPORTED\n";
  WriteReferenceOptions(output, result.solver_options);
  output << "INITIAL_FULL_SQUARED_COST "
         << result.initial_evaluation.squared_cost << '\n'
         << "INITIAL_FULL_SQUARED_COST_BITS "
         << DoubleBits(result.initial_evaluation.squared_cost) << '\n'
         << "INITIAL_OBSERVATIONS " << result.initial_evaluation.observation_count
         << '\n'
         << "INITIAL_ALL_OBSERVATIONS_VALID 1\n"
         << "INITIAL_POSITIVE_DEPTH "
         << result.initial_evaluation.positive_depth_count << '\n'
         << "INITIAL_MIN_DEPTH " << result.initial_evaluation.minimum_depth << '\n'
         << "INITIAL_MAX_DEPTH " << result.initial_evaluation.maximum_depth << '\n'
         << "INITIAL_CERES_EIGEN_MAX_RESIDUAL_ABS_DIFF "
         << result.initial_evaluation.max_ceres_eigen_residual_abs_diff << '\n'
         << "INITIAL_CERES_EIGEN_MAX_DEPTH_ABS_DIFF "
         << result.initial_evaluation.max_ceres_eigen_depth_abs_diff << '\n'
         << "FINAL_FULL_SQUARED_COST " << result.final_evaluation.squared_cost
         << '\n'
         << "FINAL_FULL_SQUARED_COST_BITS "
         << DoubleBits(result.final_evaluation.squared_cost) << '\n'
         << "FINAL_OBSERVATIONS " << result.final_evaluation.observation_count
         << '\n'
         << "FINAL_ALL_OBSERVATIONS_VALID 1\n"
         << "FINAL_POSITIVE_DEPTH "
         << result.final_evaluation.positive_depth_count << '\n'
         << "FINAL_MIN_DEPTH " << result.final_evaluation.minimum_depth << '\n'
         << "FINAL_MAX_DEPTH " << result.final_evaluation.maximum_depth << '\n'
         << "FINAL_CERES_EIGEN_MAX_RESIDUAL_ABS_DIFF "
         << result.final_evaluation.max_ceres_eigen_residual_abs_diff << '\n'
         << "FINAL_CERES_EIGEN_MAX_DEPTH_ABS_DIFF "
         << result.final_evaluation.max_ceres_eigen_depth_abs_diff << '\n'
         << "CERES_SUMMARY_INITIAL_HALF_COST "
         << result.solver_summary.initial_cost << '\n'
         << "CERES_SUMMARY_FINAL_HALF_COST "
         << result.solver_summary.final_cost << '\n'
         << "CERES_SUMMARY_TERMINATION_TYPE "
         << static_cast<int>(result.solver_summary.termination_type) << '\n'
         << "CERES_SUMMARY_IS_SOLUTION_USABLE "
         << (result.solver_summary.IsSolutionUsable() ? 1 : 0) << '\n'
         << "CERES_SUMMARY_NUM_SUCCESSFUL_STEPS "
         << result.solver_summary.num_successful_steps << '\n'
         << "CERES_SUMMARY_NUM_UNSUCCESSFUL_STEPS "
         << result.solver_summary.num_unsuccessful_steps << '\n'
         << "CERES_SUMMARY_TOTAL_TIME_SECONDS "
         << result.solver_summary.total_time_in_seconds << '\n'
         << "CERES_SUMMARY_MESSAGE_BEGIN\n"
         << result.solver_summary.message << '\n'
         << "CERES_SUMMARY_MESSAGE_END\n"
         << "CERES_SUMMARY_FULL_REPORT_BEGIN\n"
         << result.solver_summary.FullReport();
  if (result.solver_summary.FullReport().empty() ||
      result.solver_summary.FullReport().back() != '\n') {
    output << '\n';
  }
  output << "CERES_SUMMARY_FULL_REPORT_END\n"
         << "ITERATION_COUNT " << result.solver_summary.iterations.size() << '\n';
  for (const ceres::IterationSummary& iteration :
       result.solver_summary.iterations) {
    output << "ITERATION " << iteration.iteration << ' '
           << (iteration.step_is_valid ? 1 : 0) << ' '
           << (iteration.step_is_successful ? 1 : 0) << ' '
           << (iteration.step_is_nonmonotonic ? 1 : 0) << ' '
           << iteration.cost << ' ' << iteration.cost_change << ' '
           << iteration.gradient_max_norm << ' ' << iteration.gradient_norm << ' '
           << iteration.step_norm << ' ' << iteration.relative_decrease << ' '
           << iteration.trust_region_radius << ' ' << iteration.eta << ' '
           << iteration.linear_solver_iterations << ' '
           << iteration.iteration_time_in_seconds << ' '
           << iteration.step_solver_time_in_seconds << ' '
           << iteration.cumulative_time_in_seconds << '\n';
  }
  output << "POSE_STATE_BEGIN\n";
  for (std::size_t index = 0; index < fixture.poses.size(); ++index) {
    const std::array<double, 7>& pose = result.parameters.poses[index];
    output << "POSE " << fixture.poses[index].id << ' ' << pose[0] << ' '
           << pose[1] << ' ' << pose[2] << ' ' << pose[3] << ' ' << pose[4]
           << ' ' << pose[5] << ' ' << pose[6] << '\n';
  }
  output << "POSE_STATE_END\nLANDMARK_STATE_BEGIN\n";
  for (std::size_t index = 0; index < fixture.landmarks.size(); ++index) {
    const std::array<double, 3>& landmark = result.parameters.landmarks[index];
    output << "LANDMARK " << fixture.landmarks[index].id << ' ' << landmark[0]
           << ' ' << landmark[1] << ' ' << landmark[2] << '\n';
  }
  output << "LANDMARK_STATE_END\nEND\n";
}

SolveResult SolveToFile(const Fixture& fixture,
                        const fs::path& fixture_path,
                        const fs::path& output_path) {
  ValidateOutputPath(fixture_path, output_path);
  SolveResult result = SolveInMemory(fixture);
  const fs::path staging = MakeStagingPath(output_path);
  bool owns_staging = false;
  try {
    const int file_descriptor =
        ::open(staging.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (file_descriptor < 0) {
      throw FixtureError("cannot exclusively create solve staging file: " +
                         std::string(std::strerror(errno)));
    }
    owns_staging = true;
    {
      ExclusiveFileBuffer buffer(file_descriptor);
      std::ostream output(&buffer);
      WriteSolveState(output, fixture, result);
      output.flush();
      if (!output) {
        throw FixtureError("cannot flush solve staging file");
      }
    }
    std::error_code error;
    fs::create_hard_link(staging, output_path, error);
    if (error) {
      throw FixtureError("cannot publish solve output without overwrite: " +
                         error.message());
    }
    fs::remove(staging, error);
    if (error) {
      throw FixtureError("cannot remove owned solve staging file: " +
                         error.message());
    }
    owns_staging = false;
    std::ifstream state_input(output_path, std::ios::binary);
    if (!state_input) {
      throw FixtureError("published solve output cannot be reopened");
    }
    return result;
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

std::string SyntheticSolveFixture() {
  constexpr const char* kDigest =
      "0000000000000000000000000000000000000000000000000000000000000000";
  Camera camera1;
  camera1.id = 1;
  camera1.width = 640;
  camera1.height = 480;
  camera1.intrinsics = {300.0, 300.0, 320.0, 240.0};
  Camera camera2 = camera1;
  camera2.id = 2;

  Pose true_pose0;
  true_pose0.id = 0;
  true_pose0.rotation = Eigen::Quaterniond::Identity();
  true_pose0.translation = Eigen::Vector3d::Zero();
  Pose true_pose1;
  true_pose1.id = 1;
  true_pose1.rotation = Eigen::Quaterniond(
      Eigen::AngleAxisd(0.025, Eigen::Vector3d(0.0, 1.0, 0.0)));
  true_pose1.translation = Eigen::Vector3d(0.18, -0.10, 0.04);

  Pose initial_pose1;
  initial_pose1.id = 1;
  initial_pose1.rotation = Eigen::Quaterniond(
      Eigen::AngleAxisd(0.060, Eigen::Vector3d(0.0, 1.0, 0.0)));
  initial_pose1.translation = Eigen::Vector3d(0.30, -0.18, 0.12);

  Landmark true_landmark1;
  true_landmark1.id = 1;
  true_landmark1.position = Eigen::Vector3d(0.20, -0.15, 4.20);
  Landmark true_landmark2;
  true_landmark2.id = 2;
  true_landmark2.position = Eigen::Vector3d(-0.35, 0.25, 5.30);
  Landmark true_landmark3;
  true_landmark3.id = 3;
  true_landmark3.position = Eigen::Vector3d(0.55, -0.35, 6.10);
  Landmark true_landmark4;
  true_landmark4.id = 4;
  true_landmark4.position = Eigen::Vector3d(-0.65, -0.20, 4.80);
  Landmark initial_landmark1 = true_landmark1;
  initial_landmark1.position = Eigen::Vector3d(0.28, -0.08, 4.45);
  Landmark initial_landmark2 = true_landmark2;
  initial_landmark2.position = Eigen::Vector3d(-0.22, 0.18, 5.00);
  Landmark initial_landmark3 = true_landmark3;
  initial_landmark3.position = Eigen::Vector3d(0.48, -0.30, 5.80);
  Landmark initial_landmark4 = true_landmark4;
  initial_landmark4.position = Eigen::Vector3d(-0.58, -0.15, 5.10);

  const Eigen::Quaterniond sensor1_rotation = Eigen::Quaterniond::Identity();
  const Eigen::Vector3d sensor1_translation = Eigen::Vector3d::Zero();
  const Eigen::Quaterniond sensor2_rotation(
      Eigen::AngleAxisd(0.18, Eigen::Vector3d(0.2, 0.5, 0.3).normalized()));
  const Eigen::Vector3d sensor2_translation(0.35, -0.03, 0.02);

  const std::array<Pose, 2> true_poses = {true_pose0, true_pose1};
  const std::array<Landmark, 4> true_landmarks = {
      true_landmark1, true_landmark2, true_landmark3, true_landmark4};
  const std::array<Camera, 2> cameras = {camera1, camera2};
  const std::array<Eigen::Quaterniond, 2> sensor_rotations = {
      sensor1_rotation, sensor2_rotation};
  const std::array<Eigen::Vector3d, 2> sensor_translations = {
      sensor1_translation, sensor2_translation};

  auto project = [&](const Pose& pose,
                     const Landmark& landmark,
                     std::size_t camera_index) -> Eigen::Vector2d {
    const Eigen::Vector3d point_sensor =
        sensor_rotations[camera_index] *
            (pose.rotation * landmark.position + pose.translation) +
        sensor_translations[camera_index];
    if (!point_sensor.allFinite() || !(point_sensor.z() > 0.0)) {
      throw FixtureError("synthetic solve fixture has invalid depth");
    }
    const Camera& camera = cameras[camera_index];
    return Eigen::Vector2d(
        camera.intrinsics[0] * point_sensor.x() / point_sensor.z() +
            camera.intrinsics[2],
        camera.intrinsics[1] * point_sensor.y() / point_sensor.z() +
            camera.intrinsics[3]);
  };
  auto write_quaternion = [](std::ostream& output,
                             const Eigen::Quaterniond& rotation) {
    output << rotation.w() << ' ' << rotation.x() << ' ' << rotation.y() << ' '
           << rotation.z();
  };

  std::ostringstream fixture;
  fixture << std::setprecision(17)
          << "VISLOC_BA_ORACLE_FIXTURE 1\n"
          << "SOURCE_SHA256 " << kDigest << "\n"
          << "SOURCE_SHA256_CAMERAS " << kDigest << "\n"
          << "SOURCE_SHA256_IMAGES " << kDigest << "\n"
          << "SOURCE_SHA256_POINTS " << kDigest << "\n"
          << "SOURCE_SHA256_MANIFEST " << kDigest << "\n"
          << "INITIAL_COST 0\n"
          << "INITIAL_COST_BITS 0\n"
          << "CAMERA_COUNT 2\n"
          << "POSE_COUNT 2\n"
          << "LANDMARK_COUNT 4\n"
          << "OBSERVATION_COUNT 16\n"
          << "CAMERA 1 PINHOLE 640 480 4 300 300 320 240\n"
          << "CAMERA 2 PINHOLE 640 480 4 300 300 320 240\n"
          << "POSE 0 ";
  write_quaternion(fixture, true_pose0.rotation);
  fixture << ' ' << true_pose0.translation.x() << ' '
          << true_pose0.translation.y() << ' ' << true_pose0.translation.z()
          << "\nPOSE 1 ";
  write_quaternion(fixture, initial_pose1.rotation);
  fixture << ' ' << initial_pose1.translation.x() << ' '
          << initial_pose1.translation.y() << ' ' << initial_pose1.translation.z()
          << "\nFIXED_POSE 0\n"
          << "LANDMARK 1 " << initial_landmark1.position.x() << ' '
          << initial_landmark1.position.y() << ' ' << initial_landmark1.position.z()
          << "\nLANDMARK 2 " << initial_landmark2.position.x() << ' '
          << initial_landmark2.position.y() << ' ' << initial_landmark2.position.z()
          << "\nLANDMARK 3 " << initial_landmark3.position.x() << ' '
          << initial_landmark3.position.y() << ' ' << initial_landmark3.position.z()
          << "\nLANDMARK 4 " << initial_landmark4.position.x() << ' '
          << initial_landmark4.position.y() << ' ' << initial_landmark4.position.z()
          << '\n';
  for (std::size_t frame_index = 0; frame_index < true_poses.size();
       ++frame_index) {
    for (const Landmark& landmark : true_landmarks) {
      for (std::size_t camera_index = 0; camera_index < cameras.size();
           ++camera_index) {
        const Eigen::Vector2d xy =
            project(true_poses[frame_index], landmark, camera_index);
        fixture << "RIG_OBSERVATION " << frame_index << ' ' << landmark.id << ' '
                << xy.x() << ' ' << xy.y() << ' ' << cameras[camera_index].id
                << ' ';
        write_quaternion(fixture, sensor_rotations[camera_index]);
        fixture << ' ' << sensor_translations[camera_index].x() << ' '
                << sensor_translations[camera_index].y() << ' '
                << sensor_translations[camera_index].z() << '\n';
      }
    }
  }
  fixture << "END\n";
  return fixture.str();
}

Eigen::Vector2d EvaluateSyntheticResidual(const Camera& camera,
                                          const Observation& observation,
                                          const std::array<double, 7>& pose,
                                          const std::array<double, 3>& landmark) {
  Eigen::Quaterniond pose_rotation(pose[0], pose[1], pose[2], pose[3]);
  const double pose_norm = pose_rotation.norm();
  if (!std::isfinite(pose_norm) || pose_norm <= 0.0) {
    throw FixtureError("synthetic derivative evaluation has invalid quaternion");
  }
  pose_rotation.normalize();
  const Eigen::Vector3d point_world(landmark[0], landmark[1], landmark[2]);
  const Eigen::Vector3d pose_translation(pose[4], pose[5], pose[6]);
  const Eigen::Vector3d point_sensor =
      observation.sensor_from_rig_rotation *
          (pose_rotation * point_world + pose_translation) +
      observation.sensor_from_rig_translation;
  if (!point_sensor.allFinite() || !(point_sensor.z() > 0.0)) {
    throw FixtureError("synthetic derivative evaluation has invalid depth");
  }
  return Eigen::Vector2d(
      camera.intrinsics[0] * point_sensor.x() / point_sensor.z() +
          camera.intrinsics[2] - observation.xy.x(),
      camera.intrinsics[1] * point_sensor.y() / point_sensor.z() +
          camera.intrinsics[3] - observation.xy.y());
}

void RequireClose(const std::string& label,
                  double actual,
                  double expected,
                  double relative_tolerance) {
  const double scale =
      std::max({1.0, std::abs(actual), std::abs(expected)});
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > relative_tolerance * scale) {
    throw FixtureError("synthetic derivative mismatch for " + label);
  }
}

void CheckSyntheticJacobians(const Fixture& fixture) {
  const Observation& observation = fixture.observations.at(15);
  const Camera& camera = fixture.cameras.at(
      fixture.camera_index.at(observation.camera_id));
  SolveParameters parameters = MakeSolveParameters(fixture);
  const std::size_t pose_index = fixture.pose_index.at(observation.frame_id);
  const std::size_t landmark_index =
      fixture.landmark_index.at(observation.landmark_id);
  const std::array<double, 7>& pose = parameters.poses[pose_index];
  const std::array<double, 3>& landmark = parameters.landmarks[landmark_index];
  const std::unique_ptr<ceres::CostFunction> cost =
      MakeRigReprojectionCost(camera, observation);
  const double* parameter_blocks[] = {pose.data(), landmark.data()};
  double residual[2] = {0.0, 0.0};
  double pose_jacobian[2 * 7] = {};
  double landmark_jacobian[2 * 3] = {};
  double* jacobians[] = {pose_jacobian, landmark_jacobian};
  if (!cost->Evaluate(parameter_blocks, residual, jacobians)) {
    throw FixtureError("synthetic AutoDiff Jacobian evaluation failed");
  }
  constexpr double kFiniteDifferenceStep = 1e-7;
  for (std::size_t coordinate = 0; coordinate < pose.size(); ++coordinate) {
    std::array<double, 7> plus = pose;
    std::array<double, 7> minus = pose;
    plus[coordinate] += kFiniteDifferenceStep;
    minus[coordinate] -= kFiniteDifferenceStep;
    const Eigen::Vector2d plus_residual =
        EvaluateSyntheticResidual(camera, observation, plus, landmark);
    const Eigen::Vector2d minus_residual =
        EvaluateSyntheticResidual(camera, observation, minus, landmark);
    const Eigen::Vector2d finite_difference =
        (plus_residual - minus_residual) / (2.0 * kFiniteDifferenceStep);
    for (std::size_t row = 0; row < 2; ++row) {
      RequireClose("ambient pose Jacobian", pose_jacobian[row * 7 + coordinate],
                   finite_difference[row], 5e-5);
    }
  }
  for (std::size_t coordinate = 0; coordinate < landmark.size(); ++coordinate) {
    std::array<double, 3> plus = landmark;
    std::array<double, 3> minus = landmark;
    plus[coordinate] += kFiniteDifferenceStep;
    minus[coordinate] -= kFiniteDifferenceStep;
    const Eigen::Vector2d plus_residual =
        EvaluateSyntheticResidual(camera, observation, pose, plus);
    const Eigen::Vector2d minus_residual =
        EvaluateSyntheticResidual(camera, observation, pose, minus);
    const Eigen::Vector2d finite_difference =
        (plus_residual - minus_residual) / (2.0 * kFiniteDifferenceStep);
    for (std::size_t row = 0; row < 2; ++row) {
      RequireClose("landmark Jacobian", landmark_jacobian[row * 3 + coordinate],
                   finite_difference[row], 5e-5);
    }
  }

  using PoseManifold =
      ceres::ProductManifold<ceres::QuaternionManifold,
                             ceres::EuclideanManifold<3>>;
  PoseManifold manifold;
  double plus_jacobian[7 * 6] = {};
  if (!manifold.PlusJacobian(pose.data(), plus_jacobian)) {
    throw FixtureError("synthetic ProductManifold Jacobian evaluation failed");
  }
  constexpr std::size_t kTangentSize = 6;
  for (std::size_t tangent_coordinate = 0;
       tangent_coordinate < kTangentSize; ++tangent_coordinate) {
    std::array<double, kTangentSize> plus_delta{};
    std::array<double, kTangentSize> minus_delta{};
    plus_delta[tangent_coordinate] = kFiniteDifferenceStep;
    minus_delta[tangent_coordinate] = -kFiniteDifferenceStep;
    std::array<double, 7> plus_pose{};
    std::array<double, 7> minus_pose{};
    if (!manifold.Plus(pose.data(), plus_delta.data(), plus_pose.data()) ||
        !manifold.Plus(pose.data(), minus_delta.data(), minus_pose.data())) {
      throw FixtureError("synthetic ProductManifold Plus failed");
    }
    const Eigen::Vector2d plus_residual =
        EvaluateSyntheticResidual(camera, observation, plus_pose, landmark);
    const Eigen::Vector2d minus_residual =
        EvaluateSyntheticResidual(camera, observation, minus_pose, landmark);
    const Eigen::Vector2d finite_difference =
        (plus_residual - minus_residual) / (2.0 * kFiniteDifferenceStep);
    for (std::size_t row = 0; row < 2; ++row) {
      double tangent_derivative = 0.0;
      for (std::size_t ambient_coordinate = 0; ambient_coordinate < 7;
           ++ambient_coordinate) {
        tangent_derivative +=
            pose_jacobian[row * 7 + ambient_coordinate] *
            plus_jacobian[ambient_coordinate * kTangentSize +
                          tangent_coordinate];
      }
      RequireClose("ProductManifold tangent Jacobian", tangent_derivative,
                   finite_difference[row], 5e-5);
    }
  }
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

  // Exercise the full exclusive staging/publication path with a missing
  // staging path.  In particular, symlink_status on a missing path may carry
  // ENOENT in its error_code; O_EXCL is the authoritative race-safe check.
  {
    std::error_code temp_error;
    const fs::path temp_directory = fs::temp_directory_path(temp_error);
    if (temp_error) {
      throw FixtureError("self-test cannot find temporary directory: " +
                         temp_error.message());
    }
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string prefix =
        "ceres-rig-reference-self-test-" + std::to_string(::getpid()) + "-" +
        std::to_string(static_cast<long long>(stamp));
    std::string directory_template =
        (temp_directory / (prefix + "-XXXXXX")).string();
    std::vector<char> directory_buffer(directory_template.begin(),
                                        directory_template.end());
    directory_buffer.push_back('\0');
    char* directory_name = ::mkdtemp(directory_buffer.data());
    if (directory_name == nullptr) {
      throw FixtureError("self-test cannot create private temporary directory: " +
                         std::string(std::strerror(errno)));
    }
    const fs::path owned_directory(directory_name);
    const fs::path fixture_path = owned_directory / "input.fixture";
    const fs::path output_path = owned_directory / "evaluation.dump";
    try {
      WriteExclusiveTextFile(fixture_path, valid);
      std::ifstream persisted_input(fixture_path, std::ios::binary);
      if (!persisted_input) {
        throw FixtureError("self-test cannot reopen synthetic fixture");
      }
      const Fixture persisted_fixture =
          ParseFixture(persisted_input, fixture_path.string());
      const EvaluationSummary file_summary =
          EvaluateToFile(persisted_fixture, fixture_path, output_path);
      if (file_summary.observation_count != 2 ||
          file_summary.positive_depth_count != 2 ||
          file_summary.squared_cost != 0.0) {
        throw FixtureError("self-test EvaluateToFile summary mismatch");
      }
      std::error_code output_error;
      const fs::file_status output_status =
          fs::symlink_status(output_path, output_error);
      if (output_error || !fs::is_regular_file(output_status) ||
          fs::is_symlink(output_status)) {
        throw FixtureError("self-test EvaluateToFile output is not a regular file");
      }
      std::ifstream output_input(output_path, std::ios::binary);
      std::ostringstream output_contents;
      output_contents << output_input.rdbuf();
      if (!output_input.is_open() || output_input.bad() || output_input.fail() ||
          output_contents.str().find("SUMMARY_OBSERVATIONS 2") ==
              std::string::npos ||
          output_contents.str().find("SUMMARY_SQUARED_COST_BITS 0") ==
              std::string::npos) {
        throw FixtureError("self-test EvaluateToFile output mismatch");
      }
      const std::string published_output = output_contents.str();
      bool existing_rejected = false;
      try {
        (void)EvaluateToFile(persisted_fixture, fixture_path, output_path);
      } catch (const FixtureError& error) {
        if (std::string(error.what()).find("output path already exists") ==
            std::string::npos) {
          throw FixtureError("self-test existing-output error mismatch: " +
                             std::string(error.what()));
        }
        existing_rejected = true;
      }
      if (!existing_rejected) {
        throw FixtureError("self-test expected existing output rejection");
      }
      std::ifstream unchanged_input(output_path, std::ios::binary);
      std::ostringstream unchanged_output;
      unchanged_output << unchanged_input.rdbuf();
      if (!unchanged_input.is_open() || unchanged_input.bad() ||
          unchanged_input.fail() || unchanged_output.str() != published_output) {
        throw FixtureError("self-test existing output was modified");
      }

      const fs::path dangling_path = owned_directory / "dangling.dump";
      std::error_code symlink_error;
      fs::create_symlink(owned_directory / "missing-target", dangling_path,
                         symlink_error);
      if (symlink_error) {
        throw FixtureError("self-test cannot create dangling output symlink: " +
                           symlink_error.message());
      }
      bool symlink_rejected = false;
      try {
        (void)EvaluateToFile(persisted_fixture, fixture_path, dangling_path);
      } catch (const FixtureError& error) {
        if (std::string(error.what()).find("output path already exists") ==
            std::string::npos) {
          throw FixtureError("self-test symlink-output error mismatch: " +
                             std::string(error.what()));
        }
        symlink_rejected = true;
      }
      if (!symlink_rejected) {
        throw FixtureError("self-test expected dangling symlink rejection");
      }
    } catch (...) {
      std::error_code cleanup_error;
      fs::remove_all(owned_directory, cleanup_error);
      throw;
    }
    std::error_code cleanup_error;
    fs::remove_all(owned_directory, cleanup_error);
    if (cleanup_error) {
      throw FixtureError("self-test cannot remove private temporary directory: " +
                         cleanup_error.message());
    }
  }

  {
    const std::string solve_text = SyntheticSolveFixture();
    std::istringstream solve_input(solve_text);
    Fixture solve_fixture = ParseFixture(solve_input, "synthetic-solve");
    NullStreamBuffer solve_initial_buffer;
    std::ostream solve_initial_output(&solve_initial_buffer);
    const EvaluationSummary solve_initial =
        Evaluate(solve_fixture, solve_initial_output);
    if (!(solve_initial.squared_cost > 0.0) ||
        solve_initial.observation_count != 16 ||
        solve_initial.positive_depth_count != 16) {
      throw FixtureError("self-test synthetic solve fixture is trivial or invalid");
    }
    solve_fixture.declared_initial_cost = solve_initial.squared_cost;
    solve_fixture.declared_initial_cost_bits =
        DoubleBits(solve_initial.squared_cost);
    CheckSyntheticJacobians(solve_fixture);
    const auto sensor_before = solve_fixture.sensor_from_rig_by_camera;
    const SolveResult first_solve = SolveInMemory(solve_fixture);
    if (!(first_solve.final_evaluation.squared_cost <
          first_solve.initial_evaluation.squared_cost) ||
        first_solve.solver_summary.num_successful_steps <= 0) {
      throw FixtureError("self-test synthetic Ceres solve made no progress");
    }
    const SolveResult second_solve = SolveInMemory(solve_fixture);
    if (DoubleBits(first_solve.final_evaluation.squared_cost) !=
            DoubleBits(second_solve.final_evaluation.squared_cost) ||
        first_solve.parameters.poses.size() != second_solve.parameters.poses.size() ||
        first_solve.parameters.landmarks.size() !=
            second_solve.parameters.landmarks.size()) {
      throw FixtureError("self-test synthetic Ceres solve is not deterministic");
    }
    for (std::size_t index = 0; index < first_solve.parameters.poses.size();
         ++index) {
      for (std::size_t coordinate = 0; coordinate < 7; ++coordinate) {
        if (DoubleBits(first_solve.parameters.poses[index][coordinate]) !=
            DoubleBits(second_solve.parameters.poses[index][coordinate])) {
          throw FixtureError("self-test synthetic pose solve is not deterministic");
        }
      }
    }
    for (std::size_t index = 0; index < first_solve.parameters.landmarks.size();
         ++index) {
      for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
        if (DoubleBits(first_solve.parameters.landmarks[index][coordinate]) !=
            DoubleBits(second_solve.parameters.landmarks[index][coordinate])) {
          throw FixtureError(
              "self-test synthetic landmark solve is not deterministic");
        }
      }
    }
    if (solve_fixture.sensor_from_rig_by_camera.size() != sensor_before.size()) {
      throw FixtureError("self-test synthetic sensor calibration changed");
    }
    for (const auto& sensor : sensor_before) {
      const auto after = solve_fixture.sensor_from_rig_by_camera.find(sensor.first);
      if (after == solve_fixture.sensor_from_rig_by_camera.end() ||
          (sensor.second.first.coeffs() - after->second.first.coeffs())
                  .cwiseAbs()
                  .maxCoeff() != 0.0 ||
          (sensor.second.second - after->second.second).cwiseAbs().maxCoeff() !=
              0.0) {
        throw FixtureError("self-test synthetic sensor calibration changed");
      }
    }

    std::error_code temp_error;
    const fs::path temp_directory = fs::temp_directory_path(temp_error);
    if (temp_error) {
      throw FixtureError("self-test cannot find temporary directory for solve: " +
                         temp_error.message());
    }
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string prefix =
        "ceres-rig-reference-solve-test-" + std::to_string(::getpid()) + "-" +
        std::to_string(static_cast<long long>(stamp));
    std::string directory_template =
        (temp_directory / (prefix + "-XXXXXX")).string();
    std::vector<char> directory_buffer(directory_template.begin(),
                                        directory_template.end());
    directory_buffer.push_back('\0');
    char* directory_name = ::mkdtemp(directory_buffer.data());
    if (directory_name == nullptr) {
      throw FixtureError("self-test cannot create solve temporary directory: " +
                         std::string(std::strerror(errno)));
    }
    const fs::path owned_directory(directory_name);
    const fs::path fixture_path = owned_directory / "input.fixture";
    const fs::path state_path = owned_directory / "solve.state";
    try {
      std::ostringstream persisted_text;
      persisted_text << std::setprecision(17)
                     << solve_text.substr(0, solve_text.find("INITIAL_COST 0"))
                     << "INITIAL_COST " << solve_fixture.declared_initial_cost
                     << "\nINITIAL_COST_BITS "
                     << solve_fixture.declared_initial_cost_bits << '\n'
                     << solve_text.substr(solve_text.find("INITIAL_COST_BITS 0\n") +
                                          std::string("INITIAL_COST_BITS 0\n").size());
      WriteExclusiveTextFile(fixture_path, persisted_text.str());
      const SolveResult published =
          SolveToFile(solve_fixture, fixture_path, state_path);
      if (DoubleBits(published.final_evaluation.squared_cost) !=
          DoubleBits(first_solve.final_evaluation.squared_cost)) {
        throw FixtureError("self-test published solve differs from in-memory solve");
      }
      std::ifstream state_input(state_path, std::ios::binary);
      std::ostringstream state_contents;
      state_contents << state_input.rdbuf();
      if (!state_input.is_open() || state_input.fail() ||
          state_contents.str().find("VISLOC_BA_CERES_SOLVE_STATE 1") ==
              std::string::npos ||
          state_contents.str().find("POSE 0 ") == std::string::npos ||
          state_contents.str().find("POSE 1 ") == std::string::npos ||
          state_contents.str().find("LANDMARK 1 ") == std::string::npos ||
          state_contents.str().find("LANDMARK 2 ") == std::string::npos ||
          state_contents.str().find("ITERATION_COUNT ") == std::string::npos ||
          state_contents.str().find("CERES_SUMMARY_FULL_REPORT_BEGIN") ==
              std::string::npos) {
        throw FixtureError("self-test solve state output mismatch");
      }
      const std::string published_state = state_contents.str();
      bool existing_state_rejected = false;
      try {
        (void)SolveToFile(solve_fixture, fixture_path, state_path);
      } catch (const FixtureError& error) {
        if (std::string(error.what()).find("output path already exists") ==
            std::string::npos) {
          throw FixtureError("self-test solve existing-state error mismatch: " +
                             std::string(error.what()));
        }
        existing_state_rejected = true;
      }
      if (!existing_state_rejected) {
        throw FixtureError("self-test expected existing solve state rejection");
      }
      std::ifstream unchanged_state_input(state_path, std::ios::binary);
      std::ostringstream unchanged_state;
      unchanged_state << unchanged_state_input.rdbuf();
      if (!unchanged_state_input.is_open() || unchanged_state_input.fail() ||
          unchanged_state.str() != published_state) {
        throw FixtureError("self-test existing solve state was modified");
      }
    } catch (...) {
      std::error_code cleanup_error;
      fs::remove_all(owned_directory, cleanup_error);
      throw;
    }
    std::error_code cleanup_error;
    fs::remove_all(owned_directory, cleanup_error);
    if (cleanup_error) {
      throw FixtureError("self-test cannot remove solve temporary directory: " +
                         cleanup_error.message());
    }
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
      if (std::string(error.what()).find("Ceres transform rejected a nonfinite or nonpositive-depth") ==
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
         "  ceres_rig_reference --solve --fixture PATH --state PATH\n";
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
    } else if (argument == "--solve") {
      if (cli.solve) {
        throw FixtureError("duplicate --solve");
      }
      cli.solve = true;
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
    } else if (argument == "--state") {
      if (!cli.state.empty()) {
        throw FixtureError("duplicate --state");
      }
      cli.state = require_value("--state");
    } else if (argument == "--help" || argument == "-h") {
      throw FixtureError(Usage());
    } else {
      throw FixtureError("unknown argument " + argument + "\n" + Usage());
    }
  }
  if (cli.self_test) {
    if (cli.evaluate_only || cli.solve || !cli.fixture.empty() ||
        !cli.dump.empty() || !cli.state.empty()) {
      throw FixtureError("--self-test is standalone\n" + Usage());
    }
    return cli;
  }
  if (cli.evaluate_only && cli.solve) {
    throw FixtureError("--evaluate-only and --solve are mutually exclusive\n" +
                       Usage());
  }
  if (cli.evaluate_only) {
    if (cli.fixture.empty() || cli.dump.empty() || !cli.state.empty()) {
      throw FixtureError("evaluate-only requires --fixture and --dump\n" +
                         Usage());
    }
    return cli;
  }
  if (cli.solve) {
    if (cli.fixture.empty() || cli.state.empty() || !cli.dump.empty()) {
      throw FixtureError("solve requires --fixture and --state\n" + Usage());
    }
    return cli;
  }
  throw FixtureError("one of --evaluate-only or --solve is required\n" +
                     Usage());
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
    if (cli.evaluate_only) {
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
                << " ceres_eigen_max_depth_abs_diff="
                << summary.max_ceres_eigen_depth_abs_diff
                << " cost_delta=" << absolute_delta
                << " min_depth=" << summary.minimum_depth
                << " max_depth=" << summary.maximum_depth
                << " dump=" << cli.dump << '\n';
    } else {
      const SolveResult result = SolveToFile(fixture, cli.fixture, cli.state);
      std::cout << std::setprecision(17)
                << "mode=solve ceres_version=" << CERES_VERSION_STRING
                << " source_sha256=" << fixture.source_sha256
                << " fixed_pose=" << fixture.fixed_pose_id
                << " cameras=" << fixture.cameras.size()
                << " poses=" << fixture.poses.size()
                << " landmarks=" << fixture.landmarks.size()
                << " observations=" << fixture.observations.size()
                << " initial_full_squared_cost="
                << result.initial_evaluation.squared_cost
                << " initial_full_squared_cost_bits="
                << DoubleBits(result.initial_evaluation.squared_cost)
                << " final_full_squared_cost="
                << result.final_evaluation.squared_cost
                << " final_full_squared_cost_bits="
                << DoubleBits(result.final_evaluation.squared_cost)
                << " termination_type="
                << static_cast<int>(result.solver_summary.termination_type)
                << " successful_steps="
                << result.solver_summary.num_successful_steps
                << " iterations=" << result.solver_summary.iterations.size()
                << " state=" << cli.state << '\n';
    }
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
