// SPDX-License-Identifier: MIT OR Apache-2.0
// Batched indexed DPVO correlation for inference-only visloc-rs use.
// This is an independent implementation of the correlation contract
// documented in crates/vision/src/dpvo/correlation.rs; it does not depend on
// PyTorch or copy upstream DPVO's extension source.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#define VISLOC_EXPORT extern "C" __declspec(dllexport)
#else
#define VISLOC_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct Context {
  float* anchors = nullptr;
  float* level0 = nullptr;
  float* level1 = nullptr;
  float* coords = nullptr;
  float* output = nullptr;
  int32_t* targets = nullptr;
  size_t anchors_capacity = 0;
  size_t level0_capacity = 0;
  size_t level1_capacity = 0;
  size_t coords_capacity = 0;
  size_t output_capacity = 0;
  size_t targets_capacity = 0;
  char error[512] = {};
};

bool reserve(void** pointer, size_t* capacity, size_t bytes, Context* context,
             const char* label) {
  if (*capacity >= bytes) return true;
  if (*pointer) cudaFree(*pointer);
  const cudaError_t status = cudaMalloc(pointer, bytes);
  if (status != cudaSuccess) {
    std::snprintf(context->error, sizeof(context->error),
                  "cudaMalloc(%s, %zu): %s", label, bytes,
                  cudaGetErrorString(status));
    *pointer = nullptr;
    *capacity = 0;
    return false;
  }
  *capacity = bytes;
  return true;
}

bool copy_to_device(void* destination, const void* source, size_t bytes,
                    Context* context, const char* label) {
  const cudaError_t status =
      cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice);
  if (status == cudaSuccess) return true;
  std::snprintf(context->error, sizeof(context->error),
                "cudaMemcpy H2D(%s, %zu): %s", label, bytes,
                cudaGetErrorString(status));
  return false;
}

__device__ inline float map_value(const float* maps, int frame, int channel,
                                  int y, int x, int channels, int height,
                                  int width) {
  if (x < 0 || x >= width || y < 0 || y >= height) return 0.0f;
  const size_t index =
      ((static_cast<size_t>(frame) * channels + channel) * height + y) * width + x;
  return maps[index];
}

__global__ void correlation_kernel(
    const float* anchors, const float* level0, const float* level1,
    const float* coords, const int32_t* targets, float* output, int edges,
    int frames, int channels, int patch, int height0, int width0, int height1,
    int width1, int radius) {
  const int taps = 2 * radius + 1;
  const int values_per_edge = patch * patch * taps * taps * 2;
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = edges * values_per_edge;
  if (linear >= total) return;

  const int output_index = linear;
  const int level = linear % 2;
  linear /= 2;
  const int tap = linear % (taps * taps);
  linear /= taps * taps;
  const int px = linear % patch;
  linear /= patch;
  const int py = linear % patch;
  const int edge = linear / patch;
  const int target = targets[edge];
  if (target < 0 || target >= frames) {
    output[output_index] = 0.0f;
    return;
  }

  const int tx = tap % taps;
  const int ty = tap / taps;
  const float scale = level == 0 ? 1.0f : 0.25f;
  const float center_x = coords[((edge * patch + py) * patch + px) * 2] * scale;
  const float center_y = coords[((edge * patch + py) * patch + px) * 2 + 1] * scale;
  const float sample_x = center_x + static_cast<float>(tx - radius);
  const float sample_y = center_y + static_cast<float>(ty - radius);
  const int x0 = static_cast<int>(floorf(sample_x));
  const int y0 = static_cast<int>(floorf(sample_y));
  const float fx = sample_x - static_cast<float>(x0);
  const float fy = sample_y - static_cast<float>(y0);
  const float* maps = level == 0 ? level0 : level1;
  const int height = level == 0 ? height0 : height1;
  const int width = level == 0 ? width0 : width1;

  float sum = 0.0f;
  for (int channel = 0; channel < channels; ++channel) {
    const size_t anchor_index =
        ((static_cast<size_t>(edge) * channels + channel) * patch + py) * patch + px;
    const float sampled =
        (1.0f - fx) * (1.0f - fy) *
            map_value(maps, target, channel, y0, x0, channels, height, width) +
        fx * (1.0f - fy) *
            map_value(maps, target, channel, y0, x0 + 1, channels, height, width) +
        (1.0f - fx) * fy *
            map_value(maps, target, channel, y0 + 1, x0, channels, height, width) +
        fx * fy * map_value(maps, target, channel, y0 + 1, x0 + 1,
                            channels, height, width);
    sum += anchors[anchor_index] * sampled;
  }
  output[output_index] = sum / sqrtf(static_cast<float>(channels));
}

void release(Context* context) {
  cudaFree(context->anchors);
  cudaFree(context->level0);
  cudaFree(context->level1);
  cudaFree(context->coords);
  cudaFree(context->output);
  cudaFree(context->targets);
}

}  // namespace

VISLOC_EXPORT uint32_t visloc_dpvo_corr_abi_version() { return 1; }

VISLOC_EXPORT void* visloc_dpvo_corr_create() { return new Context(); }

VISLOC_EXPORT void visloc_dpvo_corr_destroy(void* opaque) {
  if (!opaque) return;
  Context* context = static_cast<Context*>(opaque);
  release(context);
  delete context;
}

VISLOC_EXPORT const char* visloc_dpvo_corr_last_error(void* opaque) {
  if (!opaque) return "null context";
  return static_cast<Context*>(opaque)->error;
}

VISLOC_EXPORT int visloc_dpvo_corr_run(
    void* opaque, const float* anchors, const float* const* level0_frames,
    const float* const* level1_frames, const float* coords,
    const int32_t* targets, float* output, int edges, int frames, int channels,
    int patch, int height0, int width0, int height1, int width1, int radius,
    float* device_elapsed_ms) {
  if (!opaque || !anchors || !level0_frames || !level1_frames || !coords ||
      !targets || !output || edges <= 0 || frames <= 0 || channels <= 0 ||
      patch <= 0 || height0 <= 1 || width0 <= 1 || height1 <= 1 ||
      width1 <= 1 || radius < 0) {
    return 1;
  }
  Context* context = static_cast<Context*>(opaque);
  context->error[0] = '\0';
  const int taps = 2 * radius + 1;
  const size_t anchors_bytes =
      static_cast<size_t>(edges) * channels * patch * patch * sizeof(float);
  const size_t level0_frame_bytes =
      static_cast<size_t>(channels) * height0 * width0 * sizeof(float);
  const size_t level1_frame_bytes =
      static_cast<size_t>(channels) * height1 * width1 * sizeof(float);
  const size_t level0_bytes = static_cast<size_t>(frames) * level0_frame_bytes;
  const size_t level1_bytes = static_cast<size_t>(frames) * level1_frame_bytes;
  const size_t coords_bytes =
      static_cast<size_t>(edges) * patch * patch * 2 * sizeof(float);
  const size_t targets_bytes = static_cast<size_t>(edges) * sizeof(int32_t);
  const size_t output_bytes = static_cast<size_t>(edges) * patch * patch *
                              taps * taps * 2 * sizeof(float);

  if (!reserve(reinterpret_cast<void**>(&context->anchors),
               &context->anchors_capacity, anchors_bytes, context, "anchors") ||
      !reserve(reinterpret_cast<void**>(&context->level0),
               &context->level0_capacity, level0_bytes, context, "level0") ||
      !reserve(reinterpret_cast<void**>(&context->level1),
               &context->level1_capacity, level1_bytes, context, "level1") ||
      !reserve(reinterpret_cast<void**>(&context->coords),
               &context->coords_capacity, coords_bytes, context, "coords") ||
      !reserve(reinterpret_cast<void**>(&context->targets),
               &context->targets_capacity, targets_bytes, context, "targets") ||
      !reserve(reinterpret_cast<void**>(&context->output),
               &context->output_capacity, output_bytes, context, "output")) {
    return 2;
  }

  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  cudaEventRecord(begin);
  if (!copy_to_device(context->anchors, anchors, anchors_bytes, context,
                      "anchors") ||
      !copy_to_device(context->coords, coords, coords_bytes, context,
                      "coords") ||
      !copy_to_device(context->targets, targets, targets_bytes, context,
                      "targets")) {
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return 3;
  }
  for (int frame = 0; frame < frames; ++frame) {
    if (!copy_to_device(context->level0 + static_cast<size_t>(frame) *
                                              level0_frame_bytes / sizeof(float),
                        level0_frames[frame], level0_frame_bytes, context,
                        "level0 frame") ||
        !copy_to_device(context->level1 + static_cast<size_t>(frame) *
                                              level1_frame_bytes / sizeof(float),
                        level1_frames[frame], level1_frame_bytes, context,
                        "level1 frame")) {
      cudaEventDestroy(begin);
      cudaEventDestroy(end);
      return 4;
    }
  }

  const int values_per_edge = patch * patch * taps * taps * 2;
  const int total = edges * values_per_edge;
  correlation_kernel<<<(total + 255) / 256, 256>>>(
      context->anchors, context->level0, context->level1, context->coords,
      context->targets, context->output, edges, frames, channels, patch,
      height0, width0, height1, width1, radius);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::snprintf(context->error, sizeof(context->error),
                  "correlation kernel launch: %s", cudaGetErrorString(status));
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return 5;
  }
  status = cudaMemcpy(output, context->output, output_bytes, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    std::snprintf(context->error, sizeof(context->error),
                  "cudaMemcpy D2H(output): %s", cudaGetErrorString(status));
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return 6;
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  if (device_elapsed_ms) cudaEventElapsedTime(device_elapsed_ms, begin, end);
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return 0;
}
