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
  float* anchors_hwc = nullptr;
  float* level0 = nullptr;
  float* level1 = nullptr;
  float* level0_hwc = nullptr;
  float* level1_hwc = nullptr;
  float* coords = nullptr;
  float* output = nullptr;
  int32_t* targets = nullptr;
  size_t anchors_capacity = 0;
  size_t anchors_hwc_capacity = 0;
  size_t level0_capacity = 0;
  size_t level1_capacity = 0;
  size_t level0_hwc_capacity = 0;
  size_t level1_hwc_capacity = 0;
  size_t coords_capacity = 0;
  size_t output_capacity = 0;
  size_t targets_capacity = 0;
  bool maps_uploaded = false;
  int resident_frames = 0;
  int resident_channels = 0;
  int resident_height0 = 0;
  int resident_width0 = 0;
  int resident_height1 = 0;
  int resident_width1 = 0;
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

__global__ void transpose_chw_to_hwc(const float* source, float* destination,
                                     int batches, int channels, int height,
                                     int width) {
  const size_t total = static_cast<size_t>(batches) * channels * height * width;
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int channel = index % channels;
  size_t rest = index / channels;
  const int x = rest % width;
  rest /= width;
  const int y = rest % height;
  const int batch = rest / height;
  const size_t source_index =
      ((static_cast<size_t>(batch) * channels + channel) * height + y) * width + x;
  destination[index] = source[source_index];
}

__global__ void transpose_ecpp_to_eppc(const float* source, float* destination,
                                       int edges, int channels, int patch) {
  const size_t total = static_cast<size_t>(edges) * channels * patch * patch;
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int channel = index % channels;
  size_t rest = index / channels;
  const int px = rest % patch;
  rest /= patch;
  const int py = rest % patch;
  const int edge = rest / patch;
  const size_t source_index =
      ((static_cast<size_t>(edge) * channels + channel) * patch + py) * patch + px;
  destination[index] = source[source_index];
}

__device__ inline float map_value_hwc(const float* maps, int frame, int channel,
                                      int y, int x, int channels, int height,
                                      int width) {
  if (x < 0 || x >= width || y < 0 || y >= height) return 0.0f;
  const size_t index =
      ((static_cast<size_t>(frame) * height + y) * width + x) * channels + channel;
  return maps[index];
}

__global__ void correlation_warp_kernel(
    const float* anchors, const float* level0, const float* level1,
    const float* coords, const int32_t* targets, float* output, int edges,
    int frames, int channels, int patch, int height0, int width0, int height1,
    int width1, int radius) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int item = blockIdx.x * warps_per_block + warp_in_block;
  const int items = edges * patch * patch;
  if (item >= items) return;

  const int taps = 2 * radius + 1;
  const int px = item % patch;
  const int py = (item / patch) % patch;
  const int edge = item / (patch * patch);
  const int target = targets[edge];
  if (target < 0 || target >= frames) {
    if (lane == 0) {
      for (int value = 0; value < taps * taps * 2; ++value) {
        output[static_cast<size_t>(item) * taps * taps * 2 + value] = 0.0f;
      }
    }
    return;
  }
  const size_t anchor_base = static_cast<size_t>(item) * channels;
  float anchor_values[4];
  #pragma unroll
  for (int group = 0; group < 4; ++group) {
    anchor_values[group] = anchors[anchor_base + lane + group * 32];
  }

  for (int level = 0; level < 2; ++level) {
    const float coordinate_scale = level == 0 ? 1.0f : 0.25f;
    const float center_x =
        coords[((edge * patch + py) * patch + px) * 2] * coordinate_scale;
    const float center_y =
        coords[((edge * patch + py) * patch + px) * 2 + 1] * coordinate_scale;
    const int base_x = static_cast<int>(floorf(center_x));
    const int base_y = static_cast<int>(floorf(center_y));
    const float fx = center_x - static_cast<float>(base_x);
    const float fy = center_y - static_cast<float>(base_y);
    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w01 = fx * (1.0f - fy);
    const float w10 = (1.0f - fx) * fy;
    const float w11 = fx * fy;
    const float* maps = level == 0 ? level0 : level1;
    const int height = level == 0 ? height0 : height1;
    const int width = level == 0 ? width0 : width1;

    for (int tap = 0; tap < taps * taps; ++tap) {
      const int x0 = base_x + tap % taps - radius;
      const int y0 = base_y + tap / taps - radius;
      float sum = 0.0f;
      #pragma unroll
      for (int group = 0; group < 4; ++group) {
        const int channel = lane + group * 32;
        const float sampled =
            w00 * map_value_hwc(maps, target, channel, y0, x0, channels, height, width) +
            w01 * map_value_hwc(maps, target, channel, y0, x0 + 1, channels, height, width) +
            w10 * map_value_hwc(maps, target, channel, y0 + 1, x0, channels, height, width) +
            w11 * map_value_hwc(maps, target, channel, y0 + 1, x0 + 1, channels, height, width);
        sum += anchor_values[group] * sampled;
      }
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
      }
      if (lane == 0) {
        const size_t output_index =
            (static_cast<size_t>(item) * taps * taps + tap) * 2 + level;
        output[output_index] = sum / sqrtf(static_cast<float>(channels));
      }
    }
  }
}

void release(Context* context) {
  cudaFree(context->anchors);
  cudaFree(context->anchors_hwc);
  cudaFree(context->level0);
  cudaFree(context->level1);
  cudaFree(context->level0_hwc);
  cudaFree(context->level1_hwc);
  cudaFree(context->coords);
  cudaFree(context->output);
  cudaFree(context->targets);
}

}  // namespace

VISLOC_EXPORT uint32_t visloc_dpvo_corr_abi_version() { return 2; }

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
    int upload_frames, float* device_elapsed_ms) {
  if (!opaque || !anchors || !level0_frames || !level1_frames || !coords ||
      !targets || !output || edges <= 0 || frames <= 0 || channels != 128 ||
      patch != 3 || height0 <= 1 || width0 <= 1 || height1 <= 1 ||
      width1 <= 1 || radius < 0) {
    return 1;
  }
  Context* context = static_cast<Context*>(opaque);
  context->error[0] = '\0';
  if (!upload_frames &&
      (!context->maps_uploaded || context->resident_frames != frames ||
       context->resident_channels != channels ||
       context->resident_height0 != height0 ||
       context->resident_width0 != width0 ||
       context->resident_height1 != height1 ||
       context->resident_width1 != width1)) {
    std::snprintf(context->error, sizeof(context->error),
                  "resident feature maps are absent or have different dimensions");
    return 7;
  }
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
      !reserve(reinterpret_cast<void**>(&context->anchors_hwc),
               &context->anchors_hwc_capacity, anchors_bytes, context,
               "anchors_hwc") ||
      !reserve(reinterpret_cast<void**>(&context->level0),
               &context->level0_capacity, level0_bytes, context, "level0") ||
      !reserve(reinterpret_cast<void**>(&context->level1),
               &context->level1_capacity, level1_bytes, context, "level1") ||
      !reserve(reinterpret_cast<void**>(&context->level0_hwc),
               &context->level0_hwc_capacity, level0_bytes, context,
               "level0_hwc") ||
      !reserve(reinterpret_cast<void**>(&context->level1_hwc),
               &context->level1_hwc_capacity, level1_bytes, context,
               "level1_hwc") ||
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
  if (upload_frames) {
    for (int frame = 0; frame < frames; ++frame) {
      if (!copy_to_device(context->level0 + static_cast<size_t>(frame) *
                                                level0_frame_bytes / sizeof(float),
                          level0_frames[frame], level0_frame_bytes, context,
                          "level0 frame") ||
          !copy_to_device(context->level1 + static_cast<size_t>(frame) *
                                                level1_frame_bytes / sizeof(float),
                          level1_frames[frame], level1_frame_bytes, context,
                          "level1 frame")) {
        context->maps_uploaded = false;
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
        return 4;
      }
    }
    const size_t level0_elements = level0_bytes / sizeof(float);
    const size_t level1_elements = level1_bytes / sizeof(float);
    transpose_chw_to_hwc<<<(level0_elements + 255) / 256, 256>>>(
        context->level0, context->level0_hwc, frames, channels, height0, width0);
    transpose_chw_to_hwc<<<(level1_elements + 255) / 256, 256>>>(
        context->level1, context->level1_hwc, frames, channels, height1, width1);
    context->maps_uploaded = true;
    context->resident_frames = frames;
    context->resident_channels = channels;
    context->resident_height0 = height0;
    context->resident_width0 = width0;
    context->resident_height1 = height1;
    context->resident_width1 = width1;
  }

  const size_t anchor_elements = anchors_bytes / sizeof(float);
  transpose_ecpp_to_eppc<<<(anchor_elements + 255) / 256, 256>>>(
      context->anchors, context->anchors_hwc, edges, channels, patch);
  const int items = edges * patch * patch;
  constexpr int correlation_threads = 128;
  constexpr int warps_per_block = correlation_threads / 32;
  correlation_warp_kernel<<<(items + warps_per_block - 1) / warps_per_block,
                            correlation_threads>>>(
      context->anchors_hwc, context->level0_hwc, context->level1_hwc, context->coords,
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
