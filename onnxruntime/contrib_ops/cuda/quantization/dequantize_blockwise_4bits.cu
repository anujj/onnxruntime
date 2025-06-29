// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <cmath>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <type_traits>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/quantization/dequantize_blockwise.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

__device__ __forceinline__ void DequantizeEightElements(uint32_t values_quant, half scale, half zp, half* output) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * zp;
  half2 zp_adjust2 = {zp_adjust, zp_adjust};

  alignas(16) half2 results[4];
  half v0 = __uint2half_rn(values_quant & 0xF);
  half v1 = __uint2half_rn((values_quant >> 4) & 0xF);
  results[0] = __halves2half2(v0, v1) * scale_half2 + zp_adjust2;

  half v2 = __uint2half_rn((values_quant >> 8) & 0xF);
  half v3 = __uint2half_rn((values_quant >> 12) & 0xF);
  results[1] = __halves2half2(v2, v3) * scale_half2 + zp_adjust2;

  half v4 = __uint2half_rn((values_quant >> 16) & 0xF);
  half v5 = __uint2half_rn((values_quant >> 20) & 0xF);
  results[2] = __halves2half2(v4, v5) * scale_half2 + zp_adjust2;

  half v6 = __uint2half_rn((values_quant >> 24) & 0xF);
  half v7 = __uint2half_rn((values_quant >> 28) & 0xF);
  results[3] = __halves2half2(v6, v7) * scale_half2 + zp_adjust2;
  *(reinterpret_cast<float4*>(output)) = *(reinterpret_cast<float4*>(results));
}

__device__ __forceinline__ void DequantizeEightElements(uint32_t values_quant, __nv_bfloat16 scale, __nv_bfloat16 zp,
                                                        __nv_bfloat16* output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __nv_bfloat162 scale_bf162 = __bfloat162bfloat162(scale);
  __nv_bfloat16 zp_adjust = __hneg(scale) * zp;
  __nv_bfloat162 zp_adjust2 = __bfloat162bfloat162(zp_adjust);

  alignas(16) __nv_bfloat162 results[4];
  __nv_bfloat16 v0 = __uint2bfloat16_rn(values_quant & 0xF);
  __nv_bfloat16 v1 = __uint2bfloat16_rn((values_quant >> 4) & 0xF);
  results[0] = __halves2bfloat162(v0, v1) * scale_bf162 + zp_adjust2;

  __nv_bfloat16 v2 = __uint2bfloat16_rn((values_quant >> 8) & 0xF);
  __nv_bfloat16 v3 = __uint2bfloat16_rn((values_quant >> 12) & 0xF);
  results[1] = __halves2bfloat162(v2, v3) * scale_bf162 + zp_adjust2;

  __nv_bfloat16 v4 = __uint2bfloat16_rn((values_quant >> 16) & 0xF);
  __nv_bfloat16 v5 = __uint2bfloat16_rn((values_quant >> 20) & 0xF);
  results[2] = __halves2bfloat162(v4, v5) * scale_bf162 + zp_adjust2;

  __nv_bfloat16 v6 = __uint2bfloat16_rn((values_quant >> 24) & 0xF);
  __nv_bfloat16 v7 = __uint2bfloat16_rn((values_quant >> 28) & 0xF);
  results[3] = __halves2bfloat162(v6, v7) * scale_bf162 + zp_adjust2;
  *(reinterpret_cast<float4*>(output)) = *(reinterpret_cast<float4*>(results));
#endif
}

__device__ __forceinline__ void DequantizeEightElements(uint32_t values_quant, float scale, float zp, float* output) {
  float zp_adjust = -scale * zp;
  output[0] = float(values_quant & 0xF) * scale + zp_adjust;
  output[1] = float((values_quant >> 4) & 0xF) * scale + zp_adjust;
  output[2] = float((values_quant >> 8) & 0xF) * scale + zp_adjust;
  output[3] = float((values_quant >> 12) & 0xF) * scale + zp_adjust;
  output[4] = float((values_quant >> 16) & 0xF) * scale + zp_adjust;
  output[5] = float((values_quant >> 20) & 0xF) * scale + zp_adjust;
  output[6] = float((values_quant >> 24) & 0xF) * scale + zp_adjust;
  output[7] = float((values_quant >> 28) & 0xF) * scale + zp_adjust;
}

template <class T>
__global__ void Dequantize4BitsKernelReOrder(
    T* output,
    const uint8_t* quant_data,
    const T* scale_data,
    const uint8_t* zero_points,
    const int32_t* reorder_idx,
    int block_size,
    int groups_per_K,
    int groups_per_threadblock,
    int total_groups) {
  constexpr int bits = 4;
  constexpr int element_per_thread = 32 / bits;  // Process 8 elements per thread using uint32_t load
  constexpr int element_per_byte = 8 / bits;
  int group_id = blockIdx.x * groups_per_threadblock + ((threadIdx.x * element_per_thread) / block_size);
  if (group_id >= total_groups) {
    return;
  }

  const int zero_point_shape_x = (groups_per_K + (element_per_byte - 1)) / element_per_byte;
  const int scales_shape_x = groups_per_K;
  int n_idx = group_id / scales_shape_x;
  int kb_idx = group_id % scales_shape_x;
  int element_offset = group_id * block_size + ((threadIdx.x * element_per_thread) & (block_size - 1));
  T* output_i = output + element_offset;
  uint32_t quant_value = *(reinterpret_cast<const uint32_t*>(quant_data + element_offset / element_per_byte));
  const int32_t* reorder_idx_with_off = reorder_idx + kb_idx * block_size + ((threadIdx.x * element_per_thread) & (block_size - 1));
  for (int i = 0; i < element_per_thread; i++) {
    int32_t rid = reorder_idx_with_off[i];
    T scale = *(scale_data + n_idx * scales_shape_x + rid);
    uint8_t zp = 8;  // Default zero point is 1 << (bits - 1)
    if (zero_points) {
      zp = zero_points[n_idx * zero_point_shape_x + rid / element_per_byte];
      zp = (rid & 0x01) ? (zp >> 4) : (zp & 0x0f);
    }

    if constexpr (std::is_same_v<T, half>) {
      T zp_adjust = -scale * __short2half_rn(zp);
      output_i[i] = __uint2half_rn((quant_value >> (4 * i)) & 0xF) * scale + zp_adjust;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      T zp_adjust = __hneg(scale) * __ushort2bfloat16_rn(zp);
      output_i[i] = __uint2bfloat16_rn((quant_value >> (4 * i)) & 0xF) * scale + zp_adjust;
    } else {
      T zp_adjust = -scale * T(zp);
      output_i[i] = T((quant_value >> (4 * i)) & 0xF) * scale + zp_adjust;
    }
  }
}

template <class T, typename ZeroT = uint8_t>
__global__ void Dequantize4BitsKernel(
    T* output,
    const uint8_t* quant_data,
    const T* scale_data,
    const ZeroT* zero_points,
    int block_size,
    int groups_per_K,
    int groups_per_threadblock,
    int total_groups) {
  int block_id = blockIdx.x * groups_per_threadblock + ((threadIdx.x * 8) / block_size);
  if (block_id >= total_groups) {
    return;
  }
  int element_offset = block_id * block_size + ((threadIdx.x * 8) & (block_size - 1));
  uint32_t quant_value = *(reinterpret_cast<const uint32_t*>(quant_data + element_offset / 2));
  T scale = *(scale_data + block_id);
  T zero_point_value;
  if constexpr (std::is_same_v<ZeroT, uint8_t>) {
    const int scales_shape_x = groups_per_K;
    const int zero_point_shape_x = (groups_per_K + 1) / 2;
    int kb_idx = block_id % scales_shape_x;
    int n_idx = block_id / scales_shape_x;
    uint8_t zp = 8;
    if (zero_points) {
      zp = zero_points[n_idx * zero_point_shape_x + kb_idx / 2];
      zp = (kb_idx & 0x01) ? (zp >> 4) : (zp & 0x0f);
    }
    zero_point_value = static_cast<T>(zp);
  } else {
    zero_point_value = zero_points ? *(zero_points + block_id) : static_cast<T>(8);
  }

  output = output + element_offset;
  DequantizeEightElements(quant_value, scale, zero_point_value, output);
}

template <class T, typename ZeroT>
Status Dequantize4Bits(
    T* output,
    const uint8_t* quant_data,
    const T* scales_data,
    const ZeroT* zero_points,  // shape: [N, (block_per_K + 1)/2]
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream) {
  // k is padded and equal to block_per_K * block_size
  ORT_ENFORCE(k % block_size == 0, "k must be a multiplier of block_size");
  constexpr int element_per_thread = 8;

  int groups_per_K = k / block_size;
  int total_groups = n * groups_per_K;  // total elemenets in quant_data
  int groups_per_threadblock = GridDim::maxThreadsPerBlock * element_per_thread / block_size;
  int groups_per_grid = CeilDiv(total_groups, groups_per_threadblock);
  dim3 grid_dim(groups_per_grid);
  dim3 block_dim(GridDim::maxThreadsPerBlock);

  if (!reorder_idx || std::is_same_v<ZeroT, T>) {
    // Launch standard kernel
    Dequantize4BitsKernel<T, ZeroT><<<grid_dim, block_dim, 0, stream>>>(
        output,
        quant_data,
        scales_data,
        zero_points,
        block_size,
        groups_per_K,  // Pass groups_per_K for potential ZP indexing if needed
        groups_per_threadblock,
        total_groups);
  } else {
    // Launch reorder kernel (requires uint8_t zero points as per original structure)
    if constexpr (std::is_same_v<ZeroT, uint8_t>) {
      Dequantize4BitsKernelReOrder<T><<<grid_dim, block_dim, 0, stream>>>(
          output,
          quant_data,
          scales_data,
          (const uint8_t*)zero_points,
          reorder_idx,
          block_size,
          groups_per_K,
          groups_per_threadblock,
          total_groups);
    } else {
      return Status(::onnxruntime::common::ONNXRUNTIME, ::onnxruntime::common::INVALID_ARGUMENT,
                    "Reorder kernel currently expects uint8_t zero points.");
    }
  }

  return CUDA_CALL(cudaGetLastError());  // Check for launch errors
}

template Status Dequantize4Bits<float, uint8_t>(
    float* output,
    const uint8_t* quant_data,
    const float* scales_data,
    const uint8_t* zero_points,
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

template Status Dequantize4Bits<half, uint8_t>(
    half* output,
    const uint8_t* quant_data,
    const half* scales_data,
    const uint8_t* zero_points,
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

template Status Dequantize4Bits<__nv_bfloat16, uint8_t>(
    __nv_bfloat16* output,
    const uint8_t* quant_data,
    const __nv_bfloat16* scales_data,
    const uint8_t* zero_points,
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

template Status Dequantize4Bits<float, float>(
    float* output,
    const uint8_t* quant_data,
    const float* scales_data,
    const float* zero_points,
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

template Status Dequantize4Bits<half, half>(
    half* output,
    const uint8_t* quant_data,
    const half* scales_data,
    const half* zero_points,
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

template Status Dequantize4Bits<__nv_bfloat16, __nv_bfloat16>(
    __nv_bfloat16* output,
    const uint8_t* quant_data,
    const __nv_bfloat16* scales_data,
    const __nv_bfloat16* zero_points,
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

template <
    typename ElementT,
    int32_t block_size,
    int32_t qbits,
    bool Columnwise>
__global__ void dequantizeThread4b(ElementT* dst,
                                   const uint8_t* weights,
                                   const ElementT* scales,
                                   const uint8_t* zero_points,
                                   int rows,
                                   int columns,
                                   int thrd_row_blks) {
  using QuantBlk = typename BlkQuantTraits<ElementT, block_size, qbits, Columnwise>::QuantBlk;
  using ThreadBlk = typename BlkQuantTraits<ElementT, block_size, qbits, Columnwise>::ThreadBlk;

  static_assert(qbits == 4, "Only 4b block quantization is supported by this kernel specialization!!");

  const auto block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto row_blks = (rows + QuantBlk::kRow - 1) / QuantBlk::kRow;

  const auto meta_rows = (rows + QuantBlk::kRow - 1) / QuantBlk::kRow;

  // quantized matrix is stored in column major, packed by column
  const auto q_rows = (meta_rows * QuantBlk::kRow * qbits + 7) / 8;

  int32_t r_blk_idx = static_cast<int32_t>(block_idx % thrd_row_blks);
  int32_t c_blk_idx = static_cast<int32_t>(block_idx / thrd_row_blks);

  int32_t r = r_blk_idx * ThreadBlk::kRow;
  int32_t c = c_blk_idx * ThreadBlk::kColumn;

  int32_t r_end = std::min(r + ThreadBlk::kRow, rows);
  int32_t c_end = std::min(c + ThreadBlk::kColumn, columns);

  // for 4b quant, kPackSize = 2, so we have 2 scales and 2 offsets
  const ElementT scale_buf[2] = {
      scales[(c / QuantBlk::kColumn) * row_blks + r / QuantBlk::kRow],
      ((r / QuantBlk::kRow) < (meta_rows - 1))
          ? scales[(c / QuantBlk::kColumn) * row_blks + r / QuantBlk::kRow + 1]
          : static_cast<ElementT>(0.0f)};
  const uint8_t zp_pair = (zero_points == nullptr)
                              ? 0x88
                              : zero_points[(c / QuantBlk::kColumn) * ((row_blks + 1) / 2) + (r / QuantBlk::kRow) / 2];
  const uint16_t zp_buf[2] = {(uint16_t)(zp_pair & 0x0f), (uint16_t)((zp_pair >> 4) & 0x0f)};
  const ElementT adjust_buf[2] = {(-scale_buf[0]) * static_cast<ElementT>(zp_buf[0]),
                                  (-scale_buf[1]) * static_cast<ElementT>(zp_buf[1])};

  for (int32_t j = c; j < c_end; ++j) {
    const uint8_t* q_ptr = weights + j * q_rows;
    for (int32_t i = r; i < (r_end - 1); i += 2) {
      const auto scale0 = scale_buf[(i - r) / QuantBlk::kRow];
      const auto adjust0 = adjust_buf[(i - r) / QuantBlk::kRow];

      const auto scale1 = scale_buf[(i + 1 - r) / QuantBlk::kRow];
      const auto adjust1 = adjust_buf[(i + 1 - r) / QuantBlk::kRow];

      const auto vi = q_ptr[i / 2];

      if constexpr (std::is_same<ElementT, half>::value) {
        half2 scale_half2 = {scale0, scale1};
        half2 zp_adjust2 = {adjust0, adjust1};

        half2 v = {__ushort2half_rn(vi & 0xF), __ushort2half_rn((vi >> 4) & 0xF)};
        half2 results = v * scale_half2 + zp_adjust2;

        dst[j * rows + i] = results.x;
        dst[j * rows + (i + 1)] = results.y;
      } else if constexpr (std::is_same<ElementT, __nv_bfloat16>::value) {
        __nv_bfloat162 scale_bf162 = {scale0, scale1};
        __nv_bfloat162 zp_adjust2 = {adjust0, adjust1};

        __nv_bfloat162 v = {__ushort2bfloat16_rn(vi & 0xF), __ushort2bfloat16_rn((vi >> 4) & 0xF)};
        __nv_bfloat162 results = v * scale_bf162 + zp_adjust2;

        dst[j * rows + i] = results.x;
        dst[j * rows + (i + 1)] = results.y;
      } else {
        static_assert(std::is_same<ElementT, float>::value, "Only float, half and bfloat16 are supported!");
        const uint8_t vi0 = vi & 0xf;
        const uint8_t vi1 = vi >> 4;
        dst[j * rows + i] = static_cast<float>(vi0) * scale0 + adjust0;
        ;
        dst[j * rows + (i + 1)] = static_cast<float>(vi1) * scale1 + adjust1;
      }
    }

    if ((r_end & 1) && (r_end > r)) {
      const auto scale0 = scale_buf[(r_end - 1 - r) / QuantBlk::kRow];
      const auto adjust0 = adjust_buf[(r_end - 1 - r) / QuantBlk::kRow];

      const auto vi = q_ptr[(r_end - 1) / 2];
      const uint8_t vi0 = vi & 0xf;

      dst[j * rows + (r_end - 1)] = static_cast<ElementT>(vi0) * scale0 + adjust0;
    }
  }
}

template <
    typename ElementT,
    int32_t block_size,
    int32_t qbits,
    bool Columnwise>
static void dequantize4b_generic(ElementT* dst, const uint8_t* weights, const ElementT* scales,
                                 const uint8_t* zero_points, int32_t rows, int32_t columns,
                                 cudaStream_t stream) {
  using ThreadBlk = typename BlkQuantTraits<ElementT, block_size, qbits, Columnwise>::ThreadBlk;

  // Thread partitioning
  const auto thrd_row_blks = (rows + ThreadBlk::kRow - 1) / ThreadBlk::kRow;
  const auto thrd_col_blks = (columns + ThreadBlk::kColumn - 1) / ThreadBlk::kColumn;
  const auto total_thrd_blks = thrd_row_blks * thrd_col_blks;

  const auto grids = (total_thrd_blks + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
  dequantizeThread4b<ElementT, block_size, qbits, Columnwise><<<grids, GridDim::maxThreadsPerBlock, 0, stream>>>(
      dst,
      weights,
      scales,
      zero_points,
      rows,
      columns,
      thrd_row_blks);
}

template <typename T>
Status
DequantizeBlockwise4b(
    T* dst,
    const uint8_t* src,
    const T* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    cudaStream_t stream) {
  switch (block_size) {
    case 16:
      if (columnwise) {
        dequantize4b_generic<T, 16, 4, true>(dst, src, scales, zero_points, rows, columns, stream);
      } else {
        dequantize4b_generic<T, 16, 4, false>(dst, src, scales, zero_points, rows, columns, stream);
      }
      return Status::OK();
    case 32:
      if (columnwise) {
        dequantize4b_generic<T, 32, 4, true>(dst, src, scales, zero_points, rows, columns, stream);
      } else {
        dequantize4b_generic<T, 32, 4, false>(dst, src, scales, zero_points, rows, columns, stream);
      }
      return Status::OK();
    case 64:
      if (columnwise) {
        dequantize4b_generic<T, 64, 4, true>(dst, src, scales, zero_points, rows, columns, stream);
      } else {
        dequantize4b_generic<T, 64, 4, false>(dst, src, scales, zero_points, rows, columns, stream);
      }
      return Status::OK();
    case 128:
      if (columnwise) {
        dequantize4b_generic<T, 128, 4, true>(dst, src, scales, zero_points, rows, columns, stream);
      } else {
        dequantize4b_generic<T, 128, 4, false>(dst, src, scales, zero_points, rows, columns, stream);
      }
      return Status::OK();
    case 256:
      if (columnwise) {
        dequantize4b_generic<T, 256, 4, true>(dst, src, scales, zero_points, rows, columns, stream);
      } else {
        dequantize4b_generic<T, 256, 4, false>(dst, src, scales, zero_points, rows, columns, stream);
      }
      return Status::OK();
    default:
      // Only block size 16, 32, 64, 128, 256 are supported.
      return Status(::onnxruntime::common::ONNXRUNTIME, ::onnxruntime::common::FAIL,
                    "Unsupported block size for blockwise quantization.");
  }
}

// Template instantiations for 4-bit blockwise
template Status DequantizeBlockwise4b<float>(
    float* dst,
    const uint8_t* src,
    const float* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    cudaStream_t stream);

template Status DequantizeBlockwise4b<half>(
    half* dst,
    const uint8_t* src,
    const half* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    cudaStream_t stream);

template Status DequantizeBlockwise4b<__nv_bfloat16>(
    __nv_bfloat16* dst,
    const uint8_t* src,
    const __nv_bfloat16* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
