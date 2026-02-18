// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}

// Fuses the WebNN-specific pattern:
//   DequantizeLinear(3D, UINT4, axis=2) -> Reshape(2D) -> Transpose([1,0])
//   -> [optional Cast] -> MatMul/Gemm
// into a single MatMulNBits node.
//
// This pattern is produced when a quantized model goes through:
//   1) ORT-Web WebNN EP (lowers MatMulNBits to DQ+Reshape+Transpose+MatMul primitives)
//   2) Chromium WebNN backend (converts WebNN ops back to ONNX)
//   3) ORT native graph optimizations (may produce Gemm from MatMul+Add)
class WebNNDQMatMulNBitsFusion : public GraphTransformer {
 public:
  explicit WebNNDQMatMulNBitsFusion(
      int64_t accuracy_level = 4,
      concurrency::ThreadPool* intra_op_thread_pool = nullptr,
      const InlinedHashSet<std::string_view>& compatible_eps = {});

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                   const logging::Logger& logger) const override;

  int64_t accuracy_level_;
  concurrency::ThreadPool* intra_op_thread_pool_;
};

}  // namespace onnxruntime
