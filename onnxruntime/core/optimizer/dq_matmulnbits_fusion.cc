// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/dq_matmulnbits_fusion.h"

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

#include <cmath>
#include <cstring>
#include <optional>
#include <unordered_set>

namespace onnxruntime {

namespace {

bool IsUniformPackedUint4Value(const Initializer& init, uint8_t expected_nibble) {
  if (init.data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT4) {
    return false;
  }

  const size_t values_count = static_cast<size_t>(init.size());
  if (values_count == 0) {
    return false;
  }

  const auto packed = init.DataAsByteSpan();
  const uint8_t expected = static_cast<uint8_t>(expected_nibble & 0x0F);
  for (size_t i = 0; i < values_count; ++i) {
    const uint8_t byte = packed[i / 2];
    const uint8_t value = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    if (value != expected) {
      return false;
    }
  }

  return true;
}

bool HasRank2Shape(const ONNX_NAMESPACE::TensorProto& tp, int64_t dim0, int64_t dim1) {
  return tp.dims_size() == 2 && tp.dims(0) == dim0 && tp.dims(1) == dim1;
}

uint8_t GetPackedUint4Element(const uint8_t* packed, size_t index) {
  const uint8_t packed_byte = packed[index / 2];
  return (index % 2 == 0) ? static_cast<uint8_t>(packed_byte & 0x0F)
                          : static_cast<uint8_t>((packed_byte >> 4) & 0x0F);
}

void PackUint4Rows(const Initializer& src, int64_t rows, int64_t cols, uint8_t* dst) {
  const int64_t row_bytes = (cols + 1) / 2;
  memset(dst, 0, static_cast<size_t>(rows * row_bytes));

  const auto src_packed = src.DataAsByteSpan();
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      const size_t src_index = static_cast<size_t>(r * cols + c);
      const uint8_t value = GetPackedUint4Element(src_packed.data(), src_index);

      const size_t dst_index = static_cast<size_t>(r * row_bytes + c / 2);
      if ((c & 1) == 0) {
        dst[dst_index] = value;
      } else {
        dst[dst_index] = static_cast<uint8_t>(dst[dst_index] | (value << 4));
      }
    }
  }
}

// Transpose and pack UINT4 weights from DQ axis=0 layout [K, N] to MatMulNBits layout [N, k_blocks, blob_size].
// Source: row-major UINT4 with quantization along K (axis=0), shape [K, N].
// Dest: UINT8 [N, k_blocks, block_size/2] where each byte packs two 4-bit weights.
void TransposePackWeightsAxis0(
    const uint8_t* src_packed, int64_t K, int64_t N, int64_t block_size,
    uint8_t* dst) {
  const int64_t k_blocks = (K + block_size - 1) / block_size;
  const int64_t blob_size = block_size / 2;
  memset(dst, 0, static_cast<size_t>(N * k_blocks * blob_size));

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      const uint8_t val = GetPackedUint4Element(src_packed, static_cast<size_t>(k * N + n));

      const int64_t kb = k / block_size;
      const int64_t off = k % block_size;
      const size_t dst_byte = static_cast<size_t>(n * k_blocks * blob_size + kb * blob_size + off / 2);
      if (off % 2 == 0) {
        dst[dst_byte] = static_cast<uint8_t>((dst[dst_byte] & 0xF0) | val);
      } else {
        dst[dst_byte] = static_cast<uint8_t>((dst[dst_byte] & 0x0F) | (val << 4));
      }
    }
  }
}

// Transpose and pack UINT4 zero points from DQ axis=0 layout [k_blocks, N] to
// MatMulNBits layout UINT8 [N, ceil(k_blocks/2)].
void TransposePackZPAxis0(
    const uint8_t* src_packed, int64_t k_blocks, int64_t N,
    uint8_t* dst) {
  const int64_t zp_bytes_per_n = (k_blocks + 1) / 2;
  memset(dst, 0, static_cast<size_t>(N * zp_bytes_per_n));

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t kb = 0; kb < k_blocks; ++kb) {
      const uint8_t val = GetPackedUint4Element(src_packed, static_cast<size_t>(kb * N + n));

      const size_t dst_byte = static_cast<size_t>(n * zp_bytes_per_n + kb / 2);
      if (kb % 2 == 0) {
        dst[dst_byte] = static_cast<uint8_t>((dst[dst_byte] & 0xF0) | val);
      } else {
        dst[dst_byte] = static_cast<uint8_t>((dst[dst_byte] & 0x0F) | (val << 4));
      }
    }
  }
}

}  // namespace

DQMatMulNBitsFusion::DQMatMulNBitsFusion(
    int64_t accuracy_level,
    concurrency::ThreadPool* intra_op_thread_pool,
    const InlinedHashSet<std::string_view>& compatible_eps)
    : GraphTransformer("DQMatMulNBitsFusion", compatible_eps),
      accuracy_level_(accuracy_level),
      intra_op_thread_pool_(intra_op_thread_pool) {
  ORT_ENFORCE(accuracy_level_ >= 0 && accuracy_level_ <= 4,
              "MatMulNBits accuracy level must be between 0 and 4");
}

Status DQMatMulNBitsFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                      const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // Collect nodes to remove after iteration (cannot mutate graph during iteration)
  struct FusionMatch {
    NodeIndex matmul_idx;
    std::optional<NodeIndex> cast_idx;
    NodeIndex transpose_idx;
    NodeIndex reshape_idx;
    NodeIndex dq_idx;
  };
  std::vector<FusionMatch> matches;

  for (auto node_index : node_topology_list) {
    auto* node = graph.GetNode(node_index);
    if (!node) continue;
    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    // Match MatMul or Gemm nodes (MatMulAddFusion may have fused MatMul+Add -> Gemm)
    if (node->OpType() != "MatMul" && node->OpType() != "Gemm") {
      continue;
    }

    // Walk backwards from MatMul/Gemm input[1]:
    // expect [optional Cast] -> Transpose -> Reshape -> DQ
    const auto& mm_inputs = node->InputDefs();
    if (mm_inputs.size() < 2 || !mm_inputs[1] || !mm_inputs[1]->Exists()) continue;

    const Node* cast_node = nullptr;
    const Node* transpose_node = graph.GetProducerNode(mm_inputs[1]->Name());
    if (transpose_node && transpose_node->OpType() == "Cast") {
      cast_node = transpose_node;
      if (cast_node->GetOutputEdgesCount() != 1) continue;
      const auto& cast_inputs = cast_node->InputDefs();
      if (cast_inputs.empty() || !cast_inputs[0] || !cast_inputs[0]->Exists()) continue;
      transpose_node = graph.GetProducerNode(cast_inputs[0]->Name());
    }

    if (!transpose_node || transpose_node->OpType() != "Transpose") continue;
    if (transpose_node->GetOutputEdgesCount() != 1) continue;

    // Find Reshape node producing Transpose input
    const auto& tp_inputs = transpose_node->InputDefs();
    if (tp_inputs.empty() || !tp_inputs[0] || !tp_inputs[0]->Exists()) continue;
    const Node* reshape_node = graph.GetProducerNode(tp_inputs[0]->Name());
    if (!reshape_node || reshape_node->OpType() != "Reshape") continue;
    if (reshape_node->GetOutputEdgesCount() != 1) continue;

    // Find DQ node
    const auto& reshape_inputs = reshape_node->InputDefs();
    if (reshape_inputs.empty() || !reshape_inputs[0] || !reshape_inputs[0]->Exists()) continue;
    const Node* dq_node = graph.GetProducerNode(reshape_inputs[0]->Name());
    if (!dq_node || dq_node->OpType() != "DequantizeLinear") continue;
    if (dq_node->GetOutputEdgesCount() != 1) continue;

    // Validate DQ attributes
    const auto& dq_attrs = dq_node->GetAttributes();
    {
      auto it = dq_attrs.find("axis");
      if (it == dq_attrs.end() || it->second.i() != 2) continue;
    }
    int64_t block_size = 0;
    {
      auto it = dq_attrs.find("block_size");
      if (it == dq_attrs.end()) continue;
      block_size = it->second.i();
      if (block_size < 16 || ((block_size - 1) & block_size)) continue;
    }

    // Validate DQ weight type and constant initializer.
    const auto* weight_arg = dq_node->InputDefs()[0];
    if (!weight_arg || !weight_arg->Exists()) continue;
    const auto* weight_const_tp = graph.GetConstantInitializer(weight_arg->Name(), true);
    if (!weight_const_tp) continue;
    if (weight_const_tp->data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT4) continue;
    if (weight_const_tp->dims_size() != 3) continue;
    const int64_t N = weight_const_tp->dims(0);
    const int64_t blocks = weight_const_tp->dims(1);
    const int64_t bs_dim = weight_const_tp->dims(2);
    if (N <= 0 || blocks <= 0 || bs_dim <= 0) continue;
    if (bs_dim != block_size) continue;
    const int64_t K = blocks * bs_dim;

    // Validate DQ scale type and constant initializer.
    const auto* scale_arg = dq_node->InputDefs()[1];
    if (!scale_arg || !scale_arg->Exists()) continue;
    const auto* scale_const_tp = graph.GetConstantInitializer(scale_arg->Name(), true);
    if (!scale_const_tp) continue;
    int32_t dt_scale = scale_const_tp->data_type();
    if (dt_scale != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
        dt_scale != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) continue;

    // Activation and scales must have the same type for MatMulNBits.
    const auto* a_arg = mm_inputs[0];
    if (!a_arg || !a_arg->TypeAsProto()) continue;
    int32_t dt_a = a_arg->TypeAsProto()->tensor_type().elem_type();
    if (dt_a != dt_scale) continue;

    // Validate Reshape target is exactly [N, K] (allowing ONNX 0/-1 conventions).
    const auto* reshape_shape_arg =
        reshape_node->InputDefs().size() > 1 ? reshape_node->InputDefs()[1] : nullptr;
    if (!reshape_shape_arg || !reshape_shape_arg->Exists()) continue;
    const auto* reshape_shape_tp = graph.GetConstantInitializer(reshape_shape_arg->Name(), true);
    if (!reshape_shape_tp) continue;

    Initializer reshape_shape_init(graph, *reshape_shape_tp, graph.ModelPath());
    if (reshape_shape_init.size() != 2) continue;

    int64_t reshape_dim0 = 0;
    int64_t reshape_dim1 = 0;
    if (reshape_shape_init.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      const auto* shape_data = reshape_shape_init.data<int64_t>();
      reshape_dim0 = shape_data[0];
      reshape_dim1 = shape_data[1];
    } else if (reshape_shape_init.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
      const auto* shape_data = reshape_shape_init.data<int32_t>();
      reshape_dim0 = shape_data[0];
      reshape_dim1 = shape_data[1];
    } else {
      continue;
    }

    auto resolve_reshape_dim = [](int64_t dim, int64_t expected) -> std::optional<int64_t> {
      if (dim == expected || dim == 0 || dim == -1) {
        return expected;
      }
      return std::nullopt;
    };
    const auto resolved_reshape_dim0 = resolve_reshape_dim(reshape_dim0, N);
    const auto resolved_reshape_dim1 = resolve_reshape_dim(reshape_dim1, K);
    if (!resolved_reshape_dim0 || !resolved_reshape_dim1 ||
        *resolved_reshape_dim0 != N || *resolved_reshape_dim1 != K) {
      continue;
    }

    // Validate Transpose perm is exactly [1, 0] (or default reverse for rank-2).
    if (const auto* perm_attr = graph_utils::GetNodeAttribute(*transpose_node, "perm")) {
      if (perm_attr->ints_size() != 2 || perm_attr->ints(0) != 1 || perm_attr->ints(1) != 0) {
        continue;
      }
    }

    // Validate MatMul/Gemm K/N contract against the transformed weight shape.
    if (const auto* b_shape = mm_inputs[1]->Shape(); b_shape && b_shape->dim_size() == 2 &&
        utils::HasDimValue(b_shape->dim(0)) && utils::HasDimValue(b_shape->dim(1)) &&
        (b_shape->dim(0).dim_value() != K || b_shape->dim(1).dim_value() != N)) {
      continue;
    }

    if (const auto* a_shape = mm_inputs[0] ? mm_inputs[0]->Shape() : nullptr;
        a_shape && a_shape->dim_size() >= 1) {
      const int last_a_dim_idx = a_shape->dim_size() - 1;
      if (utils::HasDimValue(a_shape->dim(last_a_dim_idx)) &&
          a_shape->dim(last_a_dim_idx).dim_value() != K) {
        continue;
      }
    }

    const auto* y_shape = node->OutputDefs().empty() ? nullptr : node->OutputDefs()[0]->Shape();
    if (y_shape && y_shape->dim_size() >= 1) {
      const int last_y_dim_idx = y_shape->dim_size() - 1;
      if (utils::HasDimValue(y_shape->dim(last_y_dim_idx)) &&
          y_shape->dim(last_y_dim_idx).dim_value() != N) {
        continue;
      }
    }

    // Validate Gemm attributes and bias shape (if present).
    // We only support Gemm forms equivalent to MatMul (+ optional 1D bias):
    //   alpha=1, beta=1, transA=0, transB=0.
    if (node->OpType() == "Gemm") {
      if (const auto* alpha_attr = graph_utils::GetNodeAttribute(*node, "alpha");
          alpha_attr && std::abs(alpha_attr->f() - 1.0f) > 1e-6f) {
        continue;
      }

      if (const auto* beta_attr = graph_utils::GetNodeAttribute(*node, "beta");
          beta_attr && std::abs(beta_attr->f() - 1.0f) > 1e-6f) {
        continue;
      }

      if (const auto* trans_a_attr = graph_utils::GetNodeAttribute(*node, "transA");
          trans_a_attr && trans_a_attr->i() != 0) {
        continue;
      }

      if (const auto* trans_b_attr = graph_utils::GetNodeAttribute(*node, "transB");
          trans_b_attr && trans_b_attr->i() != 0) {
        continue;
      }

      // If Gemm has bias input C, it must be a 1D bias of shape [N].
      if (mm_inputs.size() > 2 && mm_inputs[2] && mm_inputs[2]->Exists()) {
        const auto* bias_shape = mm_inputs[2]->Shape();
        if (!bias_shape || bias_shape->dim_size() != 1 ||
            !utils::HasDimValue(bias_shape->dim(0))) {
          continue;
        }
        if (bias_shape->dim(0).dim_value() != N) {
          continue;
        }
      }
    }

    // Validate optional Cast is type-preserving.
    if (cast_node) {
      const auto* cast_in = cast_node->InputDefs().empty() ? nullptr : cast_node->InputDefs()[0];
      const auto* cast_out = cast_node->OutputDefs().empty() ? nullptr : cast_node->OutputDefs()[0];
      if (!cast_in || !cast_out || !cast_in->TypeAsProto() || !cast_out->TypeAsProto()) continue;
      if (cast_in->TypeAsProto()->tensor_type().elem_type() !=
          cast_out->TypeAsProto()->tensor_type().elem_type()) {
        continue;
      }
    }

    // Validate zero_points constant initializer if present.
    const auto* zp_arg = dq_node->InputDefs().size() > 2 ? dq_node->InputDefs()[2] : nullptr;
    bool has_zp = zp_arg && zp_arg->Exists();
    if (has_zp) {
      const auto* zp_const_tp = graph.GetConstantInitializer(zp_arg->Name(), true);
      if (!zp_const_tp || zp_const_tp->data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT4) continue;
    }

    LOGS(logger, INFO) << "DQMatMulNBitsFusion: matched pattern at MatMul node '"
                       << node->Name() << "'";

    matches.push_back({node->Index(),
                       cast_node ? std::optional<NodeIndex>(cast_node->Index()) : std::nullopt,
                       transpose_node->Index(),
                       reshape_node->Index(), dq_node->Index()});
  }

  // Collect already-matched MatMul nodes to avoid double-matching
  std::unordered_set<NodeIndex> matched_matmul_indices;
  for (const auto& m : matches) {
    matched_matmul_indices.insert(m.matmul_idx);
  }

  // Second pattern: direct DQ(axis=0, 2D UINT4) -> MatMul/Gemm
  // This covers original QDQ-quantized models (e.g. SDXL _qdq_q4f16.onnx)
  // where DequantizeLinear feeds directly into MatMul with no Reshape/Transpose.
  struct DirectDQMatch {
    NodeIndex matmul_idx;
    NodeIndex dq_idx;
  };
  std::vector<DirectDQMatch> direct_matches;

  for (auto node_index : node_topology_list) {
    auto* node = graph.GetNode(node_index);
    if (!node) continue;

    if (node->OpType() != "MatMul" && node->OpType() != "Gemm") continue;
    if (matched_matmul_indices.count(node->Index())) continue;

    const auto& mm_inputs = node->InputDefs();
    if (mm_inputs.size() < 2 || !mm_inputs[1] || !mm_inputs[1]->Exists()) continue;

    const Node* dq_node = graph.GetProducerNode(mm_inputs[1]->Name());
    if (!dq_node || dq_node->OpType() != "DequantizeLinear") continue;
    if (dq_node->GetOutputEdgesCount() != 1) continue;

    const auto& dq_attrs = dq_node->GetAttributes();
    {
      auto it = dq_attrs.find("axis");
      if (it == dq_attrs.end() || it->second.i() != 0) continue;
    }
    int64_t block_size = 0;
    {
      auto it = dq_attrs.find("block_size");
      if (it == dq_attrs.end()) continue;
      block_size = it->second.i();
      if (block_size < 16 || ((block_size - 1) & block_size)) continue;
    }

    const auto* weight_arg = dq_node->InputDefs()[0];
    if (!weight_arg || !weight_arg->Exists()) continue;
    const auto* weight_const_tp = graph.GetConstantInitializer(weight_arg->Name(), true);
    if (!weight_const_tp) continue;
    if (weight_const_tp->data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT4) continue;
    if (weight_const_tp->dims_size() != 2) continue;
    const int64_t K = weight_const_tp->dims(0);
    const int64_t N = weight_const_tp->dims(1);
    if (K <= 0 || N <= 0 || K % block_size != 0) continue;
    const int64_t k_blocks = K / block_size;

    const auto* scale_arg = dq_node->InputDefs()[1];
    if (!scale_arg || !scale_arg->Exists()) continue;
    const auto* scale_const_tp = graph.GetConstantInitializer(scale_arg->Name(), true);
    if (!scale_const_tp) continue;
    int32_t dt_scale = scale_const_tp->data_type();
    if (dt_scale != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
        dt_scale != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) continue;
    if (!HasRank2Shape(*scale_const_tp, k_blocks, N)) continue;

    const auto* a_arg = mm_inputs[0];
    if (!a_arg || !a_arg->TypeAsProto()) continue;
    int32_t dt_a = a_arg->TypeAsProto()->tensor_type().elem_type();
    if (dt_a != dt_scale) continue;

    const auto* zp_arg = dq_node->InputDefs().size() > 2 ? dq_node->InputDefs()[2] : nullptr;
    bool has_zp = zp_arg && zp_arg->Exists();
    if (has_zp) {
      const auto* zp_const_tp = graph.GetConstantInitializer(zp_arg->Name(), true);
      if (!zp_const_tp || zp_const_tp->data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT4) continue;
      if (!HasRank2Shape(*zp_const_tp, k_blocks, N)) continue;
    }

    if (node->OpType() == "Gemm") {
      if (const auto* alpha_attr = graph_utils::GetNodeAttribute(*node, "alpha");
          alpha_attr && std::abs(alpha_attr->f() - 1.0f) > 1e-6f) continue;
      if (const auto* beta_attr = graph_utils::GetNodeAttribute(*node, "beta");
          beta_attr && std::abs(beta_attr->f() - 1.0f) > 1e-6f) continue;
      if (const auto* trans_a_attr = graph_utils::GetNodeAttribute(*node, "transA");
          trans_a_attr && trans_a_attr->i() != 0) continue;
      if (const auto* trans_b_attr = graph_utils::GetNodeAttribute(*node, "transB");
          trans_b_attr && trans_b_attr->i() != 0) continue;
      if (mm_inputs.size() > 2 && mm_inputs[2] && mm_inputs[2]->Exists()) {
        const auto* bias_shape = mm_inputs[2]->Shape();
        if (!bias_shape || bias_shape->dim_size() != 1 ||
            !utils::HasDimValue(bias_shape->dim(0)) ||
            bias_shape->dim(0).dim_value() != N) continue;
      }
    }

    LOGS(logger, INFO) << "DQMatMulNBitsFusion: matched direct DQ->MatMul pattern at node '"
                       << node->Name() << "' (K=" << K << ", N=" << N << ", block_size=" << block_size << ")";
    direct_matches.push_back({node->Index(), dq_node->Index()});
  }

  // Apply fusions
  for (const auto& match : matches) {
    const Node* mm_node = graph.GetNode(match.matmul_idx);
    const Node* cast_node = match.cast_idx ? graph.GetNode(*match.cast_idx) : nullptr;
    const Node* tp_node = graph.GetNode(match.transpose_idx);
    const Node* dq_node = graph.GetNode(match.dq_idx);
    const Node* reshape_node = graph.GetNode(match.reshape_idx);
    if (!mm_node || !tp_node || !dq_node || !reshape_node ||
        (match.cast_idx && !cast_node)) {
      continue;
    }

    const auto* weight_arg = dq_node->InputDefs()[0];
    const auto* scale_arg = dq_node->InputDefs()[1];
    const auto* zp_arg = dq_node->InputDefs().size() > 2 ? dq_node->InputDefs()[2] : nullptr;
    bool has_zp = zp_arg && zp_arg->Exists();

    const auto& dq_attrs = dq_node->GetAttributes();
    const int64_t block_size = dq_attrs.at("block_size").i();

    // Load source tensors.
    const ONNX_NAMESPACE::TensorProto* weight_tp = nullptr;
    if (!graph.GetInitializedTensor(weight_arg->Name(), weight_tp) || !weight_tp) continue;
    const ONNX_NAMESPACE::TensorProto* scale_tp = nullptr;
    if (!graph.GetInitializedTensor(scale_arg->Name(), scale_tp) || !scale_tp) continue;
    const ONNX_NAMESPACE::TensorProto* zp_tp = nullptr;
    if (has_zp) {
      if (!graph.GetInitializedTensor(zp_arg->Name(), zp_tp) || !zp_tp) continue;
    }

    if (weight_tp->data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT4 ||
        weight_tp->dims_size() != 3) {
      continue;
    }

    const int64_t N = weight_tp->dims(0);
    const int64_t quant_num = weight_tp->dims(1);
    const int64_t bs_dim = weight_tp->dims(2);
    if (N <= 0 || quant_num <= 0 || bs_dim <= 0 || bs_dim != block_size) continue;
    const int64_t K = quant_num * bs_dim;
    const int64_t blob_bytes = (block_size + 1) / 2;

    Initializer weight_src(graph, *weight_tp, graph.ModelPath());
    Initializer scale_src(graph, *scale_tp, graph.ModelPath());
    if (scale_src.data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
        scale_src.data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      continue;
    }

    auto uint8_type = DataTypeImpl::TensorTypeFromONNXEnum(
                          ONNX_NAMESPACE::TensorProto_DataType_UINT8)->GetElementType();
    auto scale_type = DataTypeImpl::TensorTypeFromONNXEnum(
                          scale_src.data_type())->GetElementType();

    auto cpu_allocator = CPUAllocator::DefaultInstance();

    // Allocate destination tensors
    auto weight_dst_name = graph.GenerateNodeArgName(weight_arg->Name() + "_mnb");
    auto weight_dst = Tensor(uint8_type, TensorShape{N, quant_num, blob_bytes}, cpu_allocator);

    auto scale_dst_name = graph.GenerateNodeArgName(scale_arg->Name() + "_mnb");
    const int64_t scale_size = (TensorShape{N, quant_num}).Size();
    if (scale_src.size() != static_cast<size_t>(scale_size)) continue;
    auto scale_dst = Tensor(scale_type, TensorShape{scale_size}, cpu_allocator);

    std::string zp_dst_name;
    std::optional<Tensor> zp_dst;
    const int64_t zp_size = (TensorShape{N, (quant_num + 1) / 2}).Size();

    bool elide_default_uint4_zp8_input = false;
    std::optional<Initializer> zp_src;

    const auto weight_bytes = weight_src.DataAsByteSpan();
    if (weight_bytes.size() != static_cast<size_t>(weight_dst.SizeInBytes())) continue;
    memcpy(weight_dst.MutableDataRaw(), weight_bytes.data(), weight_bytes.size());

    if (scale_src.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      memcpy(scale_dst.MutableData<float>(), scale_src.data<float>(),
             static_cast<size_t>(scale_size) * sizeof(float));
    } else {
      memcpy(scale_dst.MutableData<MLFloat16>(), scale_src.data<MLFloat16>(),
             static_cast<size_t>(scale_size) * sizeof(MLFloat16));
    }

    if (zp_tp) {
      zp_src.emplace(graph, *zp_tp, graph.ModelPath());
      if (zp_src->data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT4) continue;
      if (zp_src->size() != static_cast<size_t>(N * quant_num)) continue;

      // WebNN lowering always materializes UINT4 zero_points.
      // If they are uniformly default 8, emit 3-input MatMulNBits (without
      // zero_points).
      const bool is_default_uint4_8 =
          IsUniformPackedUint4Value(*zp_src, /*expected_nibble*/ 8);
      if (is_default_uint4_8) {
        elide_default_uint4_zp8_input = true;
      } else {
        zp_dst_name = graph.GenerateNodeArgName(zp_arg->Name() + "_mnb");
        zp_dst = Tensor(uint8_type, TensorShape{zp_size}, cpu_allocator);
        PackUint4Rows(*zp_src, N, quant_num, zp_dst->MutableData<uint8_t>());
      }
    } else {
      // DequantizeLinear default zero-point for uint4 is 0, while MatMulNBits
      // default is 8. Emit explicit zeros to preserve semantics.
      zp_dst_name = graph.GenerateNodeArgName("webnn_fused_DQ_zp_mnb");
      zp_dst = Tensor(uint8_type, TensorShape{zp_size}, cpu_allocator);
      memset(zp_dst->MutableDataRaw(), 0, zp_dst->SizeInBytes());
    }

    // Create tensor protos
    auto weight_mnb_tp = utils::TensorToTensorProto(weight_dst, weight_dst_name, true);
    auto scale_mnb_tp = utils::TensorToTensorProto(scale_dst, scale_dst_name, true);
    std::optional<ONNX_NAMESPACE::TensorProto> zp_mnb_tp;
    if (zp_dst && !elide_default_uint4_zp8_input) {
      zp_mnb_tp.emplace(utils::TensorToTensorProto(*zp_dst, zp_dst_name, true));
    }

    // Build MatMulNBits attributes
    NodeAttributes mnb_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("K", K), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("N", N), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("accuracy_level", accuracy_level_), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("bits", static_cast<int64_t>(4)), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), mnb_attrs);

    // Build inputs: input[0] = activation from MatMul input[0]
    std::vector<NodeArg*> mnb_inputs;
    mnb_inputs.push_back(const_cast<NodeArg*>(mm_node->InputDefs()[0]));
    mnb_inputs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, weight_mnb_tp, std::move(weight_dst)));
    mnb_inputs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, scale_mnb_tp, std::move(scale_dst)));
    if (zp_mnb_tp) {
      mnb_inputs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, zp_mnb_tp.value(), std::move(*zp_dst)));
    }

    // Preserve Gemm bias (if any) by wiring it to MatMulNBits input[5].
    // MatMulNBits input layout:
    //   0:A, 1:B, 2:scales, 3:zero_points(opt), 4:g_idx(opt), 5:bias(opt)
    bool fused_with_bias = false;
    if (mm_node->OpType() == "Gemm" &&
        mm_node->InputDefs().size() > 2 &&
        mm_node->InputDefs()[2] &&
        mm_node->InputDefs()[2]->Exists()) {
      NodeArg& empty_arg = graph.GetOrCreateNodeArg("", nullptr);
      while (mnb_inputs.size() < 5) {
        mnb_inputs.push_back(&empty_arg);
      }
      mnb_inputs.push_back(const_cast<NodeArg*>(mm_node->InputDefs()[2]));
      fused_with_bias = true;
    }

    // Build outputs: same as MatMul output
    std::vector<NodeArg*> mnb_outputs;
    mnb_outputs.push_back(const_cast<NodeArg*>(mm_node->OutputDefs()[0]));

    // Add MatMulNBits node
    auto& mnb_node = graph.AddNode(
        graph.GenerateNodeName("WebNNFusedMatMulNBits"),
        "MatMulNBits",
        "Fused from WebNN DQ+Reshape+Transpose+MatMul",
        mnb_inputs, mnb_outputs, &mnb_attrs, kMSDomain);
    mnb_node.SetExecutionProviderType(mm_node->GetExecutionProviderType());

    // Remove old nodes in reverse dependency order
    // (MatMul/Gemm -> [Cast] -> Transpose -> Reshape -> DQ)
    graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.matmul_idx));
    graph.RemoveNode(match.matmul_idx);

    if (match.cast_idx && graph.GetNode(*match.cast_idx)) {
      graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(*match.cast_idx));
      graph.RemoveNode(*match.cast_idx);
    }

    graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.transpose_idx));
    graph.RemoveNode(match.transpose_idx);

    graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.reshape_idx));
    graph.RemoveNode(match.reshape_idx);

    graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.dq_idx));
    graph.RemoveNode(match.dq_idx);

    LOGS(logger, INFO) << "DQMatMulNBitsFusion: fused DQ+Reshape+Transpose"
                       << (match.cast_idx ? "+Cast" : "")
                       << "+MatMul/Gemm -> MatMulNBits"
                       << (fused_with_bias ? " (bias preserved)" : "")
                       << (elide_default_uint4_zp8_input ? " (default UINT4 zp8 elided)" : "");
    modified = true;
  }

  // Apply direct DQ -> MatMul/Gemm fusions
  for (const auto& match : direct_matches) {
    const Node* mm_node = graph.GetNode(match.matmul_idx);
    const Node* dq_node = graph.GetNode(match.dq_idx);
    if (!mm_node || !dq_node) continue;

    const auto* weight_arg = dq_node->InputDefs()[0];
    const auto* scale_arg = dq_node->InputDefs()[1];
    const auto* zp_arg = dq_node->InputDefs().size() > 2 ? dq_node->InputDefs()[2] : nullptr;
    bool has_zp = zp_arg && zp_arg->Exists();

    const auto& dq_attrs = dq_node->GetAttributes();
    const int64_t block_size = dq_attrs.at("block_size").i();

    const ONNX_NAMESPACE::TensorProto* weight_tp = nullptr;
    if (!graph.GetInitializedTensor(weight_arg->Name(), weight_tp) || !weight_tp) continue;
    const ONNX_NAMESPACE::TensorProto* scale_tp = nullptr;
    if (!graph.GetInitializedTensor(scale_arg->Name(), scale_tp) || !scale_tp) continue;
    const ONNX_NAMESPACE::TensorProto* zp_tp = nullptr;
    if (has_zp) {
      if (!graph.GetInitializedTensor(zp_arg->Name(), zp_tp) || !zp_tp) continue;
    }

    if (weight_tp->data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT4 ||
        weight_tp->dims_size() != 2) continue;

    const int64_t K = weight_tp->dims(0);
    const int64_t N = weight_tp->dims(1);
    if (K <= 0 || N <= 0 || block_size <= 0 || K % block_size != 0) continue;
    const int64_t k_blocks = K / block_size;
    const int64_t blob_bytes = block_size / 2;
    if (!HasRank2Shape(*scale_tp, k_blocks, N)) continue;
    if (zp_tp && !HasRank2Shape(*zp_tp, k_blocks, N)) continue;

    Initializer weight_src(graph, *weight_tp, graph.ModelPath());
    const size_t required_weight_bytes = static_cast<size_t>(N * k_blocks * blob_bytes);
    if (weight_src.DataAsByteSpan().size() < required_weight_bytes) continue;
    Initializer scale_src(graph, *scale_tp, graph.ModelPath());
    if (scale_src.data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
        scale_src.data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) continue;

    auto uint8_type = DataTypeImpl::TensorTypeFromONNXEnum(
                          ONNX_NAMESPACE::TensorProto_DataType_UINT8)->GetElementType();
    auto scale_type = DataTypeImpl::TensorTypeFromONNXEnum(
                          scale_src.data_type())->GetElementType();
    auto cpu_allocator = CPUAllocator::DefaultInstance();

    auto weight_dst_name = graph.GenerateNodeArgName(weight_arg->Name() + "_mnb");
    auto weight_dst = Tensor(uint8_type, TensorShape{N, k_blocks, blob_bytes}, cpu_allocator);
    TransposePackWeightsAxis0(weight_src.DataAsByteSpan().data(), K, N, block_size,
                              weight_dst.MutableData<uint8_t>());

    // Transpose scales from [k_blocks, N] to [N * k_blocks] (logical [N, k_blocks])
    auto scale_dst_name = graph.GenerateNodeArgName(scale_arg->Name() + "_mnb");
    const int64_t scale_count = N * k_blocks;
    if (scale_src.size() != static_cast<size_t>(scale_count)) continue;
    auto scale_dst = Tensor(scale_type, TensorShape{scale_count}, cpu_allocator);

    if (scale_src.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      const float* src = scale_src.data<float>();
      float* dst = scale_dst.MutableData<float>();
      for (int64_t n = 0; n < N; ++n)
        for (int64_t kb = 0; kb < k_blocks; ++kb)
          dst[n * k_blocks + kb] = src[kb * N + n];
    } else {
      const MLFloat16* src = scale_src.data<MLFloat16>();
      MLFloat16* dst = scale_dst.MutableData<MLFloat16>();
      for (int64_t n = 0; n < N; ++n)
        for (int64_t kb = 0; kb < k_blocks; ++kb)
          dst[n * k_blocks + kb] = src[kb * N + n];
    }

    std::string zp_dst_name;
    std::optional<Tensor> zp_dst;
    const int64_t zp_bytes_total = N * ((k_blocks + 1) / 2);

    bool elide_zp = false;

    if (zp_tp) {
      Initializer zp_src(graph, *zp_tp, graph.ModelPath());
      if (zp_src.data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT4) continue;
      if (zp_src.size() != static_cast<size_t>(k_blocks * N)) continue;

      if (IsUniformPackedUint4Value(zp_src, 8)) {
        elide_zp = true;
      } else {
        zp_dst_name = graph.GenerateNodeArgName(zp_arg->Name() + "_mnb");
        zp_dst = Tensor(uint8_type, TensorShape{zp_bytes_total}, cpu_allocator);
        TransposePackZPAxis0(zp_src.DataAsByteSpan().data(), k_blocks, N,
                             zp_dst->MutableData<uint8_t>());
      }
    } else {
      // DQ default ZP for UINT4 is 0, MatMulNBits default is 8. Emit explicit zeros.
      zp_dst_name = graph.GenerateNodeArgName("direct_DQ_zp_mnb");
      zp_dst = Tensor(uint8_type, TensorShape{zp_bytes_total}, cpu_allocator);
      memset(zp_dst->MutableDataRaw(), 0, zp_dst->SizeInBytes());
    }

    auto weight_mnb_tp = utils::TensorToTensorProto(weight_dst, weight_dst_name, true);
    auto scale_mnb_tp = utils::TensorToTensorProto(scale_dst, scale_dst_name, true);
    std::optional<ONNX_NAMESPACE::TensorProto> zp_mnb_tp;
    if (zp_dst && !elide_zp) {
      zp_mnb_tp.emplace(utils::TensorToTensorProto(*zp_dst, zp_dst_name, true));
    }

    NodeAttributes mnb_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("K", K), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("N", N), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("accuracy_level", accuracy_level_), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("bits", static_cast<int64_t>(4)), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), mnb_attrs);

    std::vector<NodeArg*> mnb_inputs;
    mnb_inputs.push_back(const_cast<NodeArg*>(mm_node->InputDefs()[0]));
    mnb_inputs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, weight_mnb_tp, std::move(weight_dst)));
    mnb_inputs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, scale_mnb_tp, std::move(scale_dst)));
    if (zp_mnb_tp) {
      mnb_inputs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, zp_mnb_tp.value(), std::move(*zp_dst)));
    }

    bool fused_with_bias = false;
    if (mm_node->OpType() == "Gemm" &&
        mm_node->InputDefs().size() > 2 &&
        mm_node->InputDefs()[2] &&
        mm_node->InputDefs()[2]->Exists()) {
      NodeArg& empty_arg = graph.GetOrCreateNodeArg("", nullptr);
      while (mnb_inputs.size() < 5) {
        mnb_inputs.push_back(&empty_arg);
      }
      mnb_inputs.push_back(const_cast<NodeArg*>(mm_node->InputDefs()[2]));
      fused_with_bias = true;
    }

    std::vector<NodeArg*> mnb_outputs;
    mnb_outputs.push_back(const_cast<NodeArg*>(mm_node->OutputDefs()[0]));

    auto& mnb_node = graph.AddNode(
        graph.GenerateNodeName("DirectDQFusedMatMulNBits"),
        "MatMulNBits",
        "Fused from direct DQ(axis=0)+MatMul",
        mnb_inputs, mnb_outputs, &mnb_attrs, kMSDomain);
    mnb_node.SetExecutionProviderType(mm_node->GetExecutionProviderType());

    graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.matmul_idx));
    graph.RemoveNode(match.matmul_idx);

    graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.dq_idx));
    graph.RemoveNode(match.dq_idx);

    LOGS(logger, INFO) << "DQMatMulNBitsFusion: fused direct DQ(axis=0)+MatMul/Gemm -> MatMulNBits"
                       << " (K=" << K << ", N=" << N << ", block_size=" << block_size << ")"
                       << (fused_with_bias ? " (bias preserved)" : "")
                       << (elide_zp ? " (default UINT4 zp8 elided)" : "");
    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
