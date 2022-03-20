#include "tensorflow/lite/delegates/gpu/common/tasks/mediapipe/transform_landmarks.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace {

std::string GetTransformLandmarksKernelCode(const OperationDef& op_def,
                                            int dimension, float scale) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.matrix_transform.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) "
       "return;\n";
  c += "  float4 x_transform = args.matrix_transform.Read<float>(0, 0, 0);\n";
  c += "  float4 y_transform = args.matrix_transform.Read<float>(1, 0, 0);\n";
  if (scale != 1.0) {
    c += "  x_transform.w *= args.scale;\n";
    c += "  y_transform.w *= args.scale;\n";
  }
  c += "  float4 landmks = args.src_tensor.Read<float>(X, Y, Z);\n";
  c += "  float4 result = INIT_FLOAT4(0.0f);\n";
  if (dimension == 2) {
    c += "  float4 l_pair1_ = INIT_FLOAT4v4(landmks.x, landmks.y, 0.0f, "
         "1.0f);\n";
    c += "  float4 l_pair2_ = INIT_FLOAT4v4(landmks.z, landmks.w, 0.0f, "
         "1.0f);\n";
    c += "  result.x = dot(x_transform, l_pair1_);\n";
    c += "  result.y = dot(y_transform, l_pair1_);\n";
    c += "  result.z = dot(x_transform, l_pair2_);\n";
    c += "  result.w = dot(y_transform, l_pair2_);\n";
  } else if (dimension == 3) {
    c += "  int reminder = (Z * 4) % 3;\n";
    c += "  if (reminder == 0) { // 0, 3, 6\n";
    c += "    // x y z x\n";
    c += "    float4 landmks_next = args.src_tensor.Read<float>(X, Y, Z+1);\n";
    c += "    float4 l_= landmks;\n";
    c += "    l_.z = 0.0f;\n";
    c += "    l_.w = 1.0f;\n";
    c += "    result.x = dot(x_transform, l_);\n";
    c += "    result.y = dot(y_transform, l_);\n";
    c += "    result.z = landmks.z;\n";
    c += "    result.w = dot(x_transform, INIT_FLOAT4v4(landmks.w, "
         "landmks_next.x, "
         "0.0f, 1.0f));\n";
    c += "  } else if (reminder == 1) { // 1, 4, 7\n";
    c += "    // y z x y\n";
    c += "    float4 landmks_prev = args.src_tensor.Read<float>(X, Y, Z-1);\n";
    c += "    float4 l_ = INIT_FLOAT4v4(landmks.z, landmks.w, 0.0f, 1.0f);\n";
    c += "    result.x = dot(y_transform, INIT_FLOAT4v4(landmks_prev.w, "
         "landmks.x, "
         "0.0f, 1.0f));\n";
    c += "    result.y = landmks.y;\n";
    c += "    result.z = dot(x_transform, l_);\n";
    c += "    result.w = dot(y_transform, l_);\n";
    c += "  } else { // reminder == 2; // 2, 5, 8\n";
    c += "    // z, x, y, z\n";
    c += "    float4 l_ = INIT_FLOAT4v4(landmks.y, landmks.z, 0.0f, 1.0f);\n";
    c += "    result.x = landmks.x;\n";
    c += "    result.y = dot(x_transform, l_);\n";
    c += "    result.z = dot(y_transform, l_);\n";
    c += "    result.w = landmks.w;\n";
    c += "  }\n";
  }
  c += "  FLT4 res = TO_FLT4(result);\n";
  c += "  args.dst_tensor.Write(res, X, Y, Z);\n";
  c += "}\n";
  return c;
}
}  // namespace

absl::Status CreateTransformLandmarksFromNode(
    const OperationDef& op_def, const Node& node,
    std::unique_ptr<GPUOperation>* gpu_op) {
  auto attr =
      absl::any_cast<TransformLandmarksAttributes>(node.operation.attributes);
  if (attr.version != 1) {
    return absl::InvalidArgumentError(
        "Transform Landmarks operation supports only version 1.");
  }
  GPUOperation operation = CreateTransformLandmarks(op_def, attr);
  *gpu_op = absl::make_unique<GPUOperation>(std::move(operation));
  return absl::OkStatus();
}

GPUOperation CreateTransformLandmarks(
    const OperationDef& definition, const TransformLandmarksAttributes& attr) {
  GPUOperation op(definition);
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddSrcTensor("matrix_transform", definition.src_tensors[1]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.args_.AddFloat("scale", attr.scale);
  op.code_ =
      GetTransformLandmarksKernelCode(definition, attr.dimensions, attr.scale);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
