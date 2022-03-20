#include "tensorflow/lite/delegates/gpu/common/tasks/mediapipe/transform_tensor_bilinear.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace {

std::string AlignCornersCorrection(bool align_corners) {
  // Align corners correction: T -> S * ( T * A ), where T is a
  // transformation matrix, and subtruction and addition matrices are:
  // S            A
  // 1 0 0 -0.5   1 0 0 0.5
  // 0 1 0 -0.5   0 1 0 0.5
  // 0 0 1 0      0 0 1 0
  // 0 0 0 1      0 0 0 1
  // Transformation matrix column 3 and rows 3, 4 are identity, which makes
  // the final formula pretty simple and easy to get if doing a manual
  // multiuplication.
  return align_corners ? R"(
    first_line.w += first_line.x * 0.5 + first_line.y * 0.5 - 0.5;
    second_line.w += second_line.x * 0.5 + second_line.y * 0.5 - 0.5;
    )"
                       : "";
}

std::string GetTransformTensorBilinearKernelCode(const OperationDef& op_def,
                                                 bool align_corners) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int Z = GLOBAL_ID_2;\n";
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
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) "
       "return;\n";
  c += "  float4 first_line = args.matrix_transform.Read<float>(0, 0, 0);\n";
  c += "  float4 second_line = args.matrix_transform.Read<float>(1, 0, 0);\n";
  c += AlignCornersCorrection(align_corners);
  c += "  float4 before_transform_coord_2d = INIT_FLOAT4v4(INIT_FLOAT(X), "
       "INIT_FLOAT(Y), "
       "0.0f, 1.0f);\n";
  c += "  // Get transformed coordinates\n";
  c +=
      "  float2 xy = INIT_FLOAT2v2(dot(first_line, before_transform_coord_2d), "
      "dot(second_line, before_transform_coord_2d));\n";
  c += "  float2 xy_floor = floor(xy);\n";
  c += "  int4 st;\n";
  c += "  st.xy = INIT_INT2v2(xy_floor.x, xy_floor.y);\n";
  c += "  st.zw = INIT_INT2v2(xy_floor.x, xy_floor.y) + INIT_INT2v2(1, 1);\n";
  c += "  // Apply interpolation if coordinate is in bounds.\n";
  c += "  float4 result = INIT_FLOAT4(0.0f);\n";
  c += "  float2 t = xy - xy_floor;\n";
  c += "  if(xy.x >= 0.0 && xy.x <= INIT_FLOAT(args.src_tensor.Width() - 1) && "
       "xy.y >= 0.0 && "
       "xy.y <= INIT_FLOAT(args.src_tensor.Height() - 1)) {\n";
  c += "    float4 p0 = INIT_FLOAT4(0.0f);\n";
  c += "    float4 p1 = INIT_FLOAT4(0.0f);\n";
  c += "    float4 p2 = INIT_FLOAT4(0.0f);\n";
  c += "    float4 p3 = INIT_FLOAT4(0.0f);\n";
  const auto src_tensor_type = op_def.src_tensors[0].storage_type;
  const bool buffer_type = src_tensor_type == TensorStorageType::BUFFER ||
                           src_tensor_type == TensorStorageType::IMAGE_BUFFER;
  auto read_src = [&](const std::string& result, const std::string& xc,
                      const std::string& yc, const std::string& zc) {
    if (buffer_type) {
      c += "    if(" + xc + " >= 0 && " + yc + " >= 0 && " + xc +
           " < args.src_tensor.Width() && " + yc +
           " < args.src_tensor.Height()) {\n";
      c += "      " + result + " = args.src_tensor.Read<float>(" + xc + ", " +
           yc + ", " + zc + ");\n";
      c += "    }\n";
    } else {
      c += "    " + result + " = args.src_tensor.Read<float>(" + xc + ", " +
           yc + ", " + zc + ");\n";
    }
  };
  read_src("p0", "st.x", "st.y", "Z");
  read_src("p1", "st.z", "st.y", "Z");
  read_src("p2", "st.x", "st.w", "Z");
  read_src("p3", "st.z", "st.w", "Z");
  c += "    result = mix(mix(p0, p1, t.x), mix(p2, p3, t.x), t.y);\n";
  c += "  }\n";
  c += "  FLT4 res = TO_FLT4(result);\n";
  c += "  args.dst_tensor.Write(res, X, Y, Z);\n";
  c += "}\n";
  return c;
}
}  // namespace

absl::Status CreateTransformTensorBilinearFromNode(
    const OperationDef& op_def, const Node& node,
    std::unique_ptr<GPUOperation>* gpu_op) {
  auto attr = absl::any_cast<TransformTensorBilinearAttributes>(
      node.operation.attributes);
  if (attr.version != 1) {
    return absl::InvalidArgumentError(
        "Transform Tensor Bilinear operation supports only version 1.");
  }
  GPUOperation operation = CreateTransformTensorBilinear(op_def, attr);
  *gpu_op = absl::make_unique<GPUOperation>(std::move(operation));
  return absl::OkStatus();
}

GPUOperation CreateTransformTensorBilinear(
    const OperationDef& definition,
    const TransformTensorBilinearAttributes& attr) {
  GPUOperation op(definition);
  auto src_desc = definition.src_tensors[0];
  src_desc.SetAddressMode(AddressMode::kZero);
  op.AddSrcTensor("src_tensor", src_desc);
  op.AddSrcTensor("matrix_transform", definition.src_tensors[1]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ =
      GetTransformTensorBilinearKernelCode(definition, attr.align_corners);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
