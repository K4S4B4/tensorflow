#include "tensorflow/lite/delegates/gpu/common/tasks/mediapipe/landmarks_to_transform_matrix.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace {

std::string GetLandmarksToTransformMatrixV1KernelCode(
    const OperationDef& op_def,
    const LandmarksToTransformMatrixV1Attributes& attr) {
  const std::string batch_id = op_def.IsBatchSupported() ? "B" : "";
  std::string c;
  c += "#define MAT_MUL_3x3(R0, R1, R2, A0, A1, A2, B0, B1, B2) \\\n";
  c += "  R0.x = A0.x * B0.x + A1.x * B0.y + A2.x * B0.z; \\\n";
  c += "  R0.y = A0.y * B0.x + A1.y * B0.y + A2.y * B0.z; \\\n";
  c += "  R0.z = A0.z * B0.x + A1.z * B0.y + A2.z * B0.z; \\\n";
  c += "  R1.x = A0.x * B1.x + A1.x * B1.y + A2.x * B1.z; \\\n";
  c += "  R1.y = A0.y * B1.x + A1.y * B1.y + A2.y * B1.z; \\\n";
  c += "  R1.z = A0.z * B1.x + A1.z * B1.y + A2.z * B1.z; \\\n";
  c += "  R2.x = A0.x * B2.x + A1.x * B2.y + A2.x * B2.z; \\\n";
  c += "  R2.y = A0.y * B2.x + A1.y * B2.y + A2.y * B2.z; \\\n";
  c += "  R2.z = A0.z * B2.x + A1.z * B2.y + A2.z * B2.z; \n";

  c += "MAIN_FUNCTION($0) {\n";
  // temporary
  c += "  int dummy_var = GLOBAL_ID_0;\n";
  if (op_def.IsBatchSupported()) {
    c += "  int B = GLOBAL_ID_0;\n";
    c += "  if (B >= args.dst_tensor.Batch()) return;\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  }
  // reads x and y coords only.
  auto read_landmark = [&](const std::string& result, const std::string& id) {
    c += "  {\n";
    c += "    int start = " + id + " * " + std::to_string(attr.dimensions) +
         ";\n";
    c += "    int ZC = start / 4;\n";
    if (attr.dimensions == 2) {
      c += "    float4 t_res = args.src_tensor.Read<float>(0, 0, ZC);\n";
      c += "    " + result + ".xy = t_res.xy;\n";
    } else if (attr.dimensions == 3) {
      c += "    float4 t_res = args.src_tensor.Read<float>(0, 0, ZC);\n";
      c += "    int rem = start % 4;\n";
      c += "    if (rem == 0) {\n";
      c += "      " + result + ".xy = t_res.xy;\n";
      c += "    } else if (rem == 1) {\n";
      c += "      " + result + ".xy = t_res.yz;\n";
      c += "    } else if (rem == 2) {\n";
      c += "      " + result + ".xy = t_res.zw;\n";
      c += "    } else {\n";
      c += "      float4 t_res_next = args.src_tensor.Read<float>(0, 0, ZC + "
           "1);\n";
      c += "      " + result + ".xy = INIT_FLOAT2v2(t_res.w, t_res_next.x);\n";
      c += "    }\n";
    }
    c += "  }\n";
  };
  c += "  float2 l_pt, r_pt;\n";
  read_landmark("l_pt", "args.rotations_idx_x");
  read_landmark("r_pt", "args.rotations_idx_y");
  c += "  float alpha = -atan2(r_pt.y - l_pt.y, r_pt.x - l_pt.x);\n";
  c += "  float cosa = cos(alpha);\n";
  c += "  float sina = sin(alpha);\n";
  c += "  float2 max_value = INIT_FLOAT2v2(-100000.0f, -100000.0f);\n";
  c += "  float2 min_value = INIT_FLOAT2v2(100000.0f, 100000.0f);\n";
  c += "  for (int i = 0; i < args.subset_size; i++) {\n";
  c += "    float2 p0, p1;\n";
  c += "    int2 subset_v = args.subset.Read(i);\n";
  read_landmark("p0", "subset_v.x");
  read_landmark("p1", "subset_v.y");
  c += "    // rotation\n";
  c +=
      "    p0 = INIT_FLOAT2v2(p0.x*cosa - p0.y*sina, p0.x*sina + p0.y*cosa);\n";
  c +=
      "    p1 = INIT_FLOAT2v2(p1.x*cosa - p1.y*sina, p1.x*sina + p1.y*cosa);\n";
  c += "    max_value.x = max(max(p0.x, p1.x), max_value.x);\n";
  c += "    max_value.y = max(max(p0.y, p1.y), max_value.y);\n";
  c += "    min_value.x = min(min(p0.x, p1.x), min_value.x);\n";
  c += "    min_value.y = min(min(p0.y, p1.y), min_value.y);\n";
  c += "  }\n";
  c += "  float2 bbox_size = (max_value - min_value) * "
       "args.bbox_size_multiplier;\n";
  c +=
      "  float3 scale_mat_c0 = INIT_FLOAT3v3(bbox_size.x / args.l_range, 0.0f, "
      "0.0f);\n";
  c +=
      "  float3 scale_mat_c1 = INIT_FLOAT3v3(0.0f, bbox_size.y / args.l_range, "
      "0.0f);\n";
  c += "  float3 scale_mat_c2 = INIT_FLOAT3v3(0.0f, 0.0f, 1.0f);\n";
  c += "  float2 middle = (max_value + min_value) * 0.5f;\n";
  c += "  float2 rotated_middle;\n";
  c += "  float cosnega = cos(-alpha);\n";
  c += "  float sinnega = sin(-alpha);\n";
  c += "  rotated_middle.x = middle.x * cosnega - middle.y * sinnega;\n";
  c += "  rotated_middle.y = middle.x * sinnega + middle.y * cosnega;\n";
  c += "  float3 rot_mat_c0 = INIT_FLOAT3v3(cosnega, sinnega, 0.0f);\n";
  c += "  float3 rot_mat_c1 = INIT_FLOAT3v3(-sinnega, cosnega, 0.0f);\n";
  c += "  float3 rot_mat_c2 = INIT_FLOAT3v3(rotated_middle.x / args.l_range * "
       "2.0f - "
       "1.0f, rotated_middle.y / args.l_range * 2.0f - 1.0f, 1.0f);\n";
  c += "  float3 to_relative_c0 = INIT_FLOAT3v3(2.0f / (args.output_size_x - "
       "1.0f), 0.0f, 0.0f);\n";
  c += "  float3 to_relative_c1 = INIT_FLOAT3v3(0.0f, 2.0f / "
       "(args.output_size_y - 1.0f), 0.0f);\n";
  c += "  float3 to_relative_c2 = INIT_FLOAT3v3(-1.0f, -1.0f, 1.0f);\n";
  c += "  float3 to_absolute_c0 = INIT_FLOAT3v3((args.input_size_x - 1.0f) / "
       "2.0f, 0.0f, 0.0f);\n";
  c += "  float3 to_absolute_c1 = INIT_FLOAT3v3(0.0f, (args.input_size_y - "
       "1.0f) / 2.0f, 0.0f);\n";
  c += "  float3 to_absolute_c2 = INIT_FLOAT3v3((args.input_size_x - 1.0f) / "
       "2.0f, (args.input_size_y - 1.0f) / 2.0f, 1.0f);\n";
  c += "  float3 t0;\n";
  c += "  float3 t1;\n";
  c += "  float3 t2;\n";
  c += "  // t0 = to_absolute * rotation_matrix\n";
  c += "  MAT_MUL_3x3(t0, t1, t2, to_absolute_c0, to_absolute_c1, "
       "to_absolute_c2, rot_mat_c0, rot_mat_c1, rot_mat_c2);\n";
  c += "  float3 u0;\n";
  c += "  float3 u1;\n";
  c += "  float3 u2;\n";
  c += "  // u0 = t0 * scale_matrix\n";
  c += "  MAT_MUL_3x3(u0, u1, u2, t0, t1, t2, scale_mat_c0, scale_mat_c1, "
       "scale_mat_c2);\n";
  c += "  float3 res_c0;\n";
  c += "  float3 res_c1;\n";
  c += "  float3 res_c2;\n";
  c += "  MAT_MUL_3x3(res_c0, res_c1, res_c2, u0, u1, u2, to_relative_c0, "
       "to_relative_c1, to_relative_c2);\n";
  c += "  FLT4 r0 = INIT_FLT4v4(res_c0.x, res_c1.x,     0.0f, res_c2.x);\n";
  c += "  FLT4 r1 = INIT_FLT4v4(res_c0.y, res_c1.y,     0.0f, res_c2.y);\n";
  c += "  FLT4 r2 = INIT_FLT4v4(res_c0.z, res_c1.z, res_c2.z,     0.0f);\n";
  c += "  FLT4 r3 = INIT_FLT4v4(    0.0f,     0.0f,     0.0f,     1.0f);\n";
  c += "  args.dst_tensor.Write(r0, 0, 0, 0);\n";
  c += "  args.dst_tensor.Write(r1, 1, 0, 0);\n";
  c += "  args.dst_tensor.Write(r2, 2, 0, 0);\n";
  c += "  args.dst_tensor.Write(r3, 3, 0, 0);\n";
  c += "}\n";
  return c;
}

std::string GetLandmarksToTransformMatrixV2KernelCode(
    const OperationDef& op_def,
    const LandmarksToTransformMatrixV2Attributes& attr) {
  std::string c;
  c += "#define MAT_MUL_3x3(R0, R1, R2, A0, A1, A2, B0, B1, B2) \\\n";
  c += "  R0.x = A0.x * B0.x + A1.x * B0.y + A2.x * B0.z; \\\n";
  c += "  R0.y = A0.y * B0.x + A1.y * B0.y + A2.y * B0.z; \\\n";
  c += "  R0.z = A0.z * B0.x + A1.z * B0.y + A2.z * B0.z; \\\n";
  c += "  R1.x = A0.x * B1.x + A1.x * B1.y + A2.x * B1.z; \\\n";
  c += "  R1.y = A0.y * B1.x + A1.y * B1.y + A2.y * B1.z; \\\n";
  c += "  R1.z = A0.z * B1.x + A1.z * B1.y + A2.z * B1.z; \\\n";
  c += "  R2.x = A0.x * B2.x + A1.x * B2.y + A2.x * B2.z; \\\n";
  c += "  R2.y = A0.y * B2.x + A1.y * B2.y + A2.y * B2.z; \\\n";
  c += "  R2.z = A0.z * B2.x + A1.z * B2.y + A2.z * B2.z; \n";

  c += "MAIN_FUNCTION($0) {\n";
  // temporary
  c += "  int dummy_var = GLOBAL_ID_0;\n";
  if (op_def.IsBatchSupported()) {
    c += "  int B = GLOBAL_ID_0;\n";
    c += "  if (B >= args.dst_tensor.Batch()) return;\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  }
  // reads x and y coords only.
  auto read_landmark = [&](const std::string& result, const std::string& id) {
    c += "  {\n";
    c += "    int start = " + id + " * 3; // only 3 dimensional landmarks\n";
    c += "    int ZC = start / 4;\n";
    c += "    float4 t_res = args.src_tensor.Read<float>(0, 0, ZC);\n";
    c += "    int rem = start % 4;\n";
    c += "    if (rem == 0) {\n";
    c += "      " + result + ".xy = t_res.xy;\n";
    c += "    } else if (rem == 1) {\n";
    c += "      " + result + ".xy = t_res.yz;\n";
    c += "    } else if (rem == 2) {\n";
    c += "      " + result + ".xy = t_res.zw;\n";
    c += "    } else {\n";
    c += "      float4 t_res_next = args.src_tensor.Read<float>(0, 0, ZC + "
         "1);\n";
    c += "      " + result + ".xy = INIT_FLOAT2v2(t_res.w, t_res_next.x);\n";
    c += "    }\n";
    c += "    " + result + " *= args.multiplier;\n";
    c += "  }\n";
  };
  c += "  float2 left_landmark, right_landmark;\n";
  read_landmark("left_landmark", "args.left_rotation_idx");
  read_landmark("right_landmark", "args.right_rotation_idx");
  c += "  float diff_y = right_landmark.y - left_landmark.y;\n";
  c += "  float diff_x = right_landmark.x - left_landmark.x;\n";
  c += "  float rotation = 0.0;\n";
  c += "  if (diff_y != 0.0 && diff_x != 0.0) {"
       "    rotation = atan2(diff_y, diff_x);\n"
       "   }";
  c += "  float r = args.target_rotation_radians - rotation;\n";
  c += "  float cosr = cos(r);\n";
  c += "  float sinr = sin(r);\n";
  c += "  float2 max_value = INIT_FLOAT2v2(-100000.0f, -100000.0f);\n";
  c += "  float2 min_value = INIT_FLOAT2v2(100000.0f, 100000.0f);\n";
  c += "  for (int i = 0; i < args.subset_idxs_size; i++) {\n";
  c += "    float2 p0, p1;\n";
  c += "    int2 subset_idxs_v = args.subset_idxs.Read(i);\n";
  read_landmark("p0", "subset_idxs_v.x");
  read_landmark("p1", "subset_idxs_v.y");
  c += "    // rotation\n";
  c +=
      "    p0 = INIT_FLOAT2v2(p0.x*cosr - p0.y*sinr, p0.x*sinr + p0.y*cosr);\n";
  c +=
      "    p1 = INIT_FLOAT2v2(p1.x*cosr - p1.y*sinr, p1.x*sinr + p1.y*cosr);\n";
  c += "    max_value.x = max(max(p0.x, p1.x), max_value.x);\n";
  c += "    max_value.y = max(max(p0.y, p1.y), max_value.y);\n";
  c += "    min_value.x = min(min(p0.x, p1.x), min_value.x);\n";
  c += "    min_value.y = min(min(p0.y, p1.y), min_value.y);\n";
  c += "  }\n";
  c += "  float crop_width  = max_value.x - min_value.x;\n";
  c += "  float crop_height = max_value.y - min_value.y;\n";
  c += "  float2 crop_xy1 = (max_value + min_value) / 2.0f;\n";
  c += "  float crop_x = cos(-r) * crop_xy1.x - sin(-r) * crop_xy1.y;\n";
  c += "  float crop_y = sin(-r) * crop_xy1.x + cos(-r) * crop_xy1.y;\n";
  c += "  float3 shift_c0 = INIT_FLOAT3v3(1.0,    0.0,    0.0);\n";
  c += "  float3 shift_c1 = INIT_FLOAT3v3(0.0,   1.0,     0.0);\n";
  c += "  float3 shift_c2 = INIT_FLOAT3v3(crop_x, crop_y,  1.0);\n";
  c += "  r = -r;\n";
  c += "  float3 rotation_c0 = INIT_FLOAT3v3(cos(r),  sin(r),  0.0);\n";
  c += "  float3 rotation_c1 = INIT_FLOAT3v3(-sin(r), cos(r),  0.0);\n";
  c += "  float3 rotation_c2 = INIT_FLOAT3v3(0.0,    0.0, 1.0);\n";
  c += "  float3 t0;\n";
  c += "  float3 t1;\n";
  c += "  float3 t2;\n";
  c += "  MAT_MUL_3x3(t0, t1, t2, shift_c0, shift_c1, shift_c2, "
       "              rotation_c0, rotation_c1, rotation_c2);\n";
  c += "  float cs_x = args.scale_x * crop_width / args.output_width;\n";
  c += "  float cs_y = args.scale_y * crop_height / args.output_height;\n";
  c += "  float3 scale_c0 = INIT_FLOAT3v3(cs_x, 0.0, 0.0);\n";
  c += "  float3 scale_c1 = INIT_FLOAT3v3(0.0, cs_y, 0.0);\n";
  c += "  float3 scale_c2 = INIT_FLOAT3v3(0.0, 0.0, 1.0);\n";
  c += "  MAT_MUL_3x3(t0, t1, t2, t0, t1, t2, "
       "              scale_c0, scale_c1, scale_c2);\n";
  c += "  float shift_x = -1.0 * (args.output_width / 2.0);\n";
  c += "  float shift_y = -1.0 * (args.output_height / 2.0);\n";
  c += "  float3 shift2_c0 = INIT_FLOAT3v3(1.0,     0.0,    0.0);\n";
  c += "  float3 shift2_c1 = INIT_FLOAT3v3(0.0,    1.0,     0.0);\n";
  c += "  float3 shift2_c2 = INIT_FLOAT3v3(shift_x, shift_y, 1.0);\n";
  c += "  MAT_MUL_3x3(t0, t1, t2, t0, t1, t2, "
       "              shift2_c0, shift2_c1, shift2_c2);\n";
  c += "  FLT4 r0 = INIT_FLT4v4(t0.x, t1.x, 0.0f, t2.x);\n";
  c += "  FLT4 r1 = INIT_FLT4v4(t0.y, t1.y, 0.0f, t2.y);\n";
  c += "  FLT4 r2 = INIT_FLT4v4(t0.z, t1.z, t2.z, 0.0f);\n";
  c += "  FLT4 r3 = INIT_FLT4v4(0.0f, 0.0f, 0.0f, 1.0f);\n";
  c += "  args.dst_tensor.Write(r0, 0, 0, 0);\n";
  c += "  args.dst_tensor.Write(r1, 1, 0, 0);\n";
  c += "  args.dst_tensor.Write(r2, 2, 0, 0);\n";
  c += "  args.dst_tensor.Write(r3, 3, 0, 0);\n";
  c += "}\n";
  return c;
}

}  // namespace

absl::Status CreateLandmarksToTransformMatrixFromNode(
    const OperationDef& op_def, const Node& node,
    std::unique_ptr<GPUOperation>* gpu_op) {
  auto* attr_v1 = absl::any_cast<LandmarksToTransformMatrixV1Attributes>(
      &node.operation.attributes);
  if (attr_v1) {
    GPUOperation operation =
        CreateLandmarksToTransformMatrixV1(op_def, *attr_v1);
    *gpu_op = absl::make_unique<GPUOperation>(std::move(operation));
    return absl::OkStatus();
  }
  auto* attr_v2 = absl::any_cast<LandmarksToTransformMatrixV2Attributes>(
      &node.operation.attributes);
  if (attr_v2) {
    GPUOperation operation =
        CreateLandmarksToTransformMatrixV2(op_def, *attr_v2);
    *gpu_op = absl::make_unique<GPUOperation>(std::move(operation));
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      "Landmarks To Transform Matrix operation supports only version 1 or "
      "2.");
}

GPUOperation CreateLandmarksToTransformMatrixV1(
    const OperationDef& definition,
    const LandmarksToTransformMatrixV1Attributes& attr) {
  std::vector<int32_t> data(attr.subset.size() * 2);
  for (int i = 0; i < attr.subset.size(); ++i) {
    data[i * 2 + 0] = attr.subset[i].x;
    data[i * 2 + 1] = attr.subset[i].y;
  }

  BufferDescriptor desc;
  desc.element_type = DataType::INT32;
  desc.element_size = 2;
  desc.memory_type = MemoryType::GLOBAL;
  desc.size = attr.subset.size() * sizeof(int32_t) * 2;
  desc.data.resize(desc.size);
  memcpy(desc.data.data(), data.data(), desc.size);

  GPUOperation result(definition);
  result.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  result.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  result.args_.AddFloat("l_range", attr.landmarks_range);
  result.args_.AddFloat("bbox_size_multiplier", attr.bbox_size_multiplier);
  result.args_.AddInt("rotations_idx_x", attr.left_rotation_idx);
  result.args_.AddInt("rotations_idx_y", attr.right_rotation_idx);
  result.args_.AddFloat("input_size_x", attr.input_hw.w);
  result.args_.AddFloat("input_size_y", attr.input_hw.h);
  result.args_.AddFloat("output_size_x", attr.output_hw.w);
  result.args_.AddFloat("output_size_y", attr.output_hw.h);
  result.args_.AddInt("subset_size", attr.subset.size());
  result.args_.AddObject("subset",
                         absl::make_unique<BufferDescriptor>(std::move(desc)));
  result.code_ = GetLandmarksToTransformMatrixV1KernelCode(definition, attr);
  result.work_group_size_ = int3(1, 1, 1);
  result.tensor_to_grid_ = TensorToGrid::kBToX_YIs1_ZIs1;

  return result;
}

GPUOperation CreateLandmarksToTransformMatrixV2(
    const OperationDef& definition,
    const LandmarksToTransformMatrixV2Attributes& attr) {
  std::vector<int32_t> data(attr.subset_idxs.size() * 2);
  for (int i = 0; i < attr.subset_idxs.size(); ++i) {
    data[i * 2 + 0] = attr.subset_idxs[i].x;
    data[i * 2 + 1] = attr.subset_idxs[i].y;
  }

  BufferDescriptor desc;
  desc.element_type = DataType::INT32;
  desc.element_size = 2;
  desc.memory_type = MemoryType::GLOBAL;
  desc.size = attr.subset_idxs.size() * sizeof(int32_t) * 2;
  desc.data.resize(desc.size);
  memcpy(desc.data.data(), data.data(), desc.size);

  GPUOperation result(definition);
  result.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  result.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  result.args_.AddInt("left_rotation_idx", attr.left_rotation_idx);
  result.args_.AddInt("right_rotation_idx", attr.right_rotation_idx);
  result.args_.AddFloat("target_rotation_radians",
                        attr.target_rotation_radians);
  result.args_.AddFloat("output_height", attr.output_height);
  result.args_.AddFloat("output_width", attr.output_width);
  result.args_.AddFloat("scale_x", attr.scale_x);
  result.args_.AddFloat("scale_y", attr.scale_y);
  result.args_.AddFloat("multiplier", attr.multiplier);

  result.args_.AddInt("subset_idxs_size", attr.subset_idxs.size());
  result.args_.AddObject("subset_idxs",
                         absl::make_unique<BufferDescriptor>(std::move(desc)));
  result.code_ = GetLandmarksToTransformMatrixV2KernelCode(definition, attr);
  result.work_group_size_ = int3(1, 1, 1);
  result.tensor_to_grid_ = TensorToGrid::kBToX_YIs1_ZIs1;
  return result;
}

}  // namespace gpu
}  // namespace tflite
