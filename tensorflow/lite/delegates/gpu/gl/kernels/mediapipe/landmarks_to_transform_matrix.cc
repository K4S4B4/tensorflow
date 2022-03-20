#include "tensorflow/lite/delegates/gpu/gl/kernels/mediapipe/landmarks_to_transform_matrix.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/mediapipe/landmarks_to_transform_matrix.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

namespace v1 {

std::string ReadLandmark(const std::string& landmark, const std::string& idx) {
  std::string source = R"(
     vec4 )" + landmark +
                       R"(;
     {
       int z_coord = )" +
                       idx +
                       R"( * $dimensions$  / 4;
       vec4 result = $input_data_0[0, 0, z_coord]$;
       int rest = )" + idx +
                       R"( * $dimensions$  % 4;
       if (rest != 0) {
         if (rest == 1) {
          result.x = result.y;
          result.y = result.z;
         }
         if (rest == 2) {
          result.x = result.z;
          result.y = result.w;
         }
         if (rest == 3) {
         vec4 next_after_result = $input_data_0[0, 0, z_coord + 1]$;
          result.x = result.w;
          result.y = next_after_result.x;
         }
       }
       )" + landmark + R"( = result;
     }
     )";
  return source;
}

bool IsSupported(const LandmarksToTransformMatrixV1Attributes& attr) {
  return attr.dimensions == 3;
}

absl::Status GenerateCode(const LandmarksToTransformMatrixV1Attributes& attr,
                          const NodeShader::GenerationContext& ctx,
                          GeneratedCode* generated_code) {
  if (!IsSupported(attr)) {
    return absl::InvalidArgumentError(
        "This case is not supported by LandmarksToTransformMatrix v1");
  }

  std::vector<Variable> params = {
      {"dimensions", static_cast<int>(attr.dimensions)},
      {"landmarks_range", static_cast<int>(attr.landmarks_range)},
      {"left_rotation_idx", static_cast<int>(attr.left_rotation_idx)},
      {"right_rotation_idx", static_cast<int>(attr.right_rotation_idx)},
      {"bbox_size_multiplier", static_cast<float>(attr.bbox_size_multiplier)},
      {"input_h", static_cast<int>(attr.input_hw.h)},
      {"input_w", static_cast<int>(attr.input_hw.w)},
      {"output_h", static_cast<int>(attr.output_hw.h)},
      {"output_w", static_cast<int>(attr.output_hw.w)},
      {"subset", attr.subset},
      {"subset_size", static_cast<int>(attr.subset.size())},
  };

  std::string source = R"(
     )" + ReadLandmark("left_landmark", "$left_rotation_idx$") +
                       R"(

     )" + ReadLandmark("right_landmark", "$right_rotation_idx$") +
                       R"(

     float alpha = -atan(right_landmark.y - left_landmark.y,
                         right_landmark.x - left_landmark.x);

     vec4 max_value = vec4(-100000, -100000, 0.0, 0.0);
     vec4 min_value = vec4(100000, 100000, 0.0, 0.0);
     for (int i = 0; i < $subset_size$; i++) {
       for (int j = 0; j < 2; j++) {
         )" + ReadLandmark("landmark_current", "$subset$[i][j]") +
                       R"(

             vec4 rotated = vec4(landmark_current.x * cos(alpha) -
                                                landmark_current.y * sin(alpha),
                                 landmark_current.x * sin(alpha) +
                                                landmark_current.y * cos(alpha),
                                 0.0, 0.0);
             // both by x and y
             max_value = vec4(max(max_value.x, rotated.x),
                              max(max_value.y, rotated.y),
                              0.0, 0.0);
             min_value = vec4(min(min_value.x, rotated.x),
                              min(min_value.y, rotated.y),
                              0.0, 0.0);
       }
     }

    vec4 bbox_size = max_value - min_value;
    bbox_size *= $bbox_size_multiplier$;

    mat3 scale_matrix =
        mat3(bbox_size.x / float($landmarks_range$), 0.0, 0.0,  // first column
             0.0, bbox_size.y / float($landmarks_range$), 0.0,  // second column
             0.0, 0.0, 1.0);                                    // third column

    vec4 middle = (max_value + min_value) / 2.0;

    vec4 rotated_middle =
        vec4(middle.x * cos(-alpha) - middle.y * sin(-alpha),
             middle.x * sin(-alpha) + middle.y * cos(-alpha), 0.0, 0.0);

    mat3 rotation_matrix =
        mat3(cos(-alpha), sin(-alpha), 0,   // first column
             -sin(-alpha), cos(-alpha), 0,  // second column
             // third column
             (rotated_middle.x / float($landmarks_range$)) * 2.0 - 1.0,
             (rotated_middle.y / float($landmarks_range$)) * 2.0 - 1.0, 1);

    mat3 to_relative =
        mat3(2.0 / (float($output_w$) - 1.0), 0.0, 0.0,  // first column
             0.0, 2.0 / (float($output_h$) - 1.0), 0.0,  // second column
             -1.0, -1.0, 1.0);                           // third column

    mat3 to_absolute =
        mat3((float($input_w$) - 1.0) / 2.0, 0.0, 0.0,  // first column
             0.0, (float($input_h$) - 1.0) / 2.0, 0.0,  // second column
             // third column
             (float($input_w$) - 1.0) / 2.0, (float($input_h$) - 1.0)/2.0, 1.0);

    // Transformstion Matrix
    mat3 tm = to_absolute * rotation_matrix * scale_matrix * to_relative;

    // Inverse Transformation Matrix
    $output_data_0[0, 0, 0] = vec4(tm[0][0], tm[1][0],      0.0, tm[2][0])$;
    $output_data_0[1, 0, 0] = vec4(tm[0][1], tm[1][1],      0.0, tm[2][1])$;
    $output_data_0[2, 0, 0] = vec4(tm[0][2], tm[1][2], tm[2][2],      0.0)$;
    $output_data_0[3, 0, 0] = vec4(       0,        0,        0,      1.0)$;
    )";

  *generated_code = {
      /*parameters=*/params,
      /*objects=*/{},
      /*shared_variables=*/{},
      /*workload=*/uint3(1, 1, 1),
      /*workgroup=*/uint3(1, 1, 1),
      /*source_code=*/std::move(source),
      /*input=*/IOStructure::ONLY_DEFINITIONS,
      /*output=*/IOStructure::ONLY_DEFINITIONS,
  };
  return absl::OkStatus();
}

}  // namespace v1

namespace v2 {

std::string ReadLandmark(const std::string& landmark, const std::string& idx) {
  std::string source = R"(
    vec4 )" + landmark +
                       R"(;
    {
      int z_coord = )" +
                       idx +
                       R"( * $dimensions$  / 4;
      vec4 result = $input_data_0[0, 0, z_coord]$;
      int rest = )" + idx +
                       R"( * $dimensions$  % 4;
      if (rest != 0) {
        if (rest == 1) {
         result.x = result.y;
         result.y = result.z;
        }
        if (rest == 2) {
         result.x = result.z;
         result.y = result.w;
        }
        if (rest == 3) {
         vec4 next_after_result = $input_data_0[0, 0, z_coord + 1]$;
         result.x = result.w;
         result.y = next_after_result.x;
        }
      }
      result *= $multiplier$;
      )" + landmark + R"( = result;
     } )";
  return source;
}

static bool IsSupported(const NodeShader::GenerationContext& ctx) {
  return ctx.input_shapes.size() == 1 && ctx.input_shapes[0][1] == 1 &&
         ctx.input_shapes[0][2] == 1 && ctx.input_shapes[0][3] % 3 == 0;
}

absl::Status GenerateCode(const LandmarksToTransformMatrixV2Attributes& attr,
                          const NodeShader::GenerationContext& ctx,
                          GeneratedCode* generated_code) {
  if (!IsSupported(ctx)) {
    return absl::InvalidArgumentError(
        "This case is not supported by LandmarksToTransformMatrixV2");
  }

  std::vector<Variable> params = {
      {"dimensions", static_cast<int>(3)},
      {"scale_x", static_cast<float>(attr.scale_x)},
      {"scale_y", static_cast<float>(attr.scale_y)},
      {"left_rotation_idx", static_cast<int>(attr.left_rotation_idx)},
      {"right_rotation_idx", static_cast<int>(attr.right_rotation_idx)},
      {"target_rotation_radians",
       static_cast<float>(attr.target_rotation_radians)},
      {"output_width", static_cast<float>(attr.output_width)},
      {"output_height", static_cast<float>(attr.output_height)},
      {"subset_idxs", attr.subset_idxs},
      {"subset_idxs_size", static_cast<int>(attr.subset_idxs.size())},
      {"multiplier", static_cast<float>(attr.multiplier)},
  };

  std::string source = R"(
     )" + ReadLandmark("left_landmark", "$left_rotation_idx$") +
                       R"(
     )" + ReadLandmark("right_landmark", "$right_rotation_idx$") +
                       R"(

    float diff_y = right_landmark.y - left_landmark.y;
    float diff_x = right_landmark.x - left_landmark.x;
    float rotation = 0.0;
    if (diff_y != 0.0 && diff_x != 0.0) rotation = atan(diff_y, diff_x);
    float r = $target_rotation_radians$ - rotation;

    vec4 max_value = vec4(-100000, -100000, 0.0, 0.0);
    vec4 min_value = vec4(100000, 100000, 0.0, 0.0);
    for (int i = 0; i < $subset_idxs_size$; i++) {
      for (int j = 0; j < 2; j++) {
         )" + ReadLandmark("landmark_current", "$subset_idxs$[i][j]") +
                       R"(
        vec4 rotated = vec4(landmark_current.x * cos(r) -
                                                landmark_current.y * sin(r),
                                 landmark_current.x * sin(r) +
                                                landmark_current.y * cos(r),
                                 0.0, 0.0);
        // both by x and y
        max_value = vec4(max(max_value.x, rotated.x),
                         max(max_value.y, rotated.y),
                         0.0, 0.0);
        min_value = vec4(min(min_value.x, rotated.x),
                         min(min_value.y, rotated.y),
                         0.0, 0.0);
      }
    }

    float crop_width = max_value.x - min_value.x;
    float crop_height = max_value.y - min_value.y;

    vec4 crop_xy1 = (max_value + min_value) / vec4(2.0);

    float crop_x = cos(-r) * crop_xy1.x - sin(-r) * crop_xy1.y;
    float crop_y = sin(-r) * crop_xy1.x + cos(-r) * crop_xy1.y;


    mat4 t = mat4(1.0,  0.0,  0.0, 0.0,  // first  column
                  0.0,  1.0,  0.0, 0.0,  // second column
                  0.0,  0.0,  1.0, 0.0,  // third  column
                  0.0,  0.0,  0.0, 1.0); // forth  column

    mat4 t_shift = mat4(1.0,    0.0, 0.0, 0.0,  // first  column
                        0.0,    1.0, 0.0, 0.0,  // second column
                        0.0,    0.0, 1.0, 0.0,  // third  column
                     crop_x, crop_y, 0.0, 1.0); // forth  column
    t *= t_shift;

    r = -r;

    mat4 t_rotation = mat4(cos(r),  sin(r), 0.0, 0.0,  // first  column
                          -sin(r),  cos(r), 0.0, 0.0,  // second column
                              0.0,     0.0, 1.0, 0.0,  // third  column
                              0.0,     0.0, 0.0, 1.0); // forth  column

    t *= t_rotation;
    // cropped scale for x and y
    float cs_x = $scale_x$ * crop_width / $output_width$;
    float cs_y = $scale_y$ * crop_height / $output_height$;
    mat4 t_scale = mat4(cs_x,  0.0, 0.0, 0.0,  // first  column
                         0.0, cs_y, 0.0, 0.0,  // second column
                         0.0,  0.0, 1.0, 0.0,  // third  column
                         0.0,  0.0, 0.0, 1.0); // forth  column
    t *= t_scale;
    float shift_x = -1.0 * ($output_width$ / 2.0);
    float shift_y = -1.0 * ($output_height$ / 2.0);
    mat4 t_shift2 = mat4(1.0,     0.0, 0.0, 0.0,  // first  column
                         0.0,     1.0, 0.0, 0.0,  // second column
                         0.0,     0.0, 1.0, 0.0,  // third  column
                     shift_x, shift_y, 0.0, 1.0); // forth  column
    t *= t_shift2;
    // Inverse Transformation Matrix
    $output_data_0[0, 0, 0] = vec4(t[0][0], t[1][0], t[2][0], t[3][0])$;
    $output_data_0[1, 0, 0] = vec4(t[0][1], t[1][1], t[2][1], t[3][1])$;
    $output_data_0[2, 0, 0] = vec4(t[0][2], t[1][2], t[2][2], t[3][2])$;
    $output_data_0[3, 0, 0] = vec4(t[0][3], t[1][3], t[2][3], t[3][3])$;
    )";

  *generated_code = {
      /*parameters=*/params,
      /*objects=*/{},
      /*shared_variables=*/{},
      /*workload=*/uint3(1, 1, 1),
      /*workgroup=*/uint3(1, 1, 1),
      /*source_code=*/std::move(source),
      /*input=*/IOStructure::ONLY_DEFINITIONS,
      /*output=*/IOStructure::ONLY_DEFINITIONS,
  };
  return absl::OkStatus();
}

}  // namespace v2

class LandmarksToTransformMatrix : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    auto* attr_v1 =
        absl::any_cast<LandmarksToTransformMatrixV1Attributes>(&ctx.op_attr);
    if (attr_v1) return v1::GenerateCode(*attr_v1, ctx, generated_code);

    auto* attr_v2 =
        absl::any_cast<LandmarksToTransformMatrixV2Attributes>(&ctx.op_attr);
    if (attr_v2) return v2::GenerateCode(*attr_v2, ctx, generated_code);

    return absl::InvalidArgumentError("Incorrect attributes' type.");
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewLandmarksToTransformMatrixNodeShader() {
  return absl::make_unique<LandmarksToTransformMatrix>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
