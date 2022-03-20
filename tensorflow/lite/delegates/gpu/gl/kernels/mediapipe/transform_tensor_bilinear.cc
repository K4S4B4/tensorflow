#include "tensorflow/lite/delegates/gpu/gl/kernels/mediapipe/transform_tensor_bilinear.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/mediapipe/transform_tensor_bilinear.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class TransformTensorBilinear : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    if (!IsSupported(ctx)) {
      return absl::InvalidArgumentError(
          "This case is not supported by TransformTensorBilinear.");
    }

    std::vector<Variable> params = {
        {"input_data_0_h", static_cast<int>(ctx.input_shapes[0][1])},
        {"input_data_0_w", static_cast<int>(ctx.input_shapes[0][2])}};

    // Only bilinear transformation is supported right now.
    std::string source = R"(
      vec4 first_line = $input_data_1[0, 0, 0]$;
      vec4 second_line = $input_data_1[1, 0, 0]$;
      )" + AlignCornersCorrection(ctx) +
                         R"(
      vec4 before_transform_coord_2d = vec4(gid.x, gid.y, 0.0, 1.0);

      // Get transformed coordinates
      vec2 xy = vec2(dot(first_line, before_transform_coord_2d),
                     dot(second_line, before_transform_coord_2d));

      // Get coordinates of corners to interpolate from.
      int x1 = int(floor(xy.x)); // x2 is x1 + 1
      int y1 = int(floor(xy.y)); // y2 is y1 + 1

      // Apply interpolation if coordinate is in bounds.
      vec4 result = vec4(0.0);

      if(xy.x >= 0.0 && xy.x <= float($input_data_0_w$ -1) &&
         xy.y >= 0.0 && xy.y <= float($input_data_0_h$ -1)) {

        // Corners position:
        // q_11 --- q_21
        // ----     ----
        // q_12 --- q_22
)";
    source += SampleFromInput0("q_11", "x1", "y1") +
              SampleFromInput0("q_12", "x1", "y1 + 1") +
              SampleFromInput0("q_21", "x1 + 1", "y1") +
              SampleFromInput0("q_22", "x1 + 1", "y1 + 1") + R"(

        float right_contrib = xy.x - float(x1);
        float lower_contrib = xy.y - float(y1);

        vec4 upper = (1.0 - right_contrib) * q_11 + right_contrib * q_21;
        vec4 lower = (1.0 - right_contrib) * q_12 + right_contrib * q_22;

        result = lower_contrib * lower + (1.0 - lower_contrib) * upper;

      }
      value_0 = result;
    )";

    *generated_code = {
        /*parameters=*/params,
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(source),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }

 private:
  std::string SampleFromInput0(absl::string_view variable,
                               absl::string_view x_coord,
                               absl::string_view y_coord) const {
    // This function generates code, which samples data from the first input
    // tensor and checks the coordinates' bounds:
    //
    // vec4 q = vec4(0.0);
    // [0, H)
    // if (x >= 0 && x < $input_data_0_w$ && y >= 0 && y < $input_data_0_h$) {
    //   q = $input_data_0[x, y, gid.z]$;
    // }

    // Create zero initialized variable on stack
    std::string result =
        absl::Substitute("        vec4 $0 = vec4(0.0);\n", variable);
    // If coordinates are not out of scope, load value from input_data_0
    absl::SubstituteAndAppend(
        &result,
        "        if ($0 >= 0 && $1 < $$input_data_0_w$$ && "
        "$2 >= 0 && $3 < $$input_data_0_h$$) {\n",
        x_coord, x_coord, y_coord, y_coord);
    absl::SubstituteAndAppend(
        &result,
        "          $0 = $$input_data_0[$1, $2, gid.z]$$;\n        }\n\n",
        variable, x_coord, y_coord);
    return result;
  }

  std::string AlignCornersCorrection(const GenerationContext& ctx) const {
    const auto& attr =
        absl::any_cast<const TransformTensorBilinearAttributes&>(ctx.op_attr);
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
    if (attr.align_corners) {
      return R"(
      first_line.w += first_line.x * 0.5 + first_line.y * 0.5 - 0.5;
      second_line.w += second_line.x * 0.5 + second_line.y * 0.5 - 0.5;
      )";
    } else {
      return "";
    }
  }

  static bool IsSupported(const GenerationContext& ctx) {
    // if version 2 - align corners is turned on.
    // both versions expect transformation matrix as 1x1x1x16
    if (ctx.input_shapes.size() != 2) return false;

    if (ctx.input_shapes[1][0] != 1 || ctx.input_shapes[1][1] != 1 ||
        ctx.input_shapes[1][2] != 4 || ctx.input_shapes[1][3] != 4)
      return false;

    const auto& attr =
        absl::any_cast<const TransformTensorBilinearAttributes&>(ctx.op_attr);
    return attr.output_size.h > 0 && attr.output_size.w > 0 &&
           attr.version == 1;
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewTransformTensorBilinearNodeShader() {
  return absl::make_unique<TransformTensorBilinear>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
