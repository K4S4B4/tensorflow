#include "tensorflow/lite/delegates/gpu/gl/kernels/mediapipe/transform_landmarks.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/mediapipe/transform_landmarks.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class TransformLandmarks : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    if (!IsSupported(ctx)) {
      return absl::InvalidArgumentError(
          "This case is not supported by TransformLandmarks");
    }

    const auto& attr =
        absl::any_cast<const TransformLandmarksAttributes&>(ctx.op_attr);

    // For transformlandmarks v2 scale parameter is set to 1 when operation is
    // parsed.
    std::vector<Variable> params;
    if (attr.scale != 1) {
      params.push_back({"scale", static_cast<float>(attr.scale)});
    }
    std::string source = R"(
          vec4 x_transform = $input_data_1[0, 0, 0]$;
          vec4 y_transform = $input_data_1[1, 0, 0]$; )";
    if (attr.scale != 1) {
      source += R"(
          x_transform.w *= $scale$;
          y_transform.w *= $scale$;
          )";
    }
    source += R"(
          vec4 landmks = $input_data_0[gid.x, gid.y, gid.z]$;
          vec4 transformed = vec4(0.0);
    )";
    switch (attr.dimensions) {
      case 2:
        source += R"(
          // x y x y
          vec4 l_pair1_ = vec4(landmks.x, landmks.y, 0.0, 1.0);
          vec4 l_pair2_ = vec4(landmks.z, landmks.w, 0.0, 1.0);
          transformed = vec4(dot(x_transform, l_pair1_), dot(y_transform, l_pair1_),
                             dot(x_transform, l_pair2_), dot(y_transform, l_pair2_));

          value_0 = transformed;
        )";
        break;
      case 3:
        source += R"(
          if ((gid.z * 4) % 3 == 0) { // 0, 3, 6
            // x y z x
            vec4 landmks_next = $input_data_0[gid.x, gid.y, gid.z + 1]$;
            vec4 l_= landmks;
            l_.z = 0.0;
            l_.w = 1.0;
            transformed = vec4(dot(x_transform, l_),
                                  dot(y_transform, l_),
                                  landmks.z, dot(x_transform, vec4(landmks.w, landmks_next.x, 0.0, 1.0)));
          } else if ((gid.z * 4) % 3 == 1) { // 1, 4, 7
            // y z x y
            vec4 landmks_prev = $input_data_0[gid.x, gid.y, gid.z - 1]$;
            vec4 l_ = vec4(landmks.z, landmks.w, 0.0, 1.0);
            transformed = vec4(dot(y_transform, vec4(landmks_prev.w, landmks.x, 0.0, 1.0)), landmks.y,
                               dot(x_transform, l_), dot(y_transform, l_));
          } else if ((gid.z * 4) % 3 == 2) { // 2, 5, 8
            // z, x, y, z
            vec4 l_ = vec4(landmks.y, landmks.z, 0.0, 1.0);
            transformed = vec4(landmks.x, dot(x_transform, l_),
                               dot(y_transform, l_), landmks.w);
          }
          value_0 = transformed;
        )";
        break;
    }

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
  static bool IsSupported(const GenerationContext& ctx) {
    const auto& attr =
        absl::any_cast<const TransformLandmarksAttributes&>(ctx.op_attr);
    return (attr.dimensions == 2 || attr.dimensions == 3) && attr.version == 1;
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewTransformLandmarksNodeShader() {
  return absl::make_unique<TransformLandmarks>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
