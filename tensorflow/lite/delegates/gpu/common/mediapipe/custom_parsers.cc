#include "tensorflow/lite/delegates/gpu/common/custom_parsers.h"

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/delegates/gpu/common/mediapipe/landmarks_to_transform_matrix.h"
#include "tensorflow/lite/delegates/gpu/common/mediapipe/transform_landmarks.h"
#include "tensorflow/lite/delegates/gpu/common/mediapipe/transform_tensor_bilinear.h"
#include "tensorflow/lite/delegates/gpu/common/operation_parser.h"
#include "tensorflow/lite/delegates/gpu/common/unimplemented_operation_parser.h"

namespace tflite {
namespace gpu {

std::unique_ptr<TFLiteOperationParser> NewCustomOperationParser(
    absl::string_view op_name) {
  if (op_name == "Landmarks2TransformMatrix" ||
      op_name == "Landmarks2TransformMatrixV2") {
    return std::make_unique<LandmarksToTransformMatrixOperationParser>();
  }
  if (op_name == "TransformLandmarks") {
    return std::make_unique<TransformLandmarksOperationParser>();
  }
  if (op_name == "TransformTensor" /*for version 1*/ ||
      op_name == "TransformTensorBilinear" /*for version 2*/) {
    return std::make_unique<TransformTensorBilinearOperationParser>();
  }
  return absl::make_unique<UnimplementedOperationParser>(op_name);
}

}  // namespace gpu
}  // namespace tflite
