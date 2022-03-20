#include "tensorflow/lite/delegates/gpu/common/custom_transformations.h"

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/mediapipe/landmarks_to_transform_matrix.h"
#include "tensorflow/lite/delegates/gpu/common/mediapipe/transform_landmarks.h"
#include "tensorflow/lite/delegates/gpu/common/mediapipe/transform_tensor_bilinear.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"

namespace tflite {
namespace gpu {
bool ApplyCustomTransformations(ModelTransformer* transformer) {
  return transformer->Apply(
             "transform_landmarks_v2_to_v1",
             absl::make_unique<TransformLandmarksV2ToV1>().get()) &&
         transformer->Apply(
             "transform_tensor_bilinear_v2_to_v1",
             absl::make_unique<TransformTensorBilinearV2ToV1>().get()) &&
         transformer->Apply(
             "landmarks_to_transform_matrix_v2_with_mul",
             absl::make_unique<LandmarksToTransformMatrixV2ToV2WithMul>()
                 .get());
}
}  // namespace gpu
}  // namespace tflite
