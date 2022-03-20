#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/custom_registry.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/mediapipe/landmarks_to_transform_matrix.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/mediapipe/transform_landmarks.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/mediapipe/transform_tensor_bilinear.h"

namespace tflite {
namespace gpu {
namespace gl {

void RegisterCustomOps(
    absl::flat_hash_map<std::string, std::vector<std::unique_ptr<NodeShader>>>*
        shaders) {
  (*shaders)["landmarks_to_transform_matrix"].push_back(
      NewLandmarksToTransformMatrixNodeShader());
  (*shaders)["transform_landmarks"].push_back(
      NewTransformLandmarksNodeShader());
  (*shaders)["transform_tensor_bilinear"].push_back(
      NewTransformTensorBilinearNodeShader());
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
