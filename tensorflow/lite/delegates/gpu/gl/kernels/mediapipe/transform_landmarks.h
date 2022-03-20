#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_MEDIAPIPE_TRANSFORM_LANDMARKS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_MEDIAPIPE_TRANSFORM_LANDMARKS_H_

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"

namespace tflite {
namespace gpu {
namespace gl {

std::unique_ptr<NodeShader> NewTransformLandmarksNodeShader();

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_MEDIAPIPE_TRANSFORM_LANDMARKS_H_
