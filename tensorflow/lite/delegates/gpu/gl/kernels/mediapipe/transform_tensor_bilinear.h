#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_MEDIAPIPE_TRANSFORM_TENSOR_BILINEAR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_MEDIAPIPE_TRANSFORM_TENSOR_BILINEAR_H_

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"

namespace tflite {
namespace gpu {
namespace gl {

std::unique_ptr<NodeShader> NewTransformTensorBilinearNodeShader();

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_MEDIAPIPE_TRANSFORM_TENSOR_BILINEAR_H_
