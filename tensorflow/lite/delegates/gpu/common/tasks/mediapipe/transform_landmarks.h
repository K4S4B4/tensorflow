#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_MEDIAPIPETRANSFORM_LANDMARKS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_MEDIAPIPETRANSFORM_LANDMARKS_H_

#include "tensorflow/lite/delegates/gpu/common/mediapipe/transform_landmarks.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"

namespace tflite {
namespace gpu {

absl::Status CreateTransformLandmarksFromNode(
    const OperationDef& op_def, const Node& node,
    std::unique_ptr<GPUOperation>* gpu_op);

GPUOperation CreateTransformLandmarks(const OperationDef& definition,
                                      const TransformLandmarksAttributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_MEDIAPIPETRANSFORM_LANDMARKS_H_
