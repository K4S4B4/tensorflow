#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_MEDIAPIPETRANSFORM_TENSOR_BILINEAR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_MEDIAPIPETRANSFORM_TENSOR_BILINEAR_H_

#include "tensorflow/lite/delegates/gpu/common/mediapipe/transform_tensor_bilinear.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"

namespace tflite {
namespace gpu {

absl::Status CreateTransformTensorBilinearFromNode(
    const OperationDef& op_def, const Node& node,
    std::unique_ptr<GPUOperation>* gpu_op);

GPUOperation CreateTransformTensorBilinear(
    const OperationDef& definition,
    const TransformTensorBilinearAttributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_MEDIAPIPETRANSFORM_TENSOR_BILINEAR_H_
