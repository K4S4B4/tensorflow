#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_MEDIAPIPELANDMARKS_TO_TRANSFORM_MATRIX_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_MEDIAPIPELANDMARKS_TO_TRANSFORM_MATRIX_H_

#include "tensorflow/lite/delegates/gpu/common/mediapipe/landmarks_to_transform_matrix.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"

namespace tflite {
namespace gpu {

absl::Status CreateLandmarksToTransformMatrixFromNode(
    const OperationDef& op_def, const Node& node,
    std::unique_ptr<GPUOperation>* gpu_op);

GPUOperation CreateLandmarksToTransformMatrixV1(
    const OperationDef& definition,
    const LandmarksToTransformMatrixV1Attributes& attr);

GPUOperation CreateLandmarksToTransformMatrixV2(
    const OperationDef& definition,
    const LandmarksToTransformMatrixV2Attributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_MEDIAPIPELANDMARKS_TO_TRANSFORM_MATRIX_H_
