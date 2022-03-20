#include <memory>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_hints.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/subgraph.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/mediapipe/landmarks_to_transform_matrix.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/mediapipe/transform_landmarks.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/mediapipe/transform_tensor_bilinear.h"

namespace tflite {
namespace gpu {
namespace {

absl::Status CustomGPUOperationFromNode(
    const GpuInfo& gpu_info, const OperationDef& op_def, ModelHints hints,
    const std::vector<Value*>& inputs, const std::vector<Value*>& outputs,
    const Node& node, GPUOperationsSubgraph* gpu_subgraph) {
  std::unique_ptr<GPUOperation>* gpu_op =
      InitSingleOpSubgraph(inputs, outputs, gpu_subgraph);
  if (node.operation.type == kLandmarksToTransformMatrixType) {
    return CreateLandmarksToTransformMatrixFromNode(op_def, node, gpu_op);
  }
  if (node.operation.type == kTransformLandmarksType) {
    return CreateTransformLandmarksFromNode(op_def, node, gpu_op);
  }
  if (node.operation.type == kTransformTensorBilinearType) {
    return CreateTransformTensorBilinearFromNode(op_def, node, gpu_op);
  }

  return absl::UnimplementedError(
      absl::StrCat("No selector for ", node.operation.type));
}
}  // namespace

absl::Status SelectDefault(const GpuInfo& gpu_info, const OperationDef& op_def,
                           ModelHints hints, const std::vector<Value*>& inputs,
                           const std::vector<Value*>& outputs, const Node& node,
                           GPUOperationsSubgraph* gpu_subgraph) {
  return CustomGPUOperationFromNode(gpu_info, op_def, hints, inputs, outputs,
                                    node, gpu_subgraph);
}

}  // namespace gpu
}  // namespace tflite
