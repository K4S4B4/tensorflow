#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEDIAPIPE_TRANSFORM_TENSOR_BILINEAR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEDIAPIPE_TRANSFORM_TENSOR_BILINEAR_H_

#include <cstdint>

#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/object_reader.h"
#include "tensorflow/lite/delegates/gpu/common/operation_parser.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

constexpr const char kTransformTensorBilinearType[] =
    "transform_tensor_bilinear";

struct TransformTensorBilinearAttributes {
  HW output_size;
  bool align_corners = false;
  int version = 0;
};

class TransformTensorBilinearOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final;
  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final;
};

absl::Status ParseTransformTensorBilinearV1Attributes(
    const void* data, uint32_t data_size,
    TransformTensorBilinearAttributes* attr, BHWC* output_shape);

absl::Status ParseTransformTensorBilinearV2Attributes(
    const void* data, uint32_t data_size,
    TransformTensorBilinearAttributes* attr, BHWC* output_shape);

// Converts Transform Tensor Bilinear operation of version 2 to version 1 with
// align corners parameter set to true.
class TransformTensorBilinearV2ToV1 : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEDIAPIPE_TRANSFORM_TENSOR_BILINEAR_H_
