#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEDIAPIPE_TRANSFORM_LANDMARKS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEDIAPIPE_TRANSFORM_LANDMARKS_H_

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

constexpr const char kTransformLandmarksType[] = "transform_landmarks";

struct TransformLandmarksAttributes {
  int dimensions = 3;
  float scale = 1.0;
  int version = 0;
};

class TransformLandmarksOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final;
  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final;
};

absl::Status ParseTransformLandmarksV1Attributes(
    const void* data, uint32_t data_size, TransformLandmarksAttributes* attr,
    BHWC* output_shape);

absl::Status ParseTransformLandmarksV2Attributes(
    const void* data, uint32_t data_size, TransformLandmarksAttributes* attr,
    BHWC* output_shape);

// Removes reshapes from subgraph:
//
//  Value_0 [1, 1, 1, 240]
//           |
//        Reshape
//           |
//  Value_1 [1, 1, 80, 3]   Value_2 [1, 1, 4, 4]
//            \                      /
//         TransformLandmarks.version_2
//                       |
//               Value_3 [1, 1, 80, 3]
//                       |
//                    Reshape
//                       |
//               Value_4 [1, 1, 1, 240]
//
// Resulting subgraph is:
//
//  Value_0 [1, 1, 1, 240]   Value_2 [1, 1, 4, 4]
//            \                      /
//         TransformLandmarks.version_1
//                       |
//               Value_4 [1, 1, 1, 240]
class TransformLandmarksV2ToV1 : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEDIAPIPE_TRANSFORM_LANDMARKS_H_
