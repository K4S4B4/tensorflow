#include "tensorflow/lite/delegates/gpu/common/mediapipe/transform_landmarks.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/types/any.h"
#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder_helper.h"
#include "tensorflow/lite/delegates/gpu/common/object_reader.h"
#include "tensorflow/lite/delegates/gpu/common/operation_parser.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

absl::Status TransformLandmarksOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration* registration) {
  RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));
  RETURN_IF_ERROR(CheckInputsOutputs(context, tflite_node,
                                     /*runtime_inputs=*/2, /*outputs=*/1));
  return absl::OkStatus();
}

absl::Status TransformLandmarksOperationParser::Parse(
    const TfLiteNode* tflite_node, const TfLiteRegistration* registration,
    GraphFloat32* graph, ObjectReader* reader) {
  Node* node = graph->NewNode();
  RETURN_IF_ERROR(reader->AddInput(node, 0));  // data
  RETURN_IF_ERROR(reader->AddInput(node, 1));  // bbox
  RETURN_IF_ERROR(reader->AddOutputs(node));
  node->operation.type = kTransformLandmarksType;
  BHWC output_shape = graph->FindOutputs(node->id)[0]->tensor.shape;
  if (registration->version == 2) {
    TransformLandmarksAttributes attr;
    RETURN_IF_ERROR(ParseTransformLandmarksV2Attributes(
        tflite_node->custom_initial_data, tflite_node->custom_initial_data_size,
        &attr, &output_shape));
    node->operation.attributes = attr;
  } else if (registration->version == 1) {
    TransformLandmarksAttributes attr;
    RETURN_IF_ERROR(ParseTransformLandmarksV1Attributes(
        tflite_node->custom_initial_data, tflite_node->custom_initial_data_size,
        &attr, &output_shape));
    node->operation.attributes = attr;
  } else {
    return absl::UnimplementedError(
        "Transform Landmarks operation can be of version 1 or 2 only.");
  }

  auto output_value = graph->FindOutputs(node->id)[0];

  output_value->tensor.shape = graph->FindInputs(node->id)[0]->tensor.shape;
  return absl::OkStatus();
}

absl::Status ParseTransformLandmarksV1Attributes(
    const void* data, uint32_t data_size, TransformLandmarksAttributes* attr,
    BHWC* output_shape) {
  attr->version = 1;

  const flexbuffers::Map m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(data), data_size)
          .AsMap();
  const flexbuffers::TypedVector keys = m.Keys();

  for (int k = 0; k < keys.size(); ++k) {
    const std::string key = keys[k].ToString();
    const auto value = m[key];
    if (key == "dimensions") {
      attr->dimensions = value.AsInt32();
    }
    if (key == "scale") {
      attr->scale = value.AsFloat();
    }
  }
  return absl::OkStatus();
}

absl::Status ParseTransformLandmarksV2Attributes(
    const void* data, uint32_t data_size, TransformLandmarksAttributes* attr,
    BHWC* output_shape) {
  attr->version = 2;
  attr->dimensions = output_shape->c;
  attr->scale = 1.0;

  return absl::OkStatus();
}

TransformResult TransformLandmarksV2ToV1::ApplyToNode(Node* node,
                                                      GraphFloat32* graph) {
  // Recognize suitable Transform Landmarks operation.
  if (node->operation.type != kTransformLandmarksType) {
    return {TransformStatus::SKIPPED, ""};
  }
  TransformLandmarksAttributes transform_landmarks_attr =
      absl::any_cast<TransformLandmarksAttributes>(node->operation.attributes);
  if (transform_landmarks_attr.version != 2) {
    return {TransformStatus::SKIPPED,
            "Transform Landmarks operation should be of version 2."};
  }

  // Recognize suitable preceding Reshape.
  std::vector<Value*> transform_landmarks_inputs = graph->FindInputs(node->id);
  if (transform_landmarks_inputs.size() != 2) {
    return {TransformStatus::SKIPPED,
            "Transform Landmarks operation should have two inputs."};
  }
  Value* landmarks_input_tensor = transform_landmarks_inputs[1];
  if (transform_landmarks_inputs[1]->tensor.shape == BHWC(1, 1, 4, 4)) {
    landmarks_input_tensor = transform_landmarks_inputs[0];
  }
  Node* preceding_reshape = graph->FindProducer(landmarks_input_tensor->id);
  if (preceding_reshape->operation.type != ToString(OperationType::RESHAPE)) {
    return {TransformStatus::SKIPPED,
            "Expected Reshape node to be a producer of the transformation "
            "matrix input."};
  }

  // Recognize suitable succeeding Reshape.
  std::vector<Value*> transform_landmarks_outputs =
      graph->FindOutputs(node->id);
  if (transform_landmarks_outputs.size() != 1) {
    return {TransformStatus::SKIPPED,
            "Transform Landmarks operation should have one output."};
  }
  Value* landmarks_output_tensor = transform_landmarks_outputs[0];
  std::vector<Node*> landmarks__output_consumers =
      graph->FindConsumers(landmarks_output_tensor->id);
  if (landmarks__output_consumers.size() != 1) {
    return {TransformStatus::SKIPPED,
            "Transform Landmarks output should be consumed by one operation."};
  }
  Node* succeeding_reshape = landmarks__output_consumers[0];
  if (succeeding_reshape->operation.type != ToString(OperationType::RESHAPE)) {
    return {TransformStatus::SKIPPED,
            "Expected Reshape node to be a consumer of the Transform "
            "Landmarks operation's output value."};
  }

  // Delete preceding and succeding Reshape operations.
  absl::Status removed_preceding =
      RemoveSimpleNodeKeepInput(graph, preceding_reshape);
  if (!removed_preceding.ok()) {
    return {TransformStatus::INVALID,
            "Unable to remove a preceding Reshape node: " +
                std::string(removed_preceding.message())};
  }
  absl::Status removed_succeeding =
      RemoveSimpleNodeKeepOutput(graph, succeeding_reshape);
  if (!removed_succeeding.ok()) {
    return {TransformStatus::INVALID,
            "Unable to remove a succeeding Reshape node: " +
                std::string(removed_succeeding.message())};
  }

  // Switch Transform Landmarks operation back to version 1.
  transform_landmarks_attr.version = 1;
  node->operation.attributes = transform_landmarks_attr;

  return {TransformStatus::APPLIED, ""};
}

}  // namespace gpu
}  // namespace tflite
