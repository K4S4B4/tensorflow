#include "tensorflow/lite/delegates/gpu/common/mediapipe/landmarks_to_transform_matrix.h"

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
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

absl::Status LandmarksToTransformMatrixOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration* registration) {
  RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));
  return CheckInputsOutputs(context, tflite_node, /*runtime_inputs=*/1,
                            /*outputs=*/1);
}

absl::Status LandmarksToTransformMatrixOperationParser::Parse(
    const TfLiteNode* tflite_node, const TfLiteRegistration* registration,
    GraphFloat32* graph, ObjectReader* reader) {
  Node* node = graph->NewNode();
  RETURN_IF_ERROR(reader->AddInput(node, 0));  // landmarks
  RETURN_IF_ERROR(reader->AddOutputs(node));   // transform matrix

  node->operation.type = kLandmarksToTransformMatrixType;
  BHWC output_shape;
  if (registration->version == 2) {
    LandmarksToTransformMatrixV2Attributes attr;
    RETURN_IF_ERROR(ParseLandmarksToTransformMatrixV2Attributes(
        tflite_node->custom_initial_data, tflite_node->custom_initial_data_size,
        &attr, &output_shape));
    node->operation.attributes = attr;
  } else if (registration->version == 1) {
    LandmarksToTransformMatrixV1Attributes attr;
    RETURN_IF_ERROR(ParseLandmarksToTransformMatrixV1Attributes(
        tflite_node->custom_initial_data, tflite_node->custom_initial_data_size,
        &attr, &output_shape));
    node->operation.attributes = attr;
  } else {
    return absl::UnimplementedError(
        "Landmarks To Transform Matrix operation can be of version 1 or 2 "
        "only.");
  }

  auto output_value = graph->FindOutputs(node->id)[0];
  output_value->tensor.shape = output_shape;
  return absl::OkStatus();
}

absl::Status ParseLandmarksToTransformMatrixV1Attributes(
    const void* data, uint32_t data_size,
    LandmarksToTransformMatrixV1Attributes* attr, BHWC* output_shape) {
  const flexbuffers::Map m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(data), data_size)
          .AsMap();

  const auto input_hw = m["input_hw"].AsTypedVector();
  attr->input_hw = HW(input_hw[0].AsInt32(), input_hw[1].AsInt32());

  const auto output_hw = m["output_hw"].AsTypedVector();
  attr->output_hw = HW(output_hw[0].AsInt32(), output_hw[1].AsInt32());

  attr->dimensions = m["dimensions"].AsInt32();
  attr->landmarks_range = m["landmarks_range"].AsInt32();
  attr->bbox_size_multiplier = m["bbox_size_multiplier"].AsFloat();
  attr->left_rotation_idx = m["left_rotation_idx"].AsInt32();
  attr->right_rotation_idx = m["right_rotation_idx"].AsInt32();

  const auto subset = m["subset"].AsTypedVector();
  for (int i = 0; i < subset.size() / 2; i++) {
    attr->subset.emplace_back(subset[i * 2].AsInt32(),
                              subset[i * 2 + 1].AsInt32());
  }
  if (subset.size() % 2 != 0) {
    attr->subset.emplace_back(subset[subset.size() - 1].AsInt32(),
                              subset[subset.size() - 1].AsInt32());
  }
  *output_shape = BHWC(1, 1, 4, 4);
  return absl::OkStatus();
}

absl::Status ParseLandmarksToTransformMatrixV2Attributes(
    const void* data, uint32_t data_size,
    LandmarksToTransformMatrixV2Attributes* attr, BHWC* output_shape) {
  const flexbuffers::Map m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(data), data_size)
          .AsMap();
  const auto subset_idxs = m["subset_idxs"].AsTypedVector();
  int amount = subset_idxs.size();
  for (int i = 0; i < amount / 2; i++) {
    attr->subset_idxs.emplace_back(subset_idxs[i * 2].AsInt32(),
                                   subset_idxs[i * 2 + 1].AsInt32());
  }
  if (amount % 2 != 0) {
    int previous = amount - 1;
    attr->subset_idxs.emplace_back(subset_idxs[previous].AsInt32(),
                                   subset_idxs[previous].AsInt32());
  }
  attr->left_rotation_idx = m["left_rotation_idx"].AsInt32();
  attr->right_rotation_idx = m["right_rotation_idx"].AsInt32();
  attr->target_rotation_radians = m["target_rotation_radians"].AsFloat();
  attr->output_height = m["output_height"].AsInt32();
  attr->output_width = m["output_width"].AsInt32();
  attr->scale_x = m["scale_x"].AsFloat();
  attr->scale_y = m["scale_y"].AsFloat();

  *output_shape = BHWC(1, 1, 4, 4);
  return absl::OkStatus();
}

TransformResult LandmarksToTransformMatrixV2ToV2WithMul::ApplyToNode(
    Node* node, GraphFloat32* graph) {
  // Recognize Landmarks2TransformMatrix.v2 as a root operation of this
  // transformation.
  if (node->operation.type != kLandmarksToTransformMatrixType) {
    return {TransformStatus::SKIPPED, ""};
  }
  auto* landmarks2tm_attr =
      absl::any_cast<LandmarksToTransformMatrixV2Attributes>(
          &node->operation.attributes);
  if (!landmarks2tm_attr) {
    return {TransformStatus::SKIPPED, ""};
  }
  auto node_inputs = graph->FindInputs(node->id);
  if (node_inputs.size() != 1) {
    return {TransformStatus::SKIPPED, ""};
  }
  // Recognize preeceding scalar Mul operation and save the value.
  auto mul = graph->FindProducer(node_inputs[0]->id);
  if (mul->operation.type != ToString(OperationType::MUL)) {
    return {TransformStatus::SKIPPED, ""};
  }
  const auto& mul_attr =
      absl::any_cast<const ElementwiseAttributes&>(mul->operation.attributes);
  float scalar = 0.0;
  if (!absl::holds_alternative<float>(mul_attr.param)) {
    return {TransformStatus::SKIPPED, ""};
  } else {
    scalar = absl::get<float>(mul_attr.param);
  }
  auto mul_inputs = graph->FindInputs(mul->id);
  if (mul_inputs.size() != 1) {
    return {TransformStatus::SKIPPED, ""};
  }
  // Recognize preceding reshape.
  auto reshape = graph->FindProducer(mul_inputs[0]->id);
  if (reshape->operation.type != ToString(OperationType::RESHAPE)) {
    return {TransformStatus::SKIPPED, ""};
  }
  // Start modifying the graph.
  {
    absl::Status status = RemoveSimpleNodeKeepInput(graph, reshape);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove a node: " + std::string(status.message())};
    }
  }
  {
    absl::Status status = RemoveSimpleNodeKeepInput(graph, mul);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove a node: " + std::string(status.message())};
    }
  }
  // Update LandmarksToTransformMatrix attributes with a stored multiplier.
  landmarks2tm_attr->multiplier = scalar;
  return {TransformStatus::APPLIED, ""};
}

}  // namespace gpu
}  // namespace tflite
