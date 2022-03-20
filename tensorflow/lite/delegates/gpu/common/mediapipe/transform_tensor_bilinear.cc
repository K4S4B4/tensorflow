#include "tensorflow/lite/delegates/gpu/common/mediapipe/transform_tensor_bilinear.h"

#include <cstdint>
#include <string>
#include <utility>
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

absl::Status TransformTensorBilinearOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration* registration) {
  RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));
  RETURN_IF_ERROR(CheckInputsOutputs(context, tflite_node,
                                     /*runtime_inputs=*/2, /*outputs=*/1));
  return absl::OkStatus();
}

absl::Status TransformTensorBilinearOperationParser::Parse(
    const TfLiteNode* tflite_node, const TfLiteRegistration* registration,
    GraphFloat32* graph, ObjectReader* reader) {
  Node* node = graph->NewNode();
  RETURN_IF_ERROR(reader->AddInput(node, 0));  // data
  RETURN_IF_ERROR(reader->AddInput(node, 1));  // bbox
  RETURN_IF_ERROR(reader->AddOutputs(node));

  node->operation.type = kTransformTensorBilinearType;
  BHWC output_shape;
  if (registration->version == 2) {
    TransformTensorBilinearAttributes attr;
    RETURN_IF_ERROR(ParseTransformTensorBilinearV2Attributes(
        tflite_node->custom_initial_data, tflite_node->custom_initial_data_size,
        &attr, &output_shape));
    node->operation.attributes = attr;
  } else if (registration->version == 1) {
    TransformTensorBilinearAttributes attr;
    RETURN_IF_ERROR(ParseTransformTensorBilinearV1Attributes(
        tflite_node->custom_initial_data, tflite_node->custom_initial_data_size,
        &attr, &output_shape));
    node->operation.attributes = attr;
  } else {
    return absl::UnimplementedError(
        "Transform Tensor Bilinear operation can be of version 1 or 2 only.");
  }

  auto output_value = graph->FindOutputs(node->id)[0];

  output_value->tensor.shape =
      BHWC(1, output_shape.h, output_shape.w,
           graph->FindInputs(node->id)[0]->tensor.shape.c);
  return absl::OkStatus();
}

absl::Status ParseTransformTensorBilinearV1Attributes(
    const void* data, uint32_t data_size,
    TransformTensorBilinearAttributes* attr, BHWC* output_shape) {
  attr->version = 1;

  const flexbuffers::Map m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(data), data_size)
          .AsMap();
  const flexbuffers::TypedVector keys = m.Keys();

  for (int k = 0; k < keys.size(); ++k) {
    const std::string key = keys[k].ToString();
    const auto value = m[key];
    if (key == "mode") {
      if (value.AsString().str() != "bilinear") {
        return absl::UnimplementedError(
            "TransformTensor operation supports only bilinear interpolation.");
      }
    }

    if (key == "output_size") {
      attr->output_size = HW(value.AsTypedVector()[0].AsInt32(),
                             value.AsTypedVector()[1].AsInt32());
    }
  }
  attr->align_corners = false;
  *output_shape = BHWC(1, attr->output_size.h, attr->output_size.w, 1);
  return absl::OkStatus();
}

absl::Status ParseTransformTensorBilinearV2Attributes(
    const void* data, uint32_t data_size,
    TransformTensorBilinearAttributes* attr, BHWC* output_shape) {
  attr->version = 2;

  const flexbuffers::Map m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(data), data_size)
          .AsMap();
  const flexbuffers::TypedVector keys = m.Keys();
  HW output_size;
  for (int k = 0; k < keys.size(); ++k) {
    const std::string key = keys[k].ToString();
    const auto value = m[key];
    if (key == "output_height") {
      output_size.h = value.AsInt32();
    }
    if (key == "output_width") {
      output_size.w = value.AsInt32();
    }
  }
  attr->output_size = std::move(output_size);
  attr->align_corners = true;
  *output_shape = BHWC(1, attr->output_size.h, attr->output_size.w, 1);
  return absl::OkStatus();
}

TransformResult TransformTensorBilinearV2ToV1::ApplyToNode(
    Node* node, GraphFloat32* graph) {
  if (node->operation.type != kTransformTensorBilinearType) {
    return {TransformStatus::SKIPPED, ""};
  }
  TransformTensorBilinearAttributes transform_tensor_attr =
      absl::any_cast<TransformTensorBilinearAttributes>(
          node->operation.attributes);

  if (transform_tensor_attr.version != 2) {
    return {TransformStatus::SKIPPED,
            "Transform Tensor Bilinear operation should be of version 2."};
  }
  transform_tensor_attr.version = 1;
  transform_tensor_attr.align_corners = true;
  node->operation.attributes = transform_tensor_attr;

  return {TransformStatus::APPLIED, ""};
}

}  // namespace gpu
}  // namespace tflite
