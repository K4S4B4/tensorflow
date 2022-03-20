package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "custom_parsers",
    srcs = ["custom_parsers.cc"],
    hdrs = ["//tensorflow/lite/delegates/gpu/common:custom_parsers.h"],
    deps = [
        ":landmarks_to_transform_matrix",
        ":transform_landmarks",
        ":transform_tensor_bilinear",
        "//tensorflow/lite/delegates/gpu/common:operation_parser",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:unimplemented_operation_parser",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "custom_transformations",
    srcs = ["custom_transformations.cc"],
    hdrs = ["//tensorflow/lite/delegates/gpu/common:custom_transformations.h"],
    deps = [
        ":landmarks_to_transform_matrix",
        ":transform_landmarks",
        ":transform_tensor_bilinear",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "landmarks_to_transform_matrix",
    srcs = ["landmarks_to_transform_matrix.cc"],
    hdrs = ["landmarks_to_transform_matrix.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_builder_helper",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:object_reader",
        "//tensorflow/lite/delegates/gpu/common:operation_parser",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/types:any",
        "@flatbuffers",
    ],
)

cc_library(
    name = "transform_landmarks",
    srcs = ["transform_landmarks.cc"],
    hdrs = ["transform_landmarks.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_builder_helper",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:object_reader",
        "//tensorflow/lite/delegates/gpu/common:operation_parser",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/types:any",
        "@flatbuffers",
    ],
)

cc_library(
    name = "transform_tensor_bilinear",
    srcs = ["transform_tensor_bilinear.cc"],
    hdrs = ["transform_tensor_bilinear.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_builder_helper",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:object_reader",
        "//tensorflow/lite/delegates/gpu/common:operation_parser",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/types:any",
        "@flatbuffers",
    ],
)
