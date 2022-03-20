load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "registry.cc",
    "landmarks_to_transform_matrix.h",
    "transform_landmarks.h",
    "transform_tensor_bilinear.h",
])

cc_library(
    name = "all_custom_ops",
    hdrs = [
        "landmarks_to_transform_matrix.h",
        "transform_landmarks.h",
        "transform_tensor_bilinear.h",
    ],
    deps = [
        ":landmarks_to_transform_matrix",
        ":transform_landmarks",
        ":transform_tensor_bilinear",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
    ],
)

cc_library(
    name = "landmarks_to_transform_matrix",
    srcs = ["landmarks_to_transform_matrix.cc"],
    hdrs = ["landmarks_to_transform_matrix.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/mediapipe:landmarks_to_transform_matrix",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "transform_tensor_bilinear",
    srcs = ["transform_tensor_bilinear.cc"],
    hdrs = ["transform_tensor_bilinear.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/mediapipe:transform_tensor_bilinear",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "transform_landmarks",
    srcs = ["transform_landmarks.cc"],
    hdrs = ["transform_landmarks.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/mediapipe:transform_landmarks",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

tflite_portable_test_suite()
