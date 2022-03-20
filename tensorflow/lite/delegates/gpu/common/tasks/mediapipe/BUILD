package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "landmarks_to_transform_matrix",
    srcs = ["landmarks_to_transform_matrix.cc"],
    hdrs = ["landmarks_to_transform_matrix.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/mediapipe:landmarks_to_transform_matrix",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
    ],
)

cc_library(
    name = "transform_landmarks",
    srcs = ["transform_landmarks.cc"],
    hdrs = ["transform_landmarks.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/mediapipe:transform_landmarks",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "transform_tensor_bilinear",
    srcs = ["transform_tensor_bilinear.cc"],
    hdrs = ["transform_tensor_bilinear.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/mediapipe:transform_tensor_bilinear",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:work_group_picking",
    ],
)
