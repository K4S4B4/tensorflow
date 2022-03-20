package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "default_selector",
    srcs = ["default_selector.cc"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_hints",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/selectors:subgraph",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/tasks/mediapipe:landmarks_to_transform_matrix",
        "//tensorflow/lite/delegates/gpu/common/tasks/mediapipe:transform_landmarks",
        "//tensorflow/lite/delegates/gpu/common/tasks/mediapipe:transform_tensor_bilinear",
        "@com_google_absl//absl/strings",
    ],
)
