cc_binary(
    name = "pjrt_c_api_kelvin_plugin.so",
    srcs = ["pjrt_c_api_kelvin_plugin.cc"],
    deps = [
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/platform:status",
    ],
    linkshared = True,
)
