cc_library(
    name = "kelvin_pjrt_executable",
    srcs = ["kelvin_pjrt_executable.cc"],
    hdrs = ["kelvin_pjrt_executable.h"],
    deps = [
        "@xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "kelvin_pjrt_device",
    srcs = ["kelvin_pjrt_device.cc"],
    hdrs = ["kelvin_pjrt_device.h"],
    deps = [
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/platform:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "pjrt_c_api_kelvin_plugin.so",
    srcs = ["pjrt_c_api_kelvin_plugin.cc"],
    deps = [
        ":kelvin_pjrt_device",
        ":kelvin_pjrt_executable",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/platform:status",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:BufferizationPipelines",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:Transforms",

        "@llvm-project//mlir:TensorToLinalg",

        "@llvm-project//mlir:BufferizationToMemRef",


        "@llvm-project//mlir:AllPassesAndDialects",


        "@llvm-project//mlir:ExecutionEngineUtils",



        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",



        "@llvm-project//llvm:AArch64AsmParser",
        "@llvm-project//llvm:AMDGPUAsmParser",
        "@llvm-project//llvm:ARMAsmParser",
        "@llvm-project//llvm:PowerPCAsmParser",
        "@llvm-project//llvm:RISCVAsmParser",
        "@llvm-project//llvm:SystemZAsmParser",
        "@llvm-project//llvm:X86AsmParser",

        "@llvm-project//llvm:AArch64CodeGen",
        "@llvm-project//llvm:AMDGPUCodeGen",
        "@llvm-project//llvm:ARMCodeGen",
        "@llvm-project//llvm:PowerPCCodeGen",
        "@llvm-project//llvm:RISCVCodeGen",
        "@llvm-project//llvm:SystemZCodeGen",
        "@llvm-project//llvm:X86CodeGen",


        "@stablehlo//:linalg_passes",
    ],
    linkshared = True,
)
