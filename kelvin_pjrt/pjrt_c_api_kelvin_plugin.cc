#include "xla/pjrt/c/pjrt_c_api.h"

#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/status.h"

#include "absl/strings/str_format.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"



#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"


#include "mlir/ExecutionEngine/OptUtils.h"


#include "mlir/InitAllDialects.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "stablehlo/conversions/linalg/transforms/Passes.h"

#include "mlir/Target/LLVMIR/Export.h"
// #include "mlir/Target/LLVMIR/LLVMTranslationDialectInterface.h"
// #include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"


#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"

#include "kelvin_pjrt/copied_pjrt_buffer.h"
#include "kelvin_pjrt/kelvin_pjrt_device.h"
#include "kelvin_pjrt/kelvin_pjrt_executable.h"
#include "kelvin_pjrt/kelvin_pjrt_memory_space.h"


#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include <fstream>
#include <iostream>

namespace xla {
class PjRtExecutable;
class PjRtBuffer;
}  // namespace xla

namespace kelvin {

class KelvinPjRtClient;
class KelvinPjRtExecutable;
class KelvinPjRtLoadedExecutable;



class KelvinPjRtLoadedExecutable : public xla::PjRtLoadedExecutable {
 public:
  KelvinPjRtLoadedExecutable(
      xla::PjRtClient* client,
      std::unique_ptr<KelvinPjRtExecutable> executable)
    : client_(client),
      executable_(std::move(executable)),
      deleted_(false) {}

  ~KelvinPjRtLoadedExecutable() override = default;

  xla::PjRtClient* client() const override {
    return client_;
  }

  const xla::DeviceAssignment& device_assignment() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return device_assignment_;
  }

  xla::PjRtExecutable* GetExecutable() const override {
    return executable_.get();
  }

  absl::StatusOr<absl::flat_hash_map<std::string, xla::PjRtValueType>>
  GetCostAnalysis() const override {
    return absl::UnimplementedError("GetCostAnalysis is not implemented.");
  }


  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids() const override {
    std::cout<< "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return {};
  }

  absl::Span<xla::PjRtDevice* const> addressable_devices() const override {
    std::cout<< "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return {};
  }

  absl::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>>
  Execute(absl::Span<const std::vector<xla::PjRtBuffer*>> argument_handles,
          const xla::ExecuteOptions& options,
          std::optional<std::vector<xla::PjRtFuture<>>>& returned_futures) const override {
    return absl::UnimplementedError("Execute is not implemented.");
  }

  absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
  ExecuteSharded(absl::Span<xla::PjRtBuffer* const> argument_handles,
                 xla::PjRtDevice* device, const xla::ExecuteOptions& options,
                 std::optional<xla::PjRtFuture<>>& returned_future,
                 bool fill_future) const override {
    std::cout<< "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("ExecuteSharded is not implemented.");
  }

  absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
  ExecutePortable(absl::Span<xla::PjRtBuffer* const> argument_handles,
                  xla::PjRtDevice* device, const xla::ExecuteOptions& options,
                  std::optional<xla::PjRtFuture<>>& returned_future,
                  bool fill_future) const override {
    std::cout<< "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("ExecutePortable is not implemented.");
  }

  void Delete() override {
    std::cout<< "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
     deleted_ = true;
  }
  
  bool IsDeleted() const override { return deleted_; }

 private:
  xla::PjRtClient* const client_;
  const std::unique_ptr<KelvinPjRtExecutable> executable_;
  bool deleted_;
  const xla::DeviceAssignment device_assignment_;
};

class KelvinPjRtClient : public xla::PjRtClient {
 public:
  explicit KelvinPjRtClient()
    : memory_space_(this, 0),
      device_(this, &memory_space_),
      addressable_devices_({&device_}),
      memory_spaces_({&memory_space_}) {}

  int process_index() const override {
    return 0;
  }

  int device_count() const override {
    return 1;
  }

  int addressable_device_count() const override {
    return 1;
  }

  absl::Span<xla::PjRtDevice* const> devices() const override {
    return addressable_devices_;
  }

  absl::Span<xla::PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  absl::StatusOr<xla::PjRtDevice*> LookupDevice(
      xla::PjRtGlobalDeviceId global_device_id) const override {
    for (xla::PjRtDevice* device : devices()) {
      if (device->global_device_id() == global_device_id) {
        return device;
      }
    }
    return absl::UnimplementedError("Unimplemented LookupDevice");
  }

  absl::StatusOr<xla::PjRtDevice*> LookupAddressableDevice(
      xla::PjRtLocalDeviceId local_device_id) const override {
    for (xla::PjRtDevice* device : addressable_devices()) {
      if (device->local_device_id() == local_device_id) {
        return device;
      }
    }
    return absl::UnimplementedError("Unimplemented LookupAddressableDevice");
  }

  void UpdateGlobalProcessInfo(
      absl::Span<tensorflow::CoordinatedTaskStateInfo> infos) override {
  }

  absl::Span<xla::PjRtMemorySpace* const> memory_spaces() const override {
    return memory_spaces_;
    // return {};
  }

  xla::PjRtPlatformId platform_id() const override {
    return xla::PjRtPlatformId(0);
  }

  absl::string_view platform_name() const override {
    return "kelvin";
  }

  absl::string_view platform_version() const override {
    return "0.0.1";
  }

  std::optional<std::shared_ptr<xla::KeyValueStoreInterface>>
  key_value_store() const override {
    return std::nullopt;
  }

  std::optional<xla::PjRtPluginAttributes> plugin_attributes() const override {
    return std::nullopt;
  }

  absl::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    return xla::Unimplemented("GetDefaultDeviceAssignment is not supported.");
  }

  absl::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, std::optional<int> num_replicas_per_slice,
      int num_partitions, const xla::MultiSliceConfig* multi_slice_config) const override {
    return xla::Unimplemented("Multi slice device assignment is not supported.");
  }

  absl::StatusOr<xla::Layout> GetDefaultLayout(
      xla::PrimitiveType element_type, absl::Span<const int64_t> dims) override {
    return xla::Unimplemented("GetDefaultLayout is not supported.");
  }

  // absl::StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
  //     const override {
  //   return xla::Unimplemented("GetHloCostAnalysis is not supported.");
  // }

  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> Compile(
      const xla::XlaComputation& computation,
      xla::CompileOptions options) override {
    return absl::UnimplementedError("XlaCompile Unimplemented");
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> Compile(
      mlir::ModuleOp module, xla::CompileOptions options) override {
    module.dump();

    auto& context = *(module.getContext());
    mlir::registerAllDialects(context);
    mlir::registerLLVMDialectTranslation(context);

    {
      mlir::PassManager pm(&context);

      // Stable HLO -> Linalg
      mlir::stablehlo::StablehloLegalizeToLinalgPassOptions options;
      options.enablePrimitiveOps = true;
      pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass(options));
      pm.addPass(mlir::createCanonicalizerPass());

      // Bufferize Linalg, going from tensor to memref
      pm.addPass(mlir::createCanonicalizerPass());
      mlir::bufferization::OneShotBufferizePassOptions bufferization_options;
      bufferization_options.allowReturnAllocsFromLoops = true;
      bufferization_options.bufferizeFunctionBoundaries = true;
      bufferization_options.functionBoundaryTypeConversion =
          mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
      pm.addPass(mlir::createCSEPass());
      pm.addPass(
          mlir::bufferization::createOneShotBufferizePass(bufferization_options));
      mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
      mlir::bufferization::buildBufferDeallocationPipeline(pm,
                                                           deallocationOptions);

      // Lower linalg to loops
      pm.addPass(mlir::createConvertLinalgToLoopsPass());

      // Needed to lower memref.subview
      pm.addPass(mlir::memref::createExpandStridedMetadataPass());

      // More lowering!
      pm.addPass(mlir::createSCFToControlFlowPass());
      pm.addPass(mlir::createConvertControlFlowToLLVMPass());
      pm.addPass(mlir::createArithToLLVMConversionPass());
      pm.addPass(mlir::createConvertFuncToLLVMPass());
      pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
      pm.addPass(mlir::createReconcileUnrealizedCastsPass());

      // Clean up
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createSCCPPass());
      pm.addPass(mlir::createCSEPass());
      pm.addPass(mlir::createSymbolDCEPass());

      if (mlir::failed(pm.run(module))) {
        std::cout << "Tuturu~ " << __LINE__ << " FAILURE EMOTIONAL DAMAGE"
                  << std::endl;
        return absl::UnimplementedError(
            "Compilation failure in Compile with MLIR Module");
      }
      std::cout << "Tuturu~ " << __LINE__ << std::endl;
      module.dump();
    }

    mlir::DialectRegistry registry;
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    module->getContext()->appendDialectRegistry(registry);

    // Translate MLIR module to LLVM IR.
    llvm::LLVMContext llvm_context;
    auto llvm_module = mlir::translateModuleToLLVMIR(module, llvm_context);
    if (!llvm_module) {
      return absl::InternalError("Failed to translate MLIR to LLVM IR.");
    }

    // Set up the target machine.
    std::string error;
    // Hook to specify target architecture
    const std::string target_triple = "riscv32-unknown-elf";
    const llvm::Target* target =
        llvm::TargetRegistry::lookupTarget(target_triple, error);
    if (!target) {
      return absl::InternalError(
          absl::StrCat("Failed to lookup target: ", error));
    }

    llvm::TargetOptions opt;
    auto reloc_model = std::optional<llvm::Reloc::Model>();
    auto target_machine =
        target->createTargetMachine(target_triple, "generic-rv32", "", opt, reloc_model);

    llvm_module->setDataLayout(target_machine->createDataLayout());

    // Do LLVM optimizations here, tuned for "Os".
    auto transformer = mlir::makeOptimizingTransformer(
        /*optLevel=*/2, /*sizeLevel=*/1, target_machine);
    {
      auto error = transformer(llvm_module.get());
      if (error) {
        return absl::InternalError("Failed to optimize LLVM IR");
      }
    }

    std::string object_file_content;
    // RAII guards to o/p streams flush to object_file_content
    {
      llvm::raw_string_ostream ostream(object_file_content);
      llvm::buffer_ostream pstream(ostream);
      llvm::legacy::PassManager pass;
      if (target_machine->addPassesToEmitFile(
              pass, pstream, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        return absl::InternalError("target_machine can't emit a file of this type.");
      }

      pass.run(*llvm_module);
    }

    // std::ofstream elf_file("/usr/local/google/home/derekjchow/tuturu.elf", std::ios::binary);
    // if (elf_file) {
    //   elf_file.write(object_file_content.c_str(), object_file_content.size());
    //   elf_file.close();
    //   std::cout << "Tuturu~ ELF file written to /usr/local/google/home/derekjchow/tuturu.elf" << std::endl;
    // } else {
    //   std::cerr << "Tuturu~ Error opening file for writing." << std::endl;
    // }

    auto name = module.getName();
    if (name) {
      return std::make_unique<KelvinPjRtExecutable>(
          name.value().str(), std::move(object_file_content));
    }

    return absl::UnimplementedError("Unimplemented Compile with MLIR Module");
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileAndLoad(
      mlir::ModuleOp module, xla::CompileOptions options) override {
    auto compilation_result = Compile(module, options);
    if (!compilation_result.ok()) {
      return compilation_result.status();
    }

    xla::LoadOptions load_options;
    return Load(std::move(compilation_result.value()), load_options);
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> Load(
      std::unique_ptr<xla::PjRtExecutable> executable,
      const xla::LoadOptions& load_options) override {
    std::unique_ptr<KelvinPjRtExecutable> kelvin_executable(
        reinterpret_cast<KelvinPjRtExecutable*>(executable.release()));
    return std::make_unique<KelvinPjRtLoadedExecutable>(
        this, std::move(kelvin_executable));
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>>
  DeserializeExecutable(absl::string_view serialized,
                        std::optional<xla::CompileOptions> options) override {
    return absl::UnimplementedError("DeserializeExecutable Unimplemented");
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHostBuffer(
      const void* data, xla::PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      xla::PjRtMemorySpace* memory_space,
      const xla::Layout* device_layout) override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return std::make_unique<CopiedPjRtBuffer>(
        this, &device_, memory_space, data,
        std::move(on_done_with_host_buffer));
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CreateUninitializedBuffer(
      const xla::Shape& shape, xla::PjRtMemorySpace* memory_space) override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("CreateUninitializedBuffer Unimplemented");
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CreateErrorBuffer(
      absl::Status error, const xla::Shape& shape, xla::PjRtMemorySpace* memory) override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return xla::Unimplemented("CreateErrorBuffer not supported.");
  }

  // xla::PjRtRuntimeType runtime_type() const override {
  //   return xla::PjRtRuntimeType::kStreamExecutor;
  // }

  // absl::StatusOr<std::string> SerializeExecutable() const override {
  //   return absl::UnimplementedError("Unimplemented");
  // }

 private:
  KelvinPjRtMemorySpace memory_space_;
  KelvinPjRtDevice device_;
  std::vector<xla::PjRtDevice*> addressable_devices_;
  std::vector<xla::PjRtMemorySpace*> memory_spaces_;
};



PJRT_Error* PJRT_Kelvin_Client_Create(PJRT_Client_Create_Args* args) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
  std::unique_ptr<xla::PjRtClient> client =
      std::make_unique<KelvinPjRtClient>();
  args->client = pjrt::CreateWrapperClient(std::move(client));
  return nullptr;
}

PJRT_Error* PJRT_Kelvin_ExecuteContext_Create(
    PJRT_ExecuteContext_Create_Args* args) {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return new PJRT_Error{absl::UnimplementedError("Implement me.")};
}

PJRT_Error* PJRT_Kelvin_DeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return new PJRT_Error{
      absl::UnimplementedError("Topology not supported for Kelvin compilation.")};
}

}  // namespace kelvin

extern "C" {
const PJRT_Api* GetPjrtApi() {
  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      kelvin::PJRT_Kelvin_Client_Create,
      kelvin::PJRT_Kelvin_ExecuteContext_Create,
      kelvin::PJRT_Kelvin_DeviceTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp);

  std::cout << "Tuturu~ API Created\n";

  return &pjrt_api;
}
}