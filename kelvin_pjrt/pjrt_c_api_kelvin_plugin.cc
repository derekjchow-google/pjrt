#include "xla/pjrt/c/pjrt_c_api.h"

#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/status.h"

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
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "xla/mlir_hlo/transforms/passes.h"

#include "xla/mlir_hlo/mhlo/transforms/passes.h"

#include "stablehlo/conversions/linalg/transforms/Passes.h"

#include "mlir/Target/LLVMIR/Export.h"
// #include "mlir/Target/LLVMIR/LLVMTranslationDialectInterface.h"
// #include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"


#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"





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

class KelvinPjRtExecutable : public xla::PjRtExecutable {
 public:
  KelvinPjRtExecutable(std::string name, std::string object_code)
      : name_(std::move(name)),
        object_code_(std::move(object_code)) {}

  ~KelvinPjRtExecutable() override = default;

  int num_replicas() const override { return 1; }

  int num_partitions() const override { return 1; }

  int64_t SizeOfGeneratedCodeInBytes() const override {
    return object_code_.size();
  }

  absl::string_view name() const override { return name_; }

  absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>> GetHloModules()
      const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("GetHloModules Not implemented.");
  }

  absl::StatusOr<std::vector<xla::Shape>> GetOutputShapes() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("GetOutputShapes Not implemented.");
  }

  absl::StatusOr<std::vector<std::vector<xla::PrimitiveType>>>
  GetOutputElementTypes() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("GetOutputElementTypes Not implemented.");
  }

  absl::StatusOr<std::vector<std::vector<xla::DimensionVector>>>
  GetOutputDimensions() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("GetOutputDimensions Not implemented.");
  }

  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetParameterLayouts() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("GetParameterLayouts is not implemented.");
  }

  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetOutputLayouts() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("GetOutputLayouts is not implemented.");
  }

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("GetOutputMemoryKinds is not implemented.");
  }

  std::optional<std::vector<xla::OpSharding>> GetParameterShardings() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return std::nullopt;
  }

  std::optional<std::vector<xla::OpSharding>> GetOutputShardings() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return std::nullopt;
  }

  // Return memory stats that allow callers to estimate device memory usage
  // when running this executable.
  absl::StatusOr<xla::CompiledMemoryStats> GetCompiledMemoryStats() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError(
        "GetCompiledMemoryStats is not implemented.");
  }

  // Returns named values for cost properties of this executable (such as
  // operations, size of input/outputs, and run time estimate). Properties may
  // differ for different platforms.
  absl::StatusOr<absl::flat_hash_map<std::string, xla::PjRtValueType>>
  GetCostAnalysis() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("GetCostAnalysis is not implemented.");
  }

  // Serialize this executable into a string and return the value.
  absl::StatusOr<std::string> SerializeExecutable() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("SerializeExecutable is not implemented.");
  }

  // Return a fingerprint of this executable.
  absl::StatusOr<std::string> FingerprintExecutable() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError(
        "FingerprintExecutable is not implemented.");
  }

  absl::StatusOr<struct xla::CompileOptions> GetCompileOptions() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("GetCompileOptions is not implemented.");
  }

 private:
  const std::string name_;
  const std::string object_code_;
};

class KelvinPjRtDeviceDescription : public xla::PjRtDeviceDescription {
 public:
  KelvinPjRtDeviceDescription(int id, int process_index)
    : id_(id),
      process_index_(process_index),
      device_kind_("KelvinV2"),
      attributes_({}) {}

  ~KelvinPjRtDeviceDescription() override = default;

  int id() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return id_;
  }

  int process_index() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return process_index_;
  }

  absl::string_view device_kind() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return device_kind_;
  }

  absl::string_view DebugString() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    // TODO(derekjchow): Implement more diligently
    return "KelvinPjRtDeviceDescription debug device";
  }

  absl::string_view ToString() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    // TODO(derekjchow): Add in id.
    return "KelvinPjRtDevice(id=)";
  }

  // Returns vendor specific attributes about the device. For example the model
  // number of a GPU, or the mesh coordinates of a TPU device. The returned
  // reference will remain valid for the lifetime of the PjRtDevice.
  const absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>&
  Attributes() const override {
    return attributes_;
  }

  // Returns all memory spaces attached to this device.
  // The memory spaces are in no particular order.
  absl::Span<const xla::PjRtMemorySpaceDescription* const> memory_spaces()
      const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return {};
  }

  // Returns the default memory space attached to this device.
  absl::StatusOr<const xla::PjRtMemorySpaceDescription*>
  default_memory_space() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("default_memory_space Not implemented.");
  }

 public:
  const int id_;
  const int process_index_;
  const std::string device_kind_;
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;
};

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

class KelvinPjRtDevice : public xla::PjRtDevice {
 public:
  explicit KelvinPjRtDevice(KelvinPjRtClient* client)
    : client_(client),
      description_(42, 0) {}

  ~KelvinPjRtDevice() override = default;

  xla::PjRtClient* client() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return reinterpret_cast<xla::PjRtClient*>(client_);
  }

  bool IsAddressable() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return true;
  }

  const xla::PjRtDeviceDescription& description() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return description_;
  }

  ABSL_DEPRECATED("Use global_device_id() instead")
  int id() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return global_device_id().value();
  }

  xla::PjRtGlobalDeviceId global_device_id() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return xla::PjRtGlobalDeviceId(description().id());
  }

  xla::PjRtLocalDeviceId local_device_id() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return xla::PjRtLocalDeviceId(local_hardware_id().value());
  }

  xla::PjRtLocalHardwareId local_hardware_id() const override {
    return xla::PjRtLocalHardwareId(9001);
  }

  int process_index() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return description().process_index();
  }

  absl::string_view device_kind() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return description().device_kind();
  }

  absl::string_view DebugString() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return description().DebugString();
  }

  absl::string_view ToString() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return description().ToString();
  }

  const absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>&
  Attributes() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return description().Attributes();
  }

  std::unique_ptr<xla::ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return nullptr;
  }

  absl::Status TransferToInfeed(const xla::LiteralSlice& literal) override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("TransferToInfeed not implemented");
  }

  absl::Status TransferFromOutfeed(xla::MutableBorrowingLiteral literal) override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("TransferFromOutfeed not implemented");
  }

  absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() const override {
    return absl::UnimplementedError("GetAllocatorStats is not supported");
  }

  absl::Span<xla::PjRtMemorySpace* const> memory_spaces() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return {};
  }

  absl::StatusOr<xla::PjRtMemorySpace*> default_memory_space() const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("default_memory_space not implemented");
  }

  absl::StatusOr<xla::PjRtMemorySpace*> memory_space_by_kind(
      absl::string_view memory_space_kind) const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("memory_space_by_kind not implemented");
  }

  absl::StatusOr<std::intptr_t> GetStreamForExternalReadyEvents()
      const override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError(
        "PjRtDevice::GetStreamForExternalReadyEvents only implemented for "
        "GPU");
  }

  absl::StatusOr<bool> PoisonExecution(int32_t launch_id,
                                       absl::Status error) override {
    std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
    return absl::UnimplementedError("PoisonExecution is not supported");
  }

 private:
  KelvinPjRtClient* const client_;
  KelvinPjRtDeviceDescription description_;
};

class KelvinPjRtClient : public xla::PjRtClient {
 public:
  explicit KelvinPjRtClient()
    : device_(this),
      addressable_devices_({&device_}) {}

  int process_index() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return 0;
  }

  int device_count() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return 1;
  }

  int addressable_device_count() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return 1;
  }

  absl::Span<xla::PjRtDevice* const> devices() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return addressable_devices_;
  }

  absl::Span<xla::PjRtDevice* const> addressable_devices() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return addressable_devices_;
  }

  absl::StatusOr<xla::PjRtDevice*> LookupDevice(
      xla::PjRtGlobalDeviceId global_device_id) const override {
    std::cout << "Tuturu~ LookupDevice " << global_device_id << std::endl;
    for (xla::PjRtDevice* device : devices()) {
      if (device->global_device_id() == global_device_id) {
        return device;
      }
    }
    return absl::UnimplementedError("Unimplemented LookupDevice");
  }

  absl::StatusOr<xla::PjRtDevice*> LookupAddressableDevice(
      xla::PjRtLocalDeviceId local_device_id) const override {
    std::cout << "Tuturu~ LookupAddressableDevice "
              << local_device_id << std::endl;
    for (xla::PjRtDevice* device : addressable_devices()) {
      if (device->local_device_id() == local_device_id) {
        return device;
      }
    }
    return absl::UnimplementedError("Unimplemented LookupAddressableDevice");
  }

  void UpdateGlobalProcessInfo(
      absl::Span<tensorflow::CoordinatedTaskStateInfo> infos) override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
  }

  absl::Span<xla::PjRtMemorySpace* const> memory_spaces() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return {};
  }

  xla::PjRtPlatformId platform_id() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return xla::PjRtPlatformId(0);
  }

  absl::string_view platform_name() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return "kelvin";
  }

  absl::string_view platform_version() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return "0.0.1";
  }

  std::optional<std::shared_ptr<xla::KeyValueStoreInterface>>
  key_value_store() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return std::nullopt;
  }

  std::optional<xla::PjRtPluginAttributes> plugin_attributes() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return std::nullopt;
  }

  absl::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return xla::Unimplemented("GetDefaultDeviceAssignment is not supported.");
  }

  absl::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, std::optional<int> num_replicas_per_slice,
      int num_partitions, const xla::MultiSliceConfig* multi_slice_config) const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return xla::Unimplemented("Multi slice device assignment is not supported.");
  }

  absl::StatusOr<xla::Layout> GetDefaultLayout(
      xla::PrimitiveType element_type, absl::Span<const int64_t> dims) override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return xla::Unimplemented("GetDefaultLayout is not supported.");
  }

  // absl::StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
  //     const override {
  //   return xla::Unimplemented("GetHloCostAnalysis is not supported.");
  // }

  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> Compile(
      const xla::XlaComputation& computation,
      xla::CompileOptions options) override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> Compile(
      mlir::ModuleOp module, xla::CompileOptions options) override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
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
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
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
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    std::unique_ptr<KelvinPjRtExecutable> kelvin_executable(
        reinterpret_cast<KelvinPjRtExecutable*>(executable.release()));
    return std::make_unique<KelvinPjRtLoadedExecutable>(
        this, std::move(kelvin_executable));
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>>
  DeserializeExecutable(absl::string_view serialized,
                        std::optional<xla::CompileOptions> options) override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHostBuffer(
      const void* data, xla::PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      xla::PjRtMemorySpace* memory_space,
      const xla::Layout* device_layout) override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CreateUninitializedBuffer(
      const xla::Shape& shape, xla::PjRtMemorySpace* memory_space) override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CreateErrorBuffer(
      absl::Status error, const xla::Shape& shape, xla::PjRtMemorySpace* memory) override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return xla::Unimplemented("CreateErrorBuffer not supported.");
  }

  // xla::PjRtRuntimeType runtime_type() const override {
  //   return xla::PjRtRuntimeType::kStreamExecutor;
  // }

  // absl::StatusOr<std::string> SerializeExecutable() const override {
  //   return absl::UnimplementedError("Unimplemented");
  // }

 private:
  KelvinPjRtDevice device_;
  std::vector<xla::PjRtDevice*> addressable_devices_;
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
  return new PJRT_Error{absl::UnimplementedError("Implement me.")};
}

PJRT_Error* PJRT_Kelvin_DeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
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