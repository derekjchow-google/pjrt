#include "xla/pjrt/c/pjrt_c_api.h"

#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/status.h"



#include <iostream>

namespace xla {
class PjRtExecutable;
class PjRtBuffer;
}  // namespace xla

namespace kelvin {

class KelvinPjRtClient;

class KelvinPjRtDevice : public xla::PjRtDevice {
 public:
  explicit KelvinPjRtDevice(KelvinPjRtClient* client)
    : client_(client) {}

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
    LOG(FATAL) << "PjRtDeviceDescription not available (must override "
                  "PjRtDevice::description).";
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
    return 0;
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
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<xla::PjRtDevice*> LookupAddressableDevice(
      xla::PjRtLocalDeviceId local_device_id) const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return absl::UnimplementedError("Unimplemented");
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
    return absl::UnimplementedError("Unimplemented");
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