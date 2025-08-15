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

// TODO(derekjchow): Implement a "device".

class KelvinPjRtClient : public xla::PjRtClient {
 public:
  int process_index() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return 0;
  }

  int device_count() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return 0;
  }

  int addressable_device_count() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return 0;
  }

  absl::Span<xla::PjRtDevice* const> devices() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return {};
  }

  absl::Span<xla::PjRtDevice* const> addressable_devices() const override {
    std::cout << "Tuturu~ " << __FUNCTION__ << std::endl;
    return {};
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