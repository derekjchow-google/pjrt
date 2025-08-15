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

  ~PjRtDevice() override = default;

  // Return the client that owns this device.
  virtual PjRtClient* client() const = 0;

  // Whether client can issue command to this device.
  virtual bool IsAddressable() const = 0;

  virtual const PjRtDeviceDescription& description() const {
    LOG(FATAL) << "PjRtDeviceDescription not available (must override "
                  "PjRtDevice::description).";
  }

  // The ID of this device. IDs are unique among devices of this type
  // (e.g. CPUs, GPUs). On multi-host platforms, this will be unique across all
  // hosts' devices.  This is the ID that should be used in a DeviceAssignment.
  ABSL_DEPRECATED("Use global_device_id() instead")
  virtual int id() const { return global_device_id().value(); }

  // There are several different IDs for a PJRT device.
  //
  // - global_device_id: The logical global device ID. This is unique among
  // devices of this type (e.g. CPUs, GPUs). On multi-host platforms, this will
  // be unique across all hosts' devices.  This is the ID that should be used in
  // a DeviceAssignment.
  //
  // - local_device_id: The logical local device ID. This will be used to look
  // up an addressable device local to a given client. It is -1 if undefined.
  //
  // - local_hardware_id: The physical local device ID, e.g., the CUDA device
  // number. Multiple PJRT devices can have the same local_hardware_id if
  // these PJRT devices share the same physical device. This is useful for
  // identifying which physical device when interacting with non-JAX code. In
  // general, not guaranteed to be dense, and -1 if undefined.

  // TODO(b/314368788): Remove `id()` and replace it with this function.
  virtual PjRtGlobalDeviceId global_device_id() const {
    return PjRtGlobalDeviceId(description().id());
  }

  virtual PjRtLocalDeviceId local_device_id() const {
    // By default, local_device_id is the same as local_hardware_id when there
    // is only one PJRT device on a physical device.
    return PjRtLocalDeviceId(local_hardware_id().value());
  }

  // Opaque hardware ID, e.g., the CUDA device number, useful for identifying
  // which GPU when interacting with non-JAX code. In general, not guaranteed to
  // be dense, and -1 if undefined.
  virtual PjRtLocalHardwareId local_hardware_id() const = 0;

  // The index of the process that this device belongs to, i.e. is addressable
  // from. This is not always identical to PjRtClient::process_index() in a
  // multi-process setting, where each client can see devices from all
  // processes, but only a subset of them are addressable and have the same
  // process_index as the client.
  virtual int process_index() const { return description().process_index(); }

  // A vendor-dependent string that uniquely identifies the kind of device,
  // e.g., "Tesla V100-SXM2-16GB". May be used to determine whether two GPUs are
  // compatible compilation.
  virtual absl::string_view device_kind() const {
    return description().device_kind();
  }

  // Debug string suitable for logging when errors occur. Should be verbose
  // enough to describe the current device unambiguously.
  virtual absl::string_view DebugString() const {
    return description().DebugString();
  }

  // Debug string suitable for reading by end users, should be reasonably terse,
  // for example: "CpuDevice(id=0)".
  virtual absl::string_view ToString() const {
    return description().ToString();
  }

  // Returns vendor specific attributes about the device. For example the model
  // number of a GPU, or the mesh coordinates of a TPU device. The returned
  // reference will remain valid for the lifetime of the PjRtDevice.
  virtual const absl::flat_hash_map<std::string, PjRtDeviceAttribute>&
  Attributes() const {
    return description().Attributes();
  }

  // Returns a scoped event that the caller uses to tell the PjRtClient that
  // there is asynchronous work happening that depends on activity on the
  // PjRtDevice. See comment on class definition in pjrt_future.h.
  //
  // Only some PjRtDevice implementations support ScopedAsyncTrackingEvent, and
  // those that do not will return nullptr.
  virtual std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const = 0;

  // Transfer the given literal to the infeed queue.
  virtual absl::Status TransferToInfeed(const LiteralSlice& literal) = 0;

  // Transfer and return a value of the given shape from the outfeed queue.
  virtual absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) = 0;

  // Returns allocator stats for the device. Only some PjRtDevice
  // implementations support allocator_stats, and those that do not will return
  // an Unimplemented error.
  virtual absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() const {
    return absl::UnimplementedError("GetAllocatorStats is not supported");
  }

  // Returns all memory spaces attached to this device.
  // The memory spaces are in no particular order.
  virtual absl::Span<PjRtMemorySpace* const> memory_spaces() const = 0;

  // Returns the default memory space attached to this device.
  virtual absl::StatusOr<PjRtMemorySpace*> default_memory_space() const = 0;

  virtual absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind(
      absl::string_view memory_space_kind) const {
    return absl::UnimplementedError("memory_space_by_kind not implemented");
  }

  // Returns a platform-specific stream handle that should be used to track when
  // an externally-managed buffer is ready to use on this device. This is
  // intended to support dlpack on GPU and is not expected to be implemented for
  // all hardware platforms.
  virtual absl::StatusOr<std::intptr_t> GetStreamForExternalReadyEvents()
      const {
    return absl::UnimplementedError(
        "PjRtDevice::GetStreamForExternalReadyEvents only implemented for "
        "GPU");
  }

  // Experimental: Poisons the earliest execution on this device with given
  // launch_id if it's not finished yet, i.e. makes its output buffers error.
  //
  // Returns true if the output buffers have been successfully poisoned.
  //
  // Returns false if the output buffers were not successfully poisoned because
  // launch_id is not in the list of executions that have not yet completed.
  // This may happen either because the execution corresponding to launch_id has
  // already completed, or because an incorrect launch_id was supplied.
  //
  // Returns error otherwise, including in the case that poisoning is not
  // implemented by this client.
  virtual absl::StatusOr<bool> PoisonExecution(int32_t launch_id,
                                               absl::Status error) {
    return absl::UnimplementedError("PoisonExecution is not supported");
  }

 private:
  KelvinPjRtClient* const client_;
};

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