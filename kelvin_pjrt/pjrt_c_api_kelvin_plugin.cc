#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/status.h"

namespace xla {
class PjRtExecutable;
class PjRtBuffer;
}  // namespace xla

namespace {

class KelvinPjRtClient : public xla::PjRtClient {
 public:
  int process_index() const override { return 0; }

  int device_count() const override { return 0; }

  int addressable_device_count() const override { return 0; }

  absl::Span<xla::PjRtDevice* const> devices() const override {
    return {};
  }

  absl::Span<xla::PjRtDevice* const> addressable_devices() const override {
    return {};
  }

  absl::StatusOr<xla::PjRtDevice*> LookupDevice(
      xla::PjRtGlobalDeviceId global_device_id) const override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<xla::PjRtDevice*> LookupAddressableDevice(
      xla::PjRtLocalDeviceId local_device_id) const override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> Compile(
      const xla::XlaComputation& computation,
      xla::CompileOptions options) override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> Compile(
      mlir::ModuleOp module, xla::CompileOptions options) override {
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHostBuffer(
      const void* data, xla::PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      xla::PjRtMemorySpace* memory_space,
      const xla::Layout* device_layout) override{
    return absl::UnimplementedError("Unimplemented");
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CreateUninitializedBuffer(
      const xla::Shape& shape, xla::PjRtMemorySpace* memory_space) override {
    return absl::UnimplementedError("Unimplemented");
  }

  xla::PjRtPlatformId platform_id() const override {
    return xla::PjRtPlatformId(0);
  }

  absl::string_view platform_name() const override { return "kelvin"; }

  absl::string_view platform_version() const override { return "0.0.1"; }

  // xla::PjRtRuntimeType runtime_type() const override {
  //   return xla::PjRtRuntimeType::kStreamExecutor;
  // }

  // absl::StatusOr<std::string> SerializeExecutable() const override {
  //   return absl::UnimplementedError("Unimplemented");
  // }

  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>>
  DeserializeExecutable(absl::string_view serialized,
                        std::optional<xla::CompileOptions> options) override {
    return absl::UnimplementedError("Unimplemented");
  }
};

}  // namespace


const PJRT_Api* GetPjrtApi() {
  // TODO(b/261916900): implement.
  return nullptr;
}