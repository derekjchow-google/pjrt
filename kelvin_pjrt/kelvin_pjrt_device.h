#ifndef KELVIN_PJRT_KELVIN_PJRT_DEVICE_H_
#define KELVIN_PJRT_KELVIN_PJRT_DEVICE_H_

#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/status.h"
#include "absl/strings/str_format.h"
#include <iostream>

namespace kelvin {

class KelvinPjRtClient;

class KelvinPjRtDeviceDescription : public xla::PjRtDeviceDescription {
 public:
  KelvinPjRtDeviceDescription(int id, int process_index);
  ~KelvinPjRtDeviceDescription() override = default;

  int id() const override { return id_; }
  int process_index() const override { return process_index_; }
  absl::string_view device_kind() const override { return device_kind_; }
  absl::string_view DebugString() const override;
  absl::string_view ToString() const override;

  const absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>&
  Attributes() const override;

  absl::Span<const xla::PjRtMemorySpaceDescription* const> memory_spaces()
      const override;

  absl::StatusOr<const xla::PjRtMemorySpaceDescription*>
  default_memory_space() const override;

 public:
  const int id_;
  const std::string string_id_;
  const int process_index_;
  const std::string device_kind_;
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;
};

class KelvinPjRtDevice : public xla::PjRtDevice {
 public:
  explicit KelvinPjRtDevice(xla::PjRtClient* client);
  ~KelvinPjRtDevice() override = default;

  xla::PjRtClient* client() const override { return client_; }
  bool IsAddressable() const override;
  const xla::PjRtDeviceDescription& description() const override {
    return description_;
  }
  ABSL_DEPRECATED("Use global_device_id() instead")
  int id() const override;
  xla::PjRtGlobalDeviceId global_device_id() const override;
  xla::PjRtLocalDeviceId local_device_id() const override;
  xla::PjRtLocalHardwareId local_hardware_id() const override;
  int process_index() const override;
  absl::string_view device_kind() const override;
  absl::string_view DebugString() const override;
  absl::string_view ToString() const override;
  const absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>&
  Attributes() const override;
  std::unique_ptr<xla::ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override;
  absl::Status TransferToInfeed(const xla::LiteralSlice& literal) override;
  absl::Status TransferFromOutfeed(xla::MutableBorrowingLiteral literal) override;
  absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() const override;
  absl::Span<xla::PjRtMemorySpace* const> memory_spaces() const override;
  absl::StatusOr<xla::PjRtMemorySpace*> default_memory_space() const override;
  absl::StatusOr<xla::PjRtMemorySpace*> memory_space_by_kind(
      absl::string_view memory_space_kind) const override;
  absl::StatusOr<std::intptr_t> GetStreamForExternalReadyEvents()
      const override;
  absl::StatusOr<bool> PoisonExecution(int32_t launch_id,
                                       absl::Status error) override;

 private:
  xla::PjRtClient* const client_;
  KelvinPjRtDeviceDescription description_;
};

}  // namespace kelvin

#endif  // KELVIN_PJRT_KELVIN_PJRT_DEVICE_H_
