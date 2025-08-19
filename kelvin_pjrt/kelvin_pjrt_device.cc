#include "kelvin_pjrt/kelvin_pjrt_device.h"

#include <iostream>

#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/status.h"

namespace kelvin {

KelvinPjRtDeviceDescription::KelvinPjRtDeviceDescription(int id,
                                                         int process_index)
    : id_(id),
      string_id_(absl::StrFormat("KelvinPjRtDevice(id=%d)", id)),
      process_index_(process_index),
      device_kind_("KelvinV2"),
      attributes_({}) {}

absl::string_view KelvinPjRtDeviceDescription::DebugString() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  // TODO(derekjchow): Implement more diligently
  return "KelvinPjRtDeviceDescription debug device";
}

absl::string_view KelvinPjRtDeviceDescription::ToString() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return string_id_;
}

const absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>&
KelvinPjRtDeviceDescription::Attributes() const {
  return attributes_;
}

absl::Span<const xla::PjRtMemorySpaceDescription* const>
KelvinPjRtDeviceDescription::memory_spaces() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return {};
}

absl::StatusOr<const xla::PjRtMemorySpaceDescription*>
KelvinPjRtDeviceDescription::default_memory_space() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("default_memory_space Not implemented.");
}

KelvinPjRtDevice::KelvinPjRtDevice(xla::PjRtClient* client)
    : client_(client),
      description_(42, 0) {}

bool KelvinPjRtDevice::IsAddressable() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return true;
}

ABSL_DEPRECATED("Use global_device_id() instead")
int KelvinPjRtDevice::id() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return global_device_id().value();
}

xla::PjRtGlobalDeviceId KelvinPjRtDevice::global_device_id() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return xla::PjRtGlobalDeviceId(description().id());
}

xla::PjRtLocalDeviceId KelvinPjRtDevice::local_device_id() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return xla::PjRtLocalDeviceId(local_hardware_id().value());
}

xla::PjRtLocalHardwareId KelvinPjRtDevice::local_hardware_id() const {
  return xla::PjRtLocalHardwareId(9001);
}

int KelvinPjRtDevice::process_index() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return description().process_index();
}

absl::string_view KelvinPjRtDevice::device_kind() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return description().device_kind();
}

absl::string_view KelvinPjRtDevice::DebugString() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return description().DebugString();
}

absl::string_view KelvinPjRtDevice::ToString() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return description().ToString();
}

const absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>&
KelvinPjRtDevice::Attributes() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return description().Attributes();
}

std::unique_ptr<xla::ScopedAsyncTrackingEvent>
KelvinPjRtDevice::CreateAsyncTrackingEvent(absl::string_view description) const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return nullptr;
}

absl::Status KelvinPjRtDevice::TransferToInfeed(
    const xla::LiteralSlice& literal) {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("TransferToInfeed not implemented");
}

absl::Status KelvinPjRtDevice::TransferFromOutfeed(
    xla::MutableBorrowingLiteral literal) {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("TransferFromOutfeed not implemented");
}

absl::StatusOr<tsl::AllocatorStats> KelvinPjRtDevice::GetAllocatorStats() const {
  return absl::UnimplementedError("GetAllocatorStats is not supported");
}

absl::Span<xla::PjRtMemorySpace* const> KelvinPjRtDevice::memory_spaces() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return {};
}

absl::StatusOr<xla::PjRtMemorySpace*> KelvinPjRtDevice::default_memory_space()
    const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("default_memory_space not implemented");
}

absl::StatusOr<xla::PjRtMemorySpace*> KelvinPjRtDevice::memory_space_by_kind(
    absl::string_view memory_space_kind) const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("memory_space_by_kind not implemented");
}

absl::StatusOr<std::intptr_t> KelvinPjRtDevice::GetStreamForExternalReadyEvents()
    const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError(
      "PjRtDevice::GetStreamForExternalReadyEvents only implemented for "
      "GPU");
}

absl::StatusOr<bool> KelvinPjRtDevice::PoisonExecution(int32_t launch_id,
                                                     absl::Status error) {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("PoisonExecution is not supported");
}

}  // namespace kelvin
