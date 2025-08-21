#include "kelvin_pjrt/copied_pjrt_buffer.h"

#include <iostream>

#include "absl/status/status.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/types.h"

namespace kelvin {

CopiedPjRtBuffer::CopiedPjRtBuffer(
    xla::PjRtClient* client, xla::PjRtDevice* device,
    xla::PjRtMemorySpace* memory_space, const void* host_data,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer)
    : client_(client),
      device_(device),
      memory_space_(memory_space),
      host_data_(host_data),
      on_done_with_host_buffer_(std::move(on_done_with_host_buffer)),
      shape_(xla::ShapeUtil::MakeNil()) {}

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer::ExternalReference>>
CopiedPjRtBuffer::AcquireExternalReference() {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
}

xla::PjRtFuture<> CopiedPjRtBuffer::ToLiteral(xla::MutableLiteralBase* literal) {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
}

xla::PjRtFuture<> CopiedPjRtBuffer::LazyToLiteral(
    absl::AnyInvocable<absl::StatusOr<xla::MutableLiteralBase*>() &&> generator) {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
}

absl::StatusOr<size_t> CopiedPjRtBuffer::GetOnDeviceSizeInBytes() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
}

xla::PjRtFuture<> CopiedPjRtBuffer::CopyRawToHost(void* dst, int64_t offset,
                                                  int64_t transfer_size) {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
}

void CopiedPjRtBuffer::Delete() {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
}

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer::ExternalReference>>
CopiedPjRtBuffer::ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
}

bool CopiedPjRtBuffer::IsDeleted() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
}

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>>
CopiedPjRtBuffer::CopyToMemorySpace(xla::PjRtMemorySpace* dst_memory_space) {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
}

void CopiedPjRtBuffer::CopyToRemoteDevice(
    xla::PjRtFuture<std::string> serialized_descriptor,
    RemoteSendCallback on_done) {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
}

xla::PjRtFuture<> CopiedPjRtBuffer::GetReadyFuture() {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
}

bool CopiedPjRtBuffer::IsOnCpu() const {
  return true;
}

}  // namespace kelvin
