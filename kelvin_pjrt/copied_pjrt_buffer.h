#ifndef KELVIN_PJRT_COPIED_PJRT_BUFFER_H_
#define KELVIN_PJRT_COPIED_PJRT_BUFFER_H_

#include "absl/functional/any_invocable.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace kelvin {

class CopiedPjRtBuffer : public xla::PjRtBuffer {
 public:
  CopiedPjRtBuffer(
      xla::PjRtClient* client, xla::PjRtDevice* device,
      xla::PjRtMemorySpace* memory_space, const void* host_data,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer);
  ~CopiedPjRtBuffer() override {
    if (on_done_with_host_buffer_) {
      std::move(on_done_with_host_buffer_)();
    }
  }

  const xla::Shape& on_device_shape() const override { return shape_; }
  xla::PjRtMemorySpace* memory_space() const { return memory_space_; }
  xla::PjRtClient* client() const override { return client_; }
  xla::PjRtDevice* device() const override { return device_; }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer::ExternalReference>>
  AcquireExternalReference() override;

  xla::PjRtFuture<> ToLiteral(xla::MutableLiteralBase* literal) override;

  xla::PjRtFuture<> LazyToLiteral(
      absl::AnyInvocable<absl::StatusOr<xla::MutableLiteralBase*>() &&> generator) override;

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  xla::PjRtFuture<> CopyRawToHost(void* dst, int64_t offset,
                                  int64_t transfer_size) override;

  void Delete() override;

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer::ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override;

  bool IsDeleted() const override;

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CopyToMemorySpace(
      xla::PjRtMemorySpace* dst_memory_space) override;

  void CopyToRemoteDevice(xla::PjRtFuture<std::string> serialized_descriptor,
                                  RemoteSendCallback on_done) override;

  xla::PjRtFuture<> GetReadyFuture() override;

  bool IsOnCpu() const override;

 private:
  xla::PjRtClient* const client_;
  xla::PjRtDevice* const device_;
  xla::PjRtMemorySpace* const memory_space_;
  const void* const host_data_;
  absl::AnyInvocable<void() &&> on_done_with_host_buffer_;
  const xla::Shape shape_;
};

}  // namespace kelvin

#endif  // KELVIN_PJRT_COPIED_PJRT_BUFFER_H_
