#ifndef KELVIN_PJRT_KELVIN_PJRT_MEMORY_SPACE_H_
#define KELVIN_PJRT_KELVIN_PJRT_MEMORY_SPACE_H_

#include <iostream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "xla/pjrt/pjrt_client.h"

namespace kelvin {

class KelvinPjRtDevice;

class KelvinPjRtMemorySpace : public xla::PjRtMemorySpace {
 public:
  KelvinPjRtMemorySpace(xla::PjRtClient* client, int id);

  ~KelvinPjRtMemorySpace() override = default;

  void AddDevice(KelvinPjRtDevice* device);
  void RemoveDevice(KelvinPjRtDevice* device);

  xla::PjRtClient* client() const override { return client_; }

  absl::Span<xla::PjRtDevice* const> devices() const override {
    return devices_;
  }

  int id() const override { return id_; }

  absl::string_view kind() const override { return kind_; }

  int kind_id() const override { return kind_id_; }

  absl::string_view DebugString() const override { return debug_string_; }

  absl::string_view ToString() const override { return to_string_; }

 private:
  xla::PjRtClient* const client_;
  const int id_;
  std::vector<xla::PjRtDevice*> devices_;
  const std::string kind_;
  const int kind_id_;
  const std::string debug_string_;
  const std::string to_string_;
  ::mpact::sim::util::FlatDemandMemory memory_;
};

}  // namespace kelvin

#endif  // KELVIN_PJRT_KELVIN_PJRT_MEMORY_SPACE_H_