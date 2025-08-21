#include "kelvin_pjrt/kelvin_pjrt_memory_space.h"

#include <string>
#include <algorithm>

#include "absl/strings/str_format.h"
#include "kelvin_pjrt/kelvin_pjrt_device.h"

namespace kelvin {

KelvinPjRtMemorySpace::KelvinPjRtMemorySpace(xla::PjRtClient* client, int id)
    : client_(client),
      id_(id),
      devices_(),
      kind_("kelvin"),
      kind_id_(0),
      debug_string_(absl::StrFormat("KelvinPjRtMemorySpace(id=%d)", id)),
      to_string_(absl::StrFormat("KelvinPjRtMemorySpace(id=%d)", id)) {}

void KelvinPjRtMemorySpace::AddDevice(KelvinPjRtDevice* device) {
  devices_.push_back(device);
}

void KelvinPjRtMemorySpace::RemoveDevice(KelvinPjRtDevice* device) {
  devices_.erase(std::remove_if(devices_.begin(), devices_.end(),
                                [device](xla::PjRtDevice* d) {
                                  return d == device;
                                }),
                   devices_.end());
}

}  // namespace kelvin