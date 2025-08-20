#include "kelvin_pjrt/kelvin_pjrt_memory_space.h"

#include <string>

#include "absl/strings/str_format.h"

namespace kelvin {

KelvinPjRtMemorySpace::KelvinPjRtMemorySpace(
    xla::PjRtClient* client, int id, xla::PjRtDevice* device)
  : client_(client),
    id_(id),
    device_(device),
    devices_(&device_, 1),
    kind_("kelvin"),
    kind_id_(0),
    debug_string_(absl::StrFormat("KelvinPjRtMemorySpace(id=%d)", id)),
    to_string_(absl::StrFormat("KelvinPjRtMemorySpace(id=%d)", id)) {}


}  // namespace kelvin