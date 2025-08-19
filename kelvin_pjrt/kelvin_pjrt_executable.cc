#include "kelvin_pjrt/kelvin_pjrt_executable.h"

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace kelvin {

KelvinPjRtExecutable::KelvinPjRtExecutable(std::string name,
                                           std::string object_code)
    : name_(std::move(name)), object_code_(std::move(object_code)) {}

absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>>
KelvinPjRtExecutable::GetHloModules() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("GetHloModules Not implemented.");
}

absl::StatusOr<std::vector<xla::Shape>> KelvinPjRtExecutable::GetOutputShapes()
    const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("GetOutputShapes Not implemented.");
}

absl::StatusOr<std::vector<std::vector<xla::PrimitiveType>>>
KelvinPjRtExecutable::GetOutputElementTypes() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("GetOutputElementTypes Not implemented.");
}

absl::StatusOr<std::vector<std::vector<xla::DimensionVector>>>
KelvinPjRtExecutable::GetOutputDimensions() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("GetOutputDimensions Not implemented.");
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
KelvinPjRtExecutable::GetParameterLayouts() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("GetParameterLayouts is not implemented.");
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
KelvinPjRtExecutable::GetOutputLayouts() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("GetOutputLayouts is not implemented.");
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
KelvinPjRtExecutable::GetOutputMemoryKinds() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("GetOutputMemoryKinds is not implemented.");
}

std::optional<std::vector<xla::OpSharding>>
KelvinPjRtExecutable::GetParameterShardings() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return std::nullopt;
}

std::optional<std::vector<xla::OpSharding>>
KelvinPjRtExecutable::GetOutputShardings() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return std::nullopt;
}

absl::StatusOr<xla::CompiledMemoryStats>
KelvinPjRtExecutable::GetCompiledMemoryStats() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError(
      "GetCompiledMemoryStats is not implemented.");
}

absl::StatusOr<absl::flat_hash_map<std::string, xla::PjRtValueType>>
KelvinPjRtExecutable::GetCostAnalysis() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("GetCostAnalysis is not implemented.");
}

absl::StatusOr<std::string> KelvinPjRtExecutable::SerializeExecutable() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("SerializeExecutable is not implemented.");
}

absl::StatusOr<std::string> KelvinPjRtExecutable::FingerprintExecutable()
    const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError(
      "FingerprintExecutable is not implemented.");
}

absl::StatusOr<struct xla::CompileOptions>
KelvinPjRtExecutable::GetCompileOptions() const {
  std::cout << "Tuturu~ " << __PRETTY_FUNCTION__ << std::endl;
  return absl::UnimplementedError("GetCompileOptions is not implemented.");
}

}  // namespace kelvin
