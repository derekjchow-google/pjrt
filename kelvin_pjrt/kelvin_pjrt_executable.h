#ifndef KELVIN_PJRT_KELVIN_PJRT_EXECUTABLE_H_
#define KELVIN_PJRT_KELVIN_PJRT_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_executable.h"

namespace kelvin {

class KelvinPjRtExecutable : public xla::PjRtExecutable {
 public:
  KelvinPjRtExecutable(std::string name, std::string object_code);

  ~KelvinPjRtExecutable() override = default;

  int num_replicas() const override { return 1; }
  int num_partitions() const override { return 1; }
  int64_t SizeOfGeneratedCodeInBytes() const override{
    return object_code_.size();
  }
  absl::string_view name() const override { return name_; }

  absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>> GetHloModules()
      const override;
  absl::StatusOr<std::vector<xla::Shape>> GetOutputShapes() const override;
  absl::StatusOr<std::vector<std::vector<xla::PrimitiveType>>>
  GetOutputElementTypes() const override;
  absl::StatusOr<std::vector<std::vector<xla::DimensionVector>>>
  GetOutputDimensions() const override;
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetParameterLayouts() const override;
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetOutputLayouts() const override;
  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override;
  std::optional<std::vector<xla::OpSharding>> GetParameterShardings()
      const override;
  std::optional<std::vector<xla::OpSharding>> GetOutputShardings()
      const override;
  absl::StatusOr<xla::CompiledMemoryStats> GetCompiledMemoryStats()
      const override;
  absl::StatusOr<absl::flat_hash_map<std::string, xla::PjRtValueType>>
  GetCostAnalysis() const override;
  absl::StatusOr<std::string> SerializeExecutable() const override;
  absl::StatusOr<std::string> FingerprintExecutable() const override;
  absl::StatusOr<struct xla::CompileOptions> GetCompileOptions() const override;

 private:
  const std::string name_;
  const std::string object_code_;
};

}  // namespace kelvin

#endif  // KELVIN_PJRT_KELVIN_PJRT_EXECUTABLE_H_
