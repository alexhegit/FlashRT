// gfx1201 bring-up stub: gfx11 WMMA builtins are not selectable on RDNA 4.
// The Python Rdna35GemmBackend keeps FLASHRT_RDNA35_HIP_SMALLM=0 and uses
// hipBLASLt instead. These symbols exist so the RDNA binding surface links.

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <stdexcept>

void smallm_wmma_bf16_rdna(const __hip_bfloat16 * /*input*/,
                           const __hip_bfloat16 * /*weight_nt*/,
                           const __hip_bfloat16 * /*bias*/,
                           __hip_bfloat16 * /*output*/, int /*rows*/,
                           int /*columns*/, int /*inner*/,
                           hipStream_t /*stream*/) {
  throw std::runtime_error(
      "smallm_wmma_bf16_rdna is unavailable on gfx1201; use hipBLASLt "
      "(FLASHRT_RDNA35_HIP_SMALLM=0)");
}

void smallm_wmma_bf16_residual_rdna(const __hip_bfloat16 * /*input*/,
                                    const __hip_bfloat16 * /*weight_nt*/,
                                    const __hip_bfloat16 * /*bias*/,
                                    __hip_bfloat16 * /*output*/, int /*rows*/,
                                    int /*columns*/, int /*inner*/,
                                    hipStream_t /*stream*/) {
  throw std::runtime_error(
      "smallm_wmma_bf16_residual_rdna is unavailable on gfx1201; use "
      "hipBLASLt (FLASHRT_RDNA35_HIP_SMALLM=0)");
}
