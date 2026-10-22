// gfx1201 bring-up stub: encoder flash attention uses gfx11 WMMA fragment
// layout and builtins. RDNA 4 needs a separate WMMA port; until then the
// Python Rdna35AttentionBackend keeps FLASHRT_RDNA35_HIP_ENCODER_ATTN=0 and
// uses SDPA.

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <stdexcept>

void attention_encoder_gqa_rdna(const __hip_bfloat16 * /*query*/,
                                const __hip_bfloat16 * /*key*/,
                                const __hip_bfloat16 * /*value*/,
                                __hip_bfloat16 * /*output*/, int /*query_rows*/,
                                int /*valid_kv_rows*/, int /*query_heads*/,
                                int /*head_dim*/, hipStream_t /*stream*/) {
  throw std::runtime_error(
      "attention_encoder_gqa_rdna is unavailable on gfx1201; use SDPA "
      "(FLASHRT_RDNA35_HIP_ENCODER_ATTN=0)");
}
