// Chlorine PLE semantics. Independent model source is pinned by oracle fixtures.
#pragma once
#include <cmath>
#include <cstddef>
#ifndef FLASH_PLE_CAUSAL
#define FLASH_PLE_CAUSAL 0
#endif
#ifdef __HIPCC__
#define FLASH_PLE_HD __host__ __device__ __forceinline__
#else
#define FLASH_PLE_HD inline
#endif
namespace chlorine_flash {
inline constexpr bool ple_causal = FLASH_PLE_CAUSAL != 0;
inline constexpr int ple_ring_slots = ple_causal ? 10 : 9;
inline constexpr const char* ple_semantics = ple_causal ? "causal-dilation3-v1" : "legacy-nine-slot-alias-v1";
FLASH_PLE_HD int ple_tap_back(int tap) { return !ple_causal && tap == 0 ? 0 : (3-tap)*3; }
FLASH_PLE_HD float ple_gate(float dot_scaled) {
  float sign = dot_scaled < 0 ? -1.f : (dot_scaled > 0 || !ple_causal ? 1.f : 0.f);
  float x=sign*sqrtf(fmaxf(fabsf(dot_scaled),1e-6f));
  return 1.f/(1.f+expf(-x));
}
FLASH_PLE_HD float ple_conv_row(const float* current,const float* ring,const float* weights,
                               int t,int channel,int width,int base) {
  float sum=0.f;
  for(int tap=0;tap<4;tap++) {
    const int back=ple_tap_back(tap),absolute=base+t-back;
    float x=0.f;
    if(absolute>=0) x=t>=back ? current[(size_t)(t-back)*width+channel]
        : ring[(size_t)(absolute%ple_ring_slots)*width+channel];
    sum+=weights[channel*4+tap]*x;
  }
  return sum/(1.f+expf(-sum));
}
// Ignore predecessors across an end-of-text boundary, including chunk history.
template<class Token>
inline Token ple_history_token(const Token* ids,int t,int back,int eos) {
  if(t<back)return eos;
  if(ple_causal) for(int j=1;j<=back;j++) if(ids[t-j]==eos)return eos;
  return ids[t-back];
}
} // namespace chlorine_flash
#undef FLASH_PLE_HD
