// Chlorine Flash packed expert decoding; ggml table attribution in LICENSE.ggml.
#pragma once
#include <cstdint>
#include <cmath>
#ifdef __HIPCC__
#define FLASH_HD __host__ __device__ __forceinline__
#else
#define FLASH_HD inline
#endif
namespace flash_quant {
inline constexpr uint32_t iq3_grid[512] = {
#include "iq3_grid.inc"
};
inline constexpr int8_t iq4_grid[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
#ifdef __HIPCC__
__device__ __constant__ inline uint32_t iq3_device_grid[512] = {
#include "iq3_grid.inc"
};
__device__ __constant__ inline int8_t iq4_device_grid[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
#endif
FLASH_HD uint16_t u16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
FLASH_HD float half_value(uint16_t h) {
  const unsigned e=(h>>10)&31, m=h&1023;
  float v=e==0 ? float(m)*0x1p-24f : e==31 ? (m ? NAN : INFINITY) : ldexpf(float(1024+m),int(e)-25);
  return h&0x8000 ? -v : v;
}
FLASH_HD uint32_t grid3(int i) {
#ifdef __HIP_DEVICE_COMPILE__
  return iq3_device_grid[i];
#else
  return iq3_grid[i];
#endif
}
FLASH_HD int grid4(int i) {
#ifdef __HIP_DEVICE_COMPILE__
  return iq4_device_grid[i];
#else
  return iq4_grid[i];
#endif
}
// Caller checks type and row/block alignment once, before dispatch.
FLASH_HD float value(int type, const uint8_t* row, uint64_t col) {
  if(type==21) {
    const uint8_t* b=row+(col/256)*110;
    const unsigned j=col%256, sub=j/32, t=j%32, group=t/8, half=(t%8)/4;
    const unsigned index=b[2+sub*8+group*2+half] | (((b[66+sub]>>(group*2+half))&1)<<8);
    const int magnitude=(grid3(index)>>(8*(t%4)))&255;
    const unsigned s=(b[106+sub/2]>>(4*(sub%2)))&15;
    const float d=half_value(u16(b))*float(1+2*s);
    const float w=d*float(magnitude);
    return b[74+sub*4+group] & (1u<<(t%8)) ? -w : w;
  }
  if(type==20) {
    const uint8_t* b=row+(col/32)*18; const unsigned j=col%32;
    const unsigned code=(b[2+j%16]>>(j>=16 ? 4 : 0))&15;
    return half_value(u16(b))*float(grid4(code));
  }
  if(type==23) {
    const uint8_t* b=row+(col/256)*136; const unsigned j=col%256, sub=j/32, t=j%32;
    const int scale=int(((b[4+sub/2]>>(4*(sub%2)))&15) | (((u16(b+2)>>(2*sub))&3)<<4))-32;
    const int code=(b[8+sub*16+t%16]>>(t>=16 ? 4 : 0))&15;
    return (half_value(u16(b))*float(scale))*float(grid4(code));
  }
  // Q8_0 (type 8), used by the five local down-expert planes.
  const uint8_t* b=row+(col/32)*34;
  return half_value(u16(b))*float(int8_t(b[2+col%32]));
}
} // namespace flash_quant
#undef FLASH_HD
