// Chlorine Flash additions. Retains upstream routing and ordered reduction.
// IQ3_S prefill, original kernel: decode only a 128x16 weight tile, never a full expert plane.
// Superseded by k_moe_lut<kIQ3S> (27_kernels_moe_lut.inc); kept as the bit-identity reference.
// FP16 operand rounding and FP32 WMMA accumulation require the GPU quality gate.
__launch_bounds__(256) __global__
void k_flash_iq3_up(const uint8_t* gate,const uint8_t* up,const __half* x,
                   const int* tokidx,const MoeTile* tiles,const int* ntiles,
                   __half* hid,int m,int k) {
  if(int(blockIdx.y)>=*ntiles) return;
  const MoeTile tile=tiles[blockIdx.y];
  const int tid=threadIdx.x,wave=tid/32,lane=tid%32,sub=lane%16,half=lane/16;
  const int r0=int(blockIdx.x)*64, live=(tile.count+15)/16;
  __shared__ __half weights[128][16];
  __shared__ __half inputs[64][16];
  __shared__ float result[8][16][16];
  moegg::v8f acc[4]={};
  const size_t rb=(size_t)(k/256)*110;
  for(int kb=0;kb<k;kb+=16) {
    for(int i=tid;i<128*16;i+=256) {
      int r=i/16,c=i%16,rr=r0+r%64;
      const uint8_t* base=(r<64?gate:up)+((size_t)tile.expert*m+(rr<m?rr:0))*rb;
      weights[r][c]=__float2half(rr<m?flash_quant::value(21,base,kb+c):0.f);
    }
    for(int i=tid;i<64*16;i+=256) {
      int t=i/16,c=i%16;
      inputs[t][c]=t<tile.count?x[(size_t)tokidx[tile.first+t]*k+kb+c]:__float2half(0.f);
    }
    __syncthreads();
    moegg::v16h a;
    for(int j=0;j<16;++j) a[j]=weights[wave*16+sub][j];
    for(int t=0;t<live;++t) {
      moegg::v16h b;
      for(int j=0;j<16;++j)b[j]=inputs[t*16+sub][j];
      acc[t]=moegg::wmma(a,b,acc[t]);
    }
    __syncthreads();
  }
  for(int t=0;t<live;++t) {
    for(int j=0;j<8;++j)result[wave][sub][2*j+half]=acc[t][j];
    __syncthreads();
    for(int i=tid;i<16*64;i+=256) {
      int tr=t*16+i/64,r=i%64;
      if(tr<tile.count && r0+r<m) {
        float g=result[r/16][i/64][r%16],u=result[r/16+4][i/64][r%16];
        hid[(size_t)(tile.first+tr)*m+r0+r]=__float2half(u*g*moegg::sigmoidf_(g));
      }
    }
    __syncthreads();
  }
}
static moelut::LutW flash_iq_weights(const uint8_t* w,const uint8_t* up,int rows) {
  return moelut::LutW{w,up,nullptr,nullptr,nullptr,0,rows,0};
}
static bool flash_moe_up(int type,const uint8_t* gate,const uint8_t* up,const __half* x,
  const int* tokidx,const MoeTile* tiles,const int* ntiles,int max_tiles,__half* hid,int m,int k,hipStream_t st,
  int tile=64) {
  if(type==21) {
    if(k%256 || m%64) return false;
    // LUT-staged kernel by default; GDEC_FLASH_IQ3_SCALAR=1 keeps the original
    // per-element kernel. Their outputs are bit-identical (moe_test checks it).
    static const bool scalar=getenv("GDEC_FLASH_IQ3_SCALAR")!=nullptr;
    if(!scalar) return moe_lut_up(21,flash_iq_weights(gate,up,m),x,tokidx,tiles,ntiles,max_tiles,hid,m,k,st,tile);
    if(tile!=64) return false;
    k_flash_iq3_up<<<dim3((m+63)/64,max_tiles),256,0,st>>>(gate,up,x,tokidx,tiles,ntiles,hid,m,k);
    return true;
  }
  if(type==20 || type==23)
    return moe_lut_up(type,flash_iq_weights(gate,up,m),x,tokidx,tiles,ntiles,max_tiles,hid,m,k,st,tile);
  if(tile!=64) return false;  // the raw-block kernels are 64-slot only
  return moe_gg_up(type,gate,up,x,tokidx,tiles,ntiles,max_tiles,hid,m,k,st);
}
static bool flash_moe_down(int type,const uint8_t* down,const __half* hid,
  const MoeTile* tiles,const int* ntiles,int max_tiles,float* pairs,int m,int k,hipStream_t st,int tile=64,
  bool half_pairs=false) {
  if(type==20)
    return moe_lut_down(type,flash_iq_weights(down,nullptr,m),hid,tiles,ntiles,max_tiles,pairs,m,k,st,tile,half_pairs);
  if(tile!=64 || half_pairs) return false;
  return moe_gg_down(type,down,hid,tiles,ntiles,max_tiles,pairs,m,k,st);
}
static bool flash_moe_gemv(int gu,int dn,const uint8_t* gate,const uint8_t* up,
  const uint8_t* down,const float* x,const int* ids,const float* ws,float* hid,float* out,
  int P,int d,int mid,int stride,hipStream_t st,int* uq=nullptr,float* pairs=nullptr) {
  if(gu!=21 && gu!=20 && gu!=23)
    return moe_gg_gemv(gu,dn,gate,up,down,x,ids,ws,hid,out,P,d,mid,stride,st,uq,pairs);
  if(d%256 || mid%32 || (dn!=20 && dn!=8)) return false;
  unsigned blocks=(unsigned)(((size_t)P*10*mid+7)/8);
#define FLASH_GU(T) k_moe_gg_gemv_gu<T><<<blocks,256,0,st>>>(gate,up,x,ids,hid,mid,d,10,stride,P)
  if(gu==21) { FLASH_GU(21); } else if(gu==20) { FLASH_GU(20); } else { FLASH_GU(23); }
#undef FLASH_GU
  unsigned db=(unsigned)(((size_t)P*((d+3)/4)+7)/8);
  if(dn==20) k_moe_gg_gemv_dn<20,8,10><<<db,256,0,st>>>(down,hid,ids,ws,out,d,mid,stride,P);
  else k_moe_gg_gemv_dn<8,8,10><<<db,256,0,st>>>(down,hid,ids,ws,out,d,mid,stride,P);
  return true;
}
// Weighted deterministic reduce over F16 pair rows, with the optional shared-expert
// add. Same slot order and FP32 accumulation as k_moe_reduce_pw / _sg; only the
// pair rows were rounded to F16 by the down kernel (GDEC_FLASH_PAIRS_F16).
template <int K>
__global__ void k_flash_reduce_pw_f16(const __half* __restrict__ pairs,const int* __restrict__ pairids,
                                      const float* __restrict__ pw,const float* __restrict__ ey,
                                      const float* __restrict__ sg,float* __restrict__ out,int P,int Dd) {
  __shared__ int ids[K];
  int tid=threadIdx.x;
  int t=blockIdx.y,d=tid+(int)blockIdx.x*blockDim.x;
  if(t>=P) return;
  if(tid<K) ids[tid]=pairids[t*K+tid];
  __syncthreads();
  if(d>=Dd) return;
  float sum=0.f;
#pragma unroll
  for(int s=0;s<K;s++) sum+=pw[ids[s]]*__half2float(pairs[(size_t)ids[s]*Dd+d]);
  const size_t i=(size_t)t*Dd+d;
  if(ey) {
    float s=1.f/(1.f+expf(-sg[t]));
    sum+=s*ey[i];
  }
  out[i]=sum;
}
// F32 matrices remain F32 in the GGUF mapping (routers, HC injection, GDN gates).
// GDEC_F32_GEMM_BF16: batched GEMMs may read such a matrix as bf16 when every value
// is exact in bf16 (the model's native weight type). This copies the high halves
// and counts any value that would lose bits; one inexact value keeps the F32 path.
__global__ void k_flash_f32_to_bf16_exact(const float* w,uint16_t* out,size_t n,unsigned* inexact) {
  const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n)return;
  const uint32_t bits=__builtin_bit_cast(uint32_t,w[i]);
  out[i]=uint16_t(bits>>16);
  if(bits&0xFFFFu)atomicAdd(inexact,1u);
}
// Q6_K vocabulary head. One wave per output row; share decoded weights across
// up to eight inputs. P=1 and multi-row use the same accumulation/reduction order.
// Original head kernel: one flash_quant::value call per element. Kept as the
// bit-identity reference for k_flash_q6_gemv (mapping_gpu_test compares them).
template<bool XBf16=false>
__global__ void k_flash_q6_gemv_scalar(const uint8_t* w,const void* x,float* out,
                              int rows,int cols,int P,size_t xs) {
  const int lane=threadIdx.x%32,r=blockIdx.x*8+threadIdx.x/32;
  if(r>=rows)return;
  const uint8_t* row=w+(size_t)r*(cols/256)*210;
  float sum[8]={};
  for(int c=lane;c<cols;c+=32) {
    const float weight=flash_quant::value(14,row,c);
    for(int t=0;t<P;t++) {
      const float xv=XBf16 ? __uint_as_float(unsigned(((const uint16_t*)x)[t*xs+c])<<16)
                          : ((const float*)x)[t*xs+c];
      sum[t]+=weight*xv;
    }
  }
  for(int t=0;t<P;t++) {
    float v=sum[t];
    for(int offset=16;offset;offset>>=1)v+=__shfl_down(v,offset,32);
    if(lane==0)out[(size_t)t*rows+r]=v;
  }
}
// A lane owns the elements whose in-group index equals its lane id, so per
// (block, half) it needs two low-nibble bytes, one high-bit byte, eight scale
// bytes and the block's d. The element order and FP32 expressions are the scalar
// kernel's, so the sums are bit-identical. BLOCKS > 0 (a row of exactly that many
// 256-element blocks) fetches the whole row before any arithmetic, so a wave
// waits on memory once instead of once per half block.
struct FlashQ6Half { uint32_t codes; uint64_t scales; };  // codes = q0 | q1 << 8 | qh << 16
__device__ __forceinline__ FlashQ6Half flash_q6_fetch(const uint8_t* b,int half,int lane) {
  FlashQ6Half h;
  h.codes=uint32_t(b[half*64+lane]) | (uint32_t(b[half*64+32+lane])<<8) | (uint32_t(b[128+half*32+lane])<<16);
  __builtin_memcpy(&h.scales,b+192+half*8,8);
  return h;
}
template<bool XBf16>
__device__ __forceinline__ void flash_q6_half(const FlashQ6Half& h,float d,int base,int lane,
                                              const void* x,int P,size_t xs,float* sum) {
  const unsigned q0=h.codes&255,q1=(h.codes>>8)&255,qh=h.codes>>16;
#pragma unroll
  for(int group=0;group<4;++group) {
    const unsigned low=group%2 ? q1 : q0;
    const unsigned code=((low>>(group>=2?4:0))&15) | (((qh>>(group*2))&3)<<4);
    const int scale=int(int8_t((h.scales>>(8*(group*2+lane/16)))&255));
    const float weight=(d*float(scale))*float(int(code)-32);
    const int c=base+group*32+lane;
    for(int t=0;t<P;t++) {
      const float xv=XBf16 ? __uint_as_float(unsigned(((const uint16_t*)x)[t*xs+c])<<16)
                          : ((const float*)x)[t*xs+c];
      sum[t]+=weight*xv;
    }
  }
}
template<bool XBf16=false,int BLOCKS=0>
__global__ void k_flash_q6_gemv(const uint8_t* w,const void* x,float* out,
                              int rows,int cols,int P,size_t xs) {
  const int lane=threadIdx.x%32,r=blockIdx.x*8+threadIdx.x/32;
  if(r>=rows)return;
  const int blocks=cols/256;
  const uint8_t* row=w+(size_t)r*blocks*210;
  float sum[8]={};
  if constexpr (BLOCKS>0) {
    FlashQ6Half raw[BLOCKS][2];
    uint32_t dbits[BLOCKS];
#pragma unroll
    for(int blk=0;blk<BLOCKS;++blk) {
      const uint8_t* b=row+(size_t)blk*210;
      dbits[blk]=flash_quant::u16(b+208);
      raw[blk][0]=flash_q6_fetch(b,0,lane);
      raw[blk][1]=flash_q6_fetch(b,1,lane);
    }
#pragma unroll
    for(int blk=0;blk<BLOCKS;++blk) {
      const float d=flash_quant::half_value(uint16_t(dbits[blk]));
      flash_q6_half<XBf16>(raw[blk][0],d,blk*256,lane,x,P,xs,sum);
      flash_q6_half<XBf16>(raw[blk][1],d,blk*256+128,lane,x,P,xs,sum);
    }
  } else {
    for(int blk=0;blk<blocks;++blk) {
      const uint8_t* b=row+(size_t)blk*210;
      const float d=flash_quant::half_value(flash_quant::u16(b+208));
      flash_q6_half<XBf16>(flash_q6_fetch(b,0,lane),d,blk*256,lane,x,P,xs,sum);
      flash_q6_half<XBf16>(flash_q6_fetch(b,1,lane),d,blk*256+128,lane,x,P,xs,sum);
    }
  }
  for(int t=0;t<P;t++) {
    float v=sum[t];
    for(int offset=16;offset;offset>>=1)v+=__shfl_down(v,offset,32);
    if(lane==0)out[(size_t)t*rows+r]=v;
  }
}
static void flash_q6_gemv(const uint8_t* w,const void* x,float* out,int rows,
                         int cols,int P,size_t xs,bool bf16,hipStream_t st) {
  if(rows<1 || cols<1 || cols%256 || P<1 || P>8 || xs<(size_t)cols)
    throw std::runtime_error("flash Q6_K GEMV: invalid shape");
  // GDEC_FLASH_Q6_LOOP=1 keeps the per-block loop for every width (A/B checks).
  static const bool loop=getenv("GDEC_FLASH_Q6_LOOP")!=nullptr;
  if(cols==2560 && !loop) {
    if(bf16)k_flash_q6_gemv<true,10><<<(rows+7)/8,256,0,st>>>(w,x,out,rows,cols,P,xs);
    else k_flash_q6_gemv<false,10><<<(rows+7)/8,256,0,st>>>(w,x,out,rows,cols,P,xs);
    return;
  }
  if(bf16)k_flash_q6_gemv<true><<<(rows+7)/8,256,0,st>>>(w,x,out,rows,cols,P,xs);
  else k_flash_q6_gemv<false><<<(rows+7)/8,256,0,st>>>(w,x,out,rows,cols,P,xs);
}
// Reference F32 row dot: 256 lane partials (lane t takes columns t, t + 256, ...)
// reduced by a 256-entry LDS tree with a barrier per level. Kept for the tests.
__global__ void k_flash_f32_gemv_tree(const float* w,const float* x,float* out,int n,int k,size_t xs) {
  int r=blockIdx.x,t=blockIdx.y,tid=threadIdx.x;
  float v=0;for(int c=tid;c<k;c+=256)v+=w[(size_t)r*k+c]*x[(size_t)t*xs+c];
  __shared__ float sum[256];sum[tid]=v;__syncthreads();
  for(int s=128;s;s>>=1){if(tid<s)sum[tid]+=sum[tid+s];__syncthreads();}
  if(tid==0)out[(size_t)t*n+r]=sum[0];
}
// The same tree with one barrier. Level s adds entry t + s into entry t, so
// levels 128, 64, 32 pair waves (w, w + 4), (w, w + 2), (w, w + 1) of one lane
// column and levels 16 .. 1 pair lanes inside wave 0. Each lane therefore sums
// its column's eight partials from LDS in that pairing, then the lanes combine
// by shuffle in the same order. Every addition has the tree's two operands, so
// the total is bit for bit the tree's, and every thread of the block gets it.
// 256-thread blocks only; sum is a __shared__ float[256].
__device__ __forceinline__ float flash_tree256(float v,float* sum) {
  const int tid=threadIdx.x,lane=tid&31;
  sum[tid]=v;__syncthreads();
  float c[8];
#pragma unroll
  for(int w=0;w<8;w++)c[w]=sum[lane+32*w];
#pragma unroll
  for(int half=4;half;half>>=1)
#pragma unroll
    for(int w=0;w<half;w++)c[w]+=c[w+half];
  float t=c[0];
#pragma unroll
  for(int off=16;off;off>>=1)t+=__shfl_xor(t,off,32);
  return t;
}
__global__ void k_flash_f32_gemv(const float* w,const float* x,float* out,int n,int k,size_t xs) {
  int r=blockIdx.x,t=blockIdx.y,tid=threadIdx.x;
  float v=0;for(int c=tid;c<k;c+=256)v+=w[(size_t)r*k+c]*x[(size_t)t*xs+c];
  __shared__ float sum[256];
  const float total=flash_tree256(v,sum);
  if(tid==0)out[(size_t)t*n+r]=total;
}

// --- Fused single-token kernels (serial decode) -----------------------------
// A decoded token issues about 37 small kernels per layer, and every launch
// costs time whatever its size. Each kernel below replaces two or three
// neighbouring launches and keeps every expression and summation order of the
// kernels it replaces (the F32 dots are k_flash_f32_gemv's lane partials and
// tree order), so decode output does not change: mapping_gpu_test compares
// each one bit for bit with the launches it replaces. GDEC_FLASH_DECODE_FUSE=0
// restores the separate launches.

// GDN gates: in_proj_a / in_proj_b (F32, hv rows each) and k_gdn_gates.
// Block r < hv is head r of a, block r >= hv is head r - hv of b.
__global__ void k_flash_gdn_ab_gates(const float* wa,const float* wb,const float* x,const float* A_log,
    const float* dt_bias,float* g_out,float* beta_out,int hv,int k,bool direct_coefficient) {
  const int r=blockIdx.x,tid=threadIdx.x,h=r<hv?r:r-hv;
  const float* w=(r<hv?wa:wb)+(size_t)h*k;
  float v=0;for(int c=tid;c<k;c+=256)v+=w[c]*x[c];
  __shared__ float sum[256];
  const float total=flash_tree256(v,sum);
  if(tid)return;
  if(r<hv) {
    float sp=total+dt_bias[h];
    sp=sp>20.f?sp:log1pf(expf(sp));
    g_out[h]=(direct_coefficient?A_log[h]:-expf(A_log[h]))*sp;
  } else {
    beta_out[h]=1.f/(1.f+expf(-total));
  }
}
// Hyperconnection read tail: k_sigmoid_inplace(G) and k_gr_combine.
__global__ void k_flash_gr_combine_sig(const float* __restrict__ G,const float* __restrict__ Rhat,
                                       float* __restrict__ x,int d,int branches) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=d)return;
  float acc=0.f;
  for(int b=0;b<branches;b++){const float g=1.f/(1.f+expf(-G[b*d+i]));acc+=g*Rhat[b*d+i];}
  x[i]=acc/branches;
}
// Hyperconnection write: the 4 x 4d F32 inject projection and k_gr_write.
// Every block repeats the four dots (164 KB of weights, cached after the first).
__global__ void k_flash_gr_write_inj(const float* __restrict__ winj,const float* __restrict__ Rhat,
                                     float* __restrict__ R,const float* __restrict__ y,int d) {
  const int tid=threadIdx.x,k=4*d;
  __shared__ float sum[4][256];
  float v[4]={0.f,0.f,0.f,0.f};
  for(int c=tid;c<k;c+=256) {
    const float xr=Rhat[c];
#pragma unroll
    for(int b=0;b<4;b++)v[b]+=winj[(size_t)b*k+c]*xr;
  }
  // flash_tree256 for the four dots at once: one barrier, then per-lane column sums.
  const int lane=tid&31;
#pragma unroll
  for(int b=0;b<4;b++)sum[b][tid]=v[b];
  __syncthreads();
  float w4[4];
#pragma unroll
  for(int b=0;b<4;b++) {
    float c[8];
#pragma unroll
    for(int w=0;w<8;w++)c[w]=sum[b][lane+32*w];
#pragma unroll
    for(int half=4;half;half>>=1)
#pragma unroll
      for(int w=0;w<half;w++)c[w]+=c[w+half];
    float t=c[0];
#pragma unroll
    for(int off=16;off;off>>=1)t+=__shfl_xor(t,off,32);
    w4[b]=t;
  }
  const int i=blockIdx.x*256+tid;
  if(i>=d)return;
#pragma unroll
  for(int b=0;b<4;b++){const float s=2.f/(1.f+expf(-w4[b]*0.25f));R[b*d+i]+=s*y[i];}
}
// Shared-expert gate (1 x k F32), k_axpy_sg and the copy to the layer output:
// y = acc + sigmoid(wsg . x) * ey. Every block repeats the one dot.
__global__ void k_flash_sg_out(const float* __restrict__ wsg,const float* __restrict__ x,
    const float* __restrict__ acc,const float* __restrict__ ey,float* __restrict__ y,int d,int k) {
  const int tid=threadIdx.x;
  float v=0;for(int c=tid;c<k;c+=256)v+=wsg[c]*x[c];
  __shared__ float sum[256];
  const float total=flash_tree256(v,sum);
  const int i=blockIdx.x*256+tid;
  if(i<d){const float s=1.f/(1.f+expf(-total));y[i]=acc[i]+s*ey[i];}
}
