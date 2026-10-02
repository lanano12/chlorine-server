// Chlorine Flash additions. Retains upstream routing and ordered reduction.
// IQ3_S prefill: decode only a 128x16 weight tile, never a full expert plane.
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
  const int* tokidx,const MoeTile* tiles,const int* ntiles,int max_tiles,__half* hid,int m,int k,hipStream_t st) {
  if(type==21) {
    if(k%256 || m%64) return false;
    k_flash_iq3_up<<<dim3((m+63)/64,max_tiles),256,0,st>>>(gate,up,x,tokidx,tiles,ntiles,hid,m,k);
    return true;
  }
  if(type==20 || type==23) {
    moe_lut_up(type,flash_iq_weights(gate,up,m),x,tokidx,tiles,ntiles,max_tiles,hid,m,k,st);
    return true;
  }
  return moe_gg_up(type,gate,up,x,tokidx,tiles,ntiles,max_tiles,hid,m,k,st);
}
static bool flash_moe_down(int type,const uint8_t* down,const __half* hid,
  const MoeTile* tiles,const int* ntiles,int max_tiles,float* pairs,int m,int k,hipStream_t st) {
  if(type==20) {
    moe_lut_down(type,flash_iq_weights(down,nullptr,m),hid,tiles,ntiles,max_tiles,pairs,m,k,st);
    return true;
  }
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
// F32 matrices remain F32 in the GGUF mapping (routers, HC injection, GDN gates).
__global__ void k_flash_f32_gemv(const float* w,const float* x,float* out,int n,int k,size_t xs) {
  int r=blockIdx.x,t=blockIdx.y,tid=threadIdx.x;
  float v=0;for(int c=tid;c<k;c+=256)v+=w[(size_t)r*k+c]*x[(size_t)t*xs+c];
  __shared__ float sum[256];sum[tid]=v;__syncthreads();
  for(int s=128;s;s>>=1){if(tid<s)sum[tid]+=sum[tid+s];__syncthreads();}
  if(tid==0)out[(size_t)t*n+r]=sum[0];
}
