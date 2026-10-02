// Compare actual nonlinear GEMV dispatch at P=1 and verification sizes.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "../upstream/src/gguf.h"
struct MoeTile {int expert,first,count;};
#include "../upstream/src/gpu/parts/26_kernels_moe_gguf.inc"
#include "../upstream/src/gpu/parts/27_kernels_moe_lut.inc"
#include "../upstream/src/gpu/parts/28_chlorine_iq.inc"
#define H(call) do{auto e=(call);if(e!=hipSuccess){fprintf(stderr,"%s\n",hipGetErrorString(e));return 1;}}while(0)
template<class T> T* copy(const std::vector<T>& h) {
 T* p=nullptr;if(hipMalloc(&p,h.size()*sizeof(T))!=hipSuccess)throw std::runtime_error("hipMalloc");
 if(hipMemcpy(p,h.data(),h.size()*sizeof(T),hipMemcpyHostToDevice)!=hipSuccess)throw std::runtime_error("hipMemcpy");return p;
}
__global__ void decode_values(int type,const uint8_t* bytes,float* out,int count) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)out[i]=flash_quant::value(type,bytes,i);
}
int main() {
 int devices=0;if(hipGetDeviceCount(&devices)!=hipSuccess || !devices){fprintf(stderr,"SKIP: HIP device unavailable\n");return 77;}
 constexpr int E=10,D=256,M=64,P=8;
 std::mt19937 rng(395);
 auto weights=[&](int type,int rows,int cols){
  uint32_t be,bb;gguf::block_info(type,be,bb);std::vector<uint8_t> w((size_t)rows*gguf::row_bytes(type,cols));
  for(auto& b:w)b=uint8_t(rng());for(size_t i=0;i<w.size();i+=bb){uint16_t h=0x0800;memcpy(w.data()+i,&h,2);}return w;
 };
 int failures=0;
 for(int gu:{21,20,23})for(int dn:{20,8}) {
  auto wg=weights(gu,E*M,D),wu=weights(gu,E*M,D),wd=weights(dn,E*D,M);
  std::vector<float> x(P*D),top(P*16,0),zeros(P*10*M,0),outputs(P*D,0);
  std::vector<int> ids(P*16,0);for(auto& v:x)v=float(int(rng()%101)-50)/100.f;
  for(int t=0;t<P;++t)for(int s=0;s<10;++s){ids[t*16+s]=(s+t)%10;top[t*16+s]=.1f;}
  auto *dg=copy(wg),*du=copy(wu),*dd=copy(wd);auto *dx=copy(x),*tw=copy(top),*hid=copy(zeros),*out=copy(outputs);auto* di=copy(ids);
  if(!flash_moe_gemv(gu,dn,dg,du,dd,dx,di,tw,hid,out,P,D,M,16,0))return 1;
  H(hipGetLastError());H(hipDeviceSynchronize());
  std::vector<float> bat(P*D),serial(P*D),hh(P*10*M);
  H(hipMemcpy(bat.data(),out,bat.size()*4,hipMemcpyDeviceToHost));
  H(hipMemcpy(hh.data(),hid,hh.size()*4,hipMemcpyDeviceToHost));
  for(int t=0;t<P;++t) {
   if(!flash_moe_gemv(gu,dn,dg,du,dd,dx+t*D,di+t*16,tw+t*16,hid,out+t*D,1,D,M,16,0))return 1;
  }
  H(hipMemcpy(serial.data(),out,serial.size()*4,hipMemcpyDeviceToHost));
  size_t diff=0;float maxabs=0;
  std::vector<float> g(D),u(D),h(M),down(M),ref(D);
  for(int t=0;t<P;++t) {
   std::fill(ref.begin(),ref.end(),0.f);
   for(int s=0;s<10;++s){int e=ids[t*16+s];
    for(int r=0;r<M;++r){
     gguf::dequant_row(gu,wg.data()+((size_t)e*M+r)*gguf::row_bytes(gu,D),g.data(),D);
     gguf::dequant_row(gu,wu.data()+((size_t)e*M+r)*gguf::row_bytes(gu,D),u.data(),D);
     double ga=0,up=0;for(int c=0;c<D;++c){ga+=double(g[c])*x[t*D+c];up+=double(u[c])*x[t*D+c];}
     h[r]=float(up*ga/(1+std::exp(-ga)));
    }
    for(int r=0;r<D;++r){
     gguf::dequant_row(dn,wd.data()+((size_t)e*D+r)*gguf::row_bytes(dn,M),down.data(),M);
     double y=0;for(int c=0;c<M;++c)y+=double(down[c])*h[c];ref[r]+=.1f*float(y);
    }
   }
   for(int r=0;r<D;++r){size_t at=t*D+r;diff+=memcmp(&bat[at],&serial[at],4)!=0;
    maxabs=std::max(maxabs,std::abs(bat[at]-ref[r]));
    if(!std::isfinite(bat[at]) || std::abs(bat[at]-ref[r])>1e-5f+1e-4f*std::abs(ref[r]))++failures;
   }
  }
  failures+=diff!=0;
  // Host and device packed decoders must return exactly the same FP32 bits.
  decode_values<<<1,256>>>(gu,dg,out,D);
  H(hipMemcpy(serial.data(),out,D*4,hipMemcpyDeviceToHost));
  for(int c=0;c<D;++c){float expected=flash_quant::value(gu,wg.data(),c);if(memcmp(&serial[c],&expected,4))++failures;}
  printf("gu=%d down=%d P=8 vs P=1 diffs=%zu reference_maxabs=%g\n",gu,dn,diff,maxabs);
  for(void* p:{(void*)dg,(void*)du,(void*)dd,(void*)dx,(void*)tw,(void*)hid,(void*)out,(void*)di})H(hipFree(p));
 }
 printf("%s failures=%d\n",failures?"FAIL":"PASS",failures);return failures?1:0;
}
