// Actual head dispatch and direct/legacy GDN gates, without loading a model.
#define main gdec_real_main
#include "../upstream/src/gpu/gdec.cpp"
#undef main
#include "q6_oracle.hpp"
#include <random>
static void checked(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
template<class T> struct Buffer {
  T* p=nullptr;size_t n;
  explicit Buffer(size_t count):n(count){checked(hipMalloc(&p,n*sizeof(T)));checked(hipMemset(p,0,n*sizeof(T)));}
  explicit Buffer(const std::vector<T>& v):Buffer(v.size()){put(v);}
  ~Buffer(){if(p && hipFree(p)!=hipSuccess)std::abort();}
  void put(const std::vector<T>& v){if(v.size()!=n)throw std::runtime_error("buffer size");checked(hipMemcpy(p,v.data(),n*sizeof(T),hipMemcpyHostToDevice));}
  std::vector<T> get(){std::vector<T> v(n);checked(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost));return v;}
};
struct Compare {
  size_t n=0,bits=0;double maxabs=0,maxrel=0,sum2=0;
  void check(float value,float ref,bool exact=false) {
    if(!std::isfinite(value) || !std::isfinite(ref))throw std::runtime_error("nonfinite output");
    double e=std::abs(double(value)-ref);n++;bits+=memcmp(&value,&ref,4)!=0;
    maxabs=std::max(maxabs,e);maxrel=std::max(maxrel,e/std::max(1e-12,std::abs(double(ref))));sum2+=e*e;
    if(e>1e-5+1e-4*std::abs(double(ref)) || (exact && memcmp(&value,&ref,4)))throw std::runtime_error("operator budget/identity failure");
  }
  void print(const char* label){printf("%s n=%zu maxabs=%.9g maxrel=%.9g relative_floor=1e-12 rms=%.9g bit_diffs=%zu\n",label,n,maxabs,maxrel,sqrt(sum2/n),bits);}
};
__global__ void q6_values(const uint8_t* w,float* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=flash_quant::value(14,w,i);}
int main() {
  int dev=0;if(hipGetDeviceCount(&dev)!=hipSuccess || !dev){fprintf(stderr,"SKIP: HIP device unavailable\n");return 77;}
  try {
    constexpr int R=19,K=2560,S=K+32;std::mt19937 rng(614395);
    std::vector<block_q6_K> w(R*K/256);
    for(auto& b:w){for(auto& q:b.ql)q=uint8_t(rng());for(auto& q:b.qh)q=uint8_t(rng());for(auto& s:b.scales)s=int8_t(rng());b.d=0x0400;}
    Buffer<block_q6_K> dw(w);WView view{12,(uint8_t*)dw.p,R,K,0};
    std::vector<float> decoded(R*K);dequantize_row_q6_K(w.data(),decoded.data(),R*K);
    Buffer<float> values(R*K);q6_values<<<(R*K+255)/256,256>>>(view.data,values.p,R*K);
    Compare bytes,head,identity,kernels;auto v=values.get();for(size_t i=0;i<v.size();i++)bytes.check(v[i],decoded[i],true);
    for(int P:{1,2,8,17,65})for(bool bf:{false,true}) {
      std::vector<float> x(P*S);std::vector<uint16_t> xb(P*S);
      for(size_t i=0;i<x.size();i++){float z=float(int(rng()%101)-50)/64.f;xb[i]=gguf_map::f2bf(z);uint32_t u=uint32_t(xb[i])<<16;memcpy(&x[i],&u,4);}
      Buffer<float> dx(x),out(P*R),serial(P*R);Buffer<uint16_t> db(xb);
      for(int base=0;base<P;base+=8) {
        const int count=std::min(8,P-base);
        if(!gemv_multi_q4cp(view,bf?(const void*)(db.p+base*S):(const void*)(dx.p+base*S),out.p+base*R,count,S,bf))throw std::runtime_error("Q6_K multi-row dispatch missing");
      }
      for(int t=0;t<P;t++)gemv(view,dx.p+t*S,serial.p+t*R);
      checked(hipGetLastError());auto got=out.get(),single=serial.get();
      for(int t=0;t<P;t++)for(int r=0;r<R;r++) {
        double ref=0;for(int c=0;c<K;c++)ref+=double(decoded[r*K+c])*x[t*S+c];
        head.check(got[t*R+r],float(ref));identity.check(got[t*R+r],single[t*R+r],true);
      }
      Buffer<float> subset(7);gemv(view,dx.p,subset.p,3,7);auto sub=subset.get();
      for(int r=0;r<7;r++)identity.check(sub[r],single[3+r],true);
      // The block-decoding head kernel must reproduce the per-element kernel exactly.
      Buffer<float> scalar(P*R);
      for(int base=0;base<P;base+=8) {
        const int count=std::min(8,P-base);
        if(bf)k_flash_q6_gemv_scalar<true><<<(R+7)/8,256>>>(view.data,db.p+base*S,scalar.p+base*R,R,K,count,S);
        else k_flash_q6_gemv_scalar<false><<<(R+7)/8,256>>>(view.data,dx.p+base*S,scalar.p+base*R,R,K,count,S);
      }
      checked(hipGetLastError());auto old=scalar.get();
      for(size_t i=0;i<old.size();i++)kernels.check(got[i],old[i],true);
    }
    bytes.print("q6_device_vs_ggml");head.print("q6_head_vs_double");identity.print("q6_batch_serial_row_offset");
    kernels.print("q6_block_vs_scalar_kernel");
    Compare gates,gidentity;constexpr int H=48,P=65;
    std::vector<float> a(P*H),b(P*H),coef(H),alog(H),bias(H);
    const float boundary[]={-80.f,-20.f,-1.f,0.f,1.f,19.999f,20.f,20.001f,80.f};
    for(int h=0;h<H;h++){coef[h]=-.017f*(h+1);alog[h]=logf(-coef[h]);bias[h]=float(h-24)*.031f;}
    for(int i=0;i<P*H;i++){a[i]=boundary[i%9];b[i]=boundary[(i/9)%9];}
    Buffer<float> da(a),db(b),dc(coef),dl(alog),dt(bias),g(P*H),beta(P*H),sg(P*H),sb(P*H);
    for(bool direct:{false,true}) {
      k_gdn_gates<<<P,64>>>(da.p,db.p,direct?dc.p:dl.p,dt.p,g.p,beta.p,H,H,direct);
      for(int t=0;t<P;t++)k_gdn_gates<<<1,64>>>(da.p+t*H,db.p+t*H,direct?dc.p:dl.p,dt.p,sg.p+t*H,sb.p+t*H,H,H,direct);
      auto got=g.get(),be=beta.get(),one=sg.get(),ob=sb.get();
      for(int i=0;i<P*H;i++) {
        const double s=float(a[i]+bias[i%H]),scale=direct?double(coef[i%H]):-std::exp(double(alog[i%H]));
        gates.check(got[i],float(scale*(s>20?s:std::log1p(std::exp(s)))));
        gates.check(be[i],float(1/(1+std::exp(-double(b[i])))));
        gidentity.check(got[i],one[i],true);gidentity.check(be[i],ob[i],true);
      }
      // In-place gate dispatch is how the engine uses the kernel.
      da.put(a);db.put(b);k_gdn_gates<<<P,64>>>(da.p,db.p,direct?dc.p:dl.p,dt.p,da.p,db.p,H,H,direct);
      auto inplace=da.get(),ib=db.get();for(int i=0;i<P*H;i++){gidentity.check(inplace[i],got[i],true);gidentity.check(ib[i],be[i],true);}
      da.put(a);db.put(b);
    }
    gates.print("gdn_gates_vs_double");gidentity.print("gdn_gate_batch_serial_inplace");
    printf("PASS packed head and GDN coefficient operators; full-model gate still required\n");return 0;
  }catch(const std::exception& e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
