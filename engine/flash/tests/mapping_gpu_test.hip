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
    {
      // Router top-k: the shuffle kernel against the tree kernel, with ties.
      constexpr int TP=4096,E=512,TK=10;std::vector<float> lg(size_t(TP)*E);std::normal_distribution<float> nd(0.f,3.f);
      for(int r=0;r<TP;r++)for(int e=0;e<E;e++) {
        float x=nd(rng);
        switch(r%8){case 1:x=std::round(x);break;case 2:x=0.f;break;case 3:x=(e%7==0)?5.f:-40.f;break;
          case 4:x=std::round(x*2)/2;break;case 5:x=(e%64==r%64)?x:-1e30f;break;case 6:x=std::round(x/4)*4;break;default:break;}
        lg[size_t(r)*E+e]=x;
      }
      Buffer<float> dl(lg),w0(size_t(TP)*16),w1(size_t(TP)*16);Buffer<int> i0(size_t(TP)*16),i1(size_t(TP)*16);
      k_router_topk_tree<<<TP,512>>>(dl.p,i0.p,w0.p,E,TK);k_router_topk<<<TP,512>>>(dl.p,i1.p,w1.p,E,TK);
      checked(hipDeviceSynchronize());checked(hipGetLastError());
      auto a0=i0.get(),a1=i1.get();auto b0=w0.get(),b1=w1.get();size_t idd=0,wd=0;
      for(int r=0;r<TP;r++)for(int j=0;j<TK;j++){idd+=a0[r*16+j]!=a1[r*16+j];wd+=memcmp(&b0[r*16+j],&b1[r*16+j],4)!=0;}
      printf("router_topk_shuffle_vs_tree rows=%d id_diffs=%zu weight_bit_diffs=%zu\n",TP,idd,wd);
      if(idd||wd)throw std::runtime_error("router top-k identity failure");
    }
    {
      // Fused single-token decode kernels against the launches they replace.
      constexpr int D=2560,K4=4*D,HV=48,MID=640,T=320;std::normal_distribution<float> nd(0.f,1.f);
      auto randv=[&](size_t n,float sc){std::vector<float> v(n);for(auto& x:v)x=nd(rng)*sc;return v;};
      auto randq=[&](size_t n){std::vector<int8_t> v(n);for(auto& x:v)x=int8_t(rng());return v;};
      auto rands=[&](size_t n){std::vector<uint16_t> v(n);for(auto& x:v)x=uint16_t(0x2800+(rng()&0x3FF));return v;};
      size_t diffs=0,total=0;
      auto cmp=[&](const std::vector<float>& a,const std::vector<float>& b) {
        if(a.size()!=b.size())throw std::runtime_error("fused compare size");
        for(size_t i=0;i<a.size();i++){if(!std::isfinite(a[i]))throw std::runtime_error("fused compare nonfinite");diffs+=memcmp(&a[i],&b[i],4)!=0;}
        total+=a.size();
      };
      {
        // One-barrier F32 row dot against the per-level barrier tree, several rows and tokens.
        constexpr int FR=37,FT=3,FK=2560,FS=FK+64;
        Buffer<float> fw(randv(size_t(FR)*FK,.05f)),fx(randv(size_t(FT)*FS,1.f)),f0(size_t(FT)*FR),f1(size_t(FT)*FR);
        k_flash_f32_gemv_tree<<<dim3(FR,FT),256>>>(fw.p,fx.p,f0.p,FR,FK,FS);
        k_flash_f32_gemv<<<dim3(FR,FT),256>>>(fw.p,fx.p,f1.p,FR,FK,FS);
        checked(hipDeviceSynchronize());cmp(f0.get(),f1.get());
      }
      Buffer<float> G(randv(K4,2.f)),G2(K4),Rh(randv(K4,1.f)),x0(D),x1(D);
      checked(hipMemcpy(G2.p,G.p,K4*4,hipMemcpyDeviceToDevice));
      k_sigmoid_inplace<<<(K4+255)/256,256>>>(G2.p,K4);k_gr_combine<<<(D+255)/256,256>>>(G2.p,Rh.p,x0.p,D,4);
      k_flash_gr_combine_sig<<<(D+255)/256,256>>>(G.p,Rh.p,x1.p,D,4);
      checked(hipDeviceSynchronize());cmp(x0.get(),x1.get());
      Buffer<float> winj(randv(size_t(4)*K4,.02f)),R0(randv(K4,1.f)),R1(K4),yv(randv(D,1.f)),w4(16);
      checked(hipMemcpy(R1.p,R0.p,K4*4,hipMemcpyDeviceToDevice));
      k_flash_f32_gemv_tree<<<dim3(4,1),256>>>(winj.p,Rh.p,w4.p,4,K4,K4);k_gr_write<<<(D+255)/256,256>>>(w4.p,R0.p,yv.p,D,4);
      k_flash_gr_write_inj<<<(D+255)/256,256>>>(winj.p,Rh.p,R1.p,yv.p,D);
      checked(hipDeviceSynchronize());cmp(R0.get(),R1.get());
      Buffer<float> wsg(randv(D,.05f)),xin(randv(D,1.f)),acc0(randv(D,1.f)),acc1(D),ey(randv(D,1.f)),sgv(4),yo(D);
      checked(hipMemcpy(acc1.p,acc0.p,D*4,hipMemcpyDeviceToDevice));
      k_flash_f32_gemv_tree<<<dim3(1,1),256>>>(wsg.p,xin.p,sgv.p,1,D,D);k_axpy_sg<<<(D+255)/256,256>>>(acc1.p,ey.p,sgv.p,D,D);
      k_flash_sg_out<<<(D+255)/256,256>>>(wsg.p,xin.p,acc0.p,ey.p,yo.p,D,D);
      checked(hipDeviceSynchronize());cmp(acc1.get(),yo.get());
      Buffer<float> wa(randv(size_t(HV)*D,.05f)),wb(randv(size_t(HV)*D,.05f)),al(randv(HV,.5f)),dtb(randv(HV,6.f)),ga(HV),gb(HV),g1(HV),b1(HV);
      for(bool direct:{false,true}) {
        k_flash_f32_gemv_tree<<<dim3(HV,1),256>>>(wa.p,xin.p,ga.p,HV,D,D);k_flash_f32_gemv_tree<<<dim3(HV,1),256>>>(wb.p,xin.p,gb.p,HV,D,D);
        k_gdn_gates<<<1,64>>>(ga.p,gb.p,al.p,dtb.p,ga.p,gb.p,HV,HV,direct);
        k_flash_gdn_ab_gates<<<2*HV,256>>>(wa.p,wb.p,xin.p,al.p,dtb.p,g1.p,b1.p,HV,D,direct);
        checked(hipDeviceSynchronize());cmp(ga.get(),g1.get());cmp(gb.get(),b1.get());
      }
      Buffer<int8_t> qd(randq(size_t(T)*K4));Buffer<uint16_t> sd(rands(size_t(T)*K4/32));Buffer<float> t0(T),t1(T);
      k_q8g32_gemv_lpr<32><<<(T+7)/8,256>>>(qd.p,(const __half*)sd.p,Rh.p,t0.p,T,K4);k_silu_scale<<<(T+255)/256,256>>>(t0.p,.25f,T);
      k_q8g32_gemv_lpr<32,true><<<(T+7)/8,256>>>(qd.p,(const __half*)sd.p,Rh.p,t1.p,T,K4);
      checked(hipDeviceSynchronize());cmp(t0.get(),t1.get());
      Buffer<int8_t> qg(randq(size_t(MID)*D)),qu(randq(size_t(MID)*D));Buffer<uint16_t> s1(rands(size_t(MID)*D/32)),s2(rands(size_t(MID)*D/32));
      Buffer<float> guv(2*MID),h0(MID),h1(MID);
      k_q8g32_gemv_lpr<32><<<(MID+7)/8,256>>>(qg.p,(const __half*)s1.p,xin.p,guv.p,MID,D);
      k_q8g32_gemv_lpr<32><<<(MID+7)/8,256>>>(qu.p,(const __half*)s2.p,xin.p,guv.p+MID,MID,D);
      k_silu_mul<<<(MID+255)/256,256>>>(guv.p,guv.p+MID,h0.p,MID);
      k_flash_q8_gate_up_silu<<<(MID+3)/4,256>>>(qg.p,(const __half*)s1.p,qu.p,(const __half*)s2.p,xin.p,h1.p,MID,D);
      checked(hipDeviceSynchronize());checked(hipGetLastError());cmp(h0.get(),h1.get());
      printf("decode_fused_vs_separate n=%zu bit_diffs=%zu\n",total,diffs);
      if(diffs)throw std::runtime_error("fused decode kernel identity failure");
    }
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
