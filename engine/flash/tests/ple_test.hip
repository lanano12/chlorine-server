// Exercise the actual engine PLE kernels on frozen independent CPU fixtures.
#define main gdec_real_main
#include "../upstream/src/gpu/gdec.cpp"
#undef main
#include "../upstream/third_party/nlohmann/json.hpp"
#include <fstream>

using Json=nlohmann::json;
static void hip_check(hipError_t e) {if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
template<class T> struct DeviceArray {
  T* p=nullptr;size_t count;
  explicit DeviceArray(size_t n):count(n){hip_check(hipMalloc(&p,n*sizeof(T)));hip_check(hipMemset(p,0,n*sizeof(T)));}
  explicit DeviceArray(const std::vector<T>& v):DeviceArray(v.size()){put(v);}
  ~DeviceArray(){if(p && hipFree(p)!=hipSuccess){fprintf(stderr,"FAIL: hipFree\n");std::abort();}}
  void put(const std::vector<T>& v){if(v.size()!=count)throw std::runtime_error("shape mismatch");hip_check(hipMemcpy(p,v.data(),count*sizeof(T),hipMemcpyHostToDevice));}
  std::vector<T> get(){std::vector<T> v(count);hip_check(hipMemcpy(v.data(),p,count*sizeof(T),hipMemcpyDeviceToHost));return v;}
};
int main(int argc,char** argv) {
  int dev=0;if(hipGetDeviceCount(&dev)!=hipSuccess || !dev){fprintf(stderr,"SKIP: HIP device unavailable\n");return 77;}
  if(argc!=2){fprintf(stderr,"usage: ple_test fixture.json\n");return 1;}
  try {
    std::ifstream file(argv[1]);Json fixtures;file>>fixtures;
    int failures=0,cases=0;float maxabs=0;size_t bit_diffs=0;
    auto compare=[&](const std::vector<float>& got,const std::vector<float>& ref,bool exact){
      if(got.size()!=ref.size())throw std::runtime_error("reference shape mismatch");
      for(size_t i=0;i<got.size();i++) {
        float err=std::abs(got[i]-ref[i]);maxabs=std::max(maxabs,err);
        if(!std::isfinite(got[i]) || err>1e-5f+1e-4f*std::abs(ref[i]))++failures;
        if(exact && memcmp(&got[i],&ref[i],sizeof(float)))++bit_diffs;
      }
    };
    std::vector<float> first_output,first_ring;
    for(const auto& item:fixtures.at("cases")) {
      Json f=item;
      if(item.contains("conv_fixture")) {f=fixtures.at("conv_fixtures").at(item.at("conv_fixture").get<int>());f["request"]["chunks"]=item["chunks"];}
      const auto& q=f.at("request");const auto op=q.at("op").get<std::string>();
      if(op=="conv") {
        int n=q.at("width"),T=q.at("rows");
        auto x=q.at("input").get<std::vector<float>>(),w=q.at("weights").get<std::vector<float>>();
        DeviceArray<float> dx(x),dw(w),ring(chlorine_flash::ple_ring_slots*n),out(T*n),serial(T*n);
        DeviceArray<int> position(1);int base=0;
        for(int P:q.at("chunks").get<std::vector<int>>()) {
          // Use the graph's device base override too, with a deliberately wrong host base.
          position.put({base});
          k_ple_conv_b<<<(P*n+255)/256,256>>>(dx.p+base*n,dw.p,out.p+base*n,P,n,-777,ring.p,position.p);
          k_ple_ring_wr<<<chlorine_flash::ple_ring_slots,256>>>(dx.p+base*n,ring.p,P,n,-777,position.p);
          base+=P;
        }
        hip_check(hipGetLastError());auto y=out.get(),handoff=ring.get();
        std::vector<float> expected(T*n);
        if(chlorine_flash::ple_causal)expected=f.at("reference").at("output").get<std::vector<float>>();
        else for(int t=0;t<T;t++)for(int c=0;c<n;c++)expected[t*n+c]=chlorine_flash::ple_conv_row(x.data(),nullptr,w.data(),t,c,n,0);
        compare(y,expected,false);
        if(item.contains("conv_fixture")) {
          if(first_output.empty()){first_output=y;first_ring=handoff;}
          else{compare(y,first_output,true);compare(handoff,first_ring,true);}
        }
        // Decode writes current before reading history. Check every ring wrap.
        hip_check(hipMemset(ring.p,0,ring.count*sizeof(float)));
        for(int t=0;t<T;t++) {
          int rp=t%chlorine_flash::ple_ring_slots;position.put({rp});
          hip_check(hipMemcpy(ring.p+rp*n,dx.p+t*n,n*sizeof(float),hipMemcpyDeviceToDevice));
          k_ple_conv<<<(n+255)/256,256>>>(ring.p,position.p,dw.p,serial.p+t*n,n);
        }
        hip_check(hipGetLastError());compare(serial.get(),y,true);compare(ring.get(),handoff,true);
        // Checkpoint the actual device ring and replay a divergent continuation.
        DeviceArray<float> snapshot(handoff),branch(n);
        int rp=T%chlorine_flash::ple_ring_slots;position.put({rp});
        auto step=[&](int row){hip_check(hipMemcpy(ring.p+rp*n,dx.p+row*n,n*sizeof(float),hipMemcpyDeviceToDevice));
          k_ple_conv<<<(n+255)/256,256>>>(ring.p,position.p,dw.p,branch.p,n);return branch.get();};
        const auto continuation=step(0);
        step(1);hip_check(hipMemcpy(ring.p,snapshot.p,ring.count*sizeof(float),hipMemcpyDeviceToDevice));
        compare(step(0),continuation,true);
        ++cases;
      } else if(op=="gate") {
        auto dots=q.at("dots").get<std::vector<float>>();int T=dots.size();
        DeviceArray<float> kn(dots),qn(std::vector<float>(T,1.f)),v(std::vector<float>(T,1.f)),out(T),serial(T);
        k_ple_gate_b<<<dim3(1,T),32>>>(kn.p,qn.p,v.p,out.p,1,1);
        for(int t=0;t<T;t++)k_ple_gate<<<1,32>>>(kn.p+t,qn.p+t,v.p+t,serial.p+t,1,1);
        auto expected=f.at("reference").at("output").get<std::vector<float>>();
        if(!chlorine_flash::ple_causal)for(int t=0;t<T;t++)expected[t]=chlorine_flash::ple_gate(dots[t]);
        compare(out.get(),expected,false);compare(out.get(),serial.get(),true);++cases;
      }
    }
    printf("mode=%s cases=%d maxabs=%g chunk_decode_restore_bit_diffs=%zu failures=%d\n",chlorine_flash::ple_semantics,cases,maxabs,bit_diffs,failures);
    return failures || bit_diffs ? 1 : 0;
  }catch(const std::exception& e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
