// Actual FP32 prefill/decode attention versus independent global-softmax oracle.
#define main gdec_original_main
#include "../upstream/src/gpu/gdec.cpp"
#undef main
#include "attention_reference.hpp"
#include "reference_gpu.hpp"
namespace A=flash_attention_reference;
int main() {
  int devices=0;if(hipGetDeviceCount(&devices)!=hipSuccess||!devices){fprintf(stderr,"SKIP: HIP device unavailable\n");return 77;}
  try {
    RefError prefill_error,decode_error,layout_identity;size_t cases=0;
    for(auto shape:std::vector<std::pair<int,int>>{{0,1},{0,33},{5,4},{17,1},{2047,10},{2051,5},{8191,1}})
    for(float scale:{1.f,8.f}) {
      const int base=shape.first,P=shape.second,N=base+P,seed=395;
      auto queries=A::query(seed,base,P,scale),keys=A::kv(seed,N+8,false,scale),values=A::kv(seed,N+8,true);
      auto expected=A::mix(queries,keys,values,seed,base,P);
      // Inaccessible tail and unused physical/guard pages contain NaNs.
      for(size_t i=size_t(N)*512;i<keys.size();i++)keys[i]=values[i]=std::numeric_limits<float>::quiet_NaN();
      std::vector<int> selected(size_t(P)*512,-1);
      for(int r=0;r<P;r++){auto ids=A::blocks(seed,base+r);std::copy(ids.begin(),ids.end(),selected.begin()+size_t(r)*512);}
      RefBuffer<float> q(queries),out(queries.size()),one(6144);
      RefBuffer<int> select(selected),pos(1);
      RefBuffer<float> pacc(2*QSA_DEC_NSPLIT*12*256),pml(2*QSA_DEC_NSPLIT*12*2);
      std::vector<float> baseline;
      for(bool paged:{false,true}) {
        const int pages=(N+8+KV_PAGE-1)/KV_PAGE;
        std::vector<int> table(pages);for(int i=0;i<pages;i++)table[i]=pages-i;
        RefBuffer<int> pt(table);
        auto pack=[&](const std::vector<float>& input){
          if(!paged)return input;
          std::vector<float> result(size_t(pages+1)*KV_PAGE*512,std::numeric_limits<float>::quiet_NaN());
          for(int r=0;r<N;r++)std::copy_n(input.data()+size_t(r)*512,512,
            result.data()+size_t(table[r/KV_PAGE]*KV_PAGE+r%KV_PAGE)*512);
          return result;
        };
        RefBuffer<float> k(pack(keys)),v(pack(values));
        for(bool shared:{false,true}) {
          int dense=std::min(P,std::max(0,2051-base));
          if(dense)k_qsa_flash<false,false,float><<<dim3((dense+15)/16,24),128>>>(
            q.p,k.p,v.p,out.p,dense,256,2,24,nullptr,0,nullptr,nullptr,nullptr,base,paged?pt.p:nullptr);
          int first=std::max(base,2051),count=N-first;
          if(count>0) {
            const int* sel=select.p+size_t(first-base)*512;
            if(shared)k_qsa_flash<true,true,float><<<dim3(count,2),128>>>(
              q.p,k.p,v.p,out.p,P,256,2,24,sel,first,nullptr,nullptr,nullptr,base,paged?pt.p:nullptr);
            else k_qsa_flash<true,false,float><<<dim3(count,2),128>>>(
              q.p,k.p,v.p,out.p,P,256,2,24,sel,first,nullptr,nullptr,nullptr,base,paged?pt.p:nullptr);
          }
          CK(hipGetLastError());auto got=out.get();
          for(size_t i=0;i<got.size();i++)prefill_error.check(got[i],expected[i]);
          if(baseline.empty())baseline=got;else for(size_t i=0;i<got.size();i++)layout_identity.check(got[i],baseline[i],true);
        }
        for(int r=0;r<P;r++) {
          pos.put({base+r});
          k_qsa_flash<true,false,float,true><<<dim3(QSA_DEC_NSPLIT,2),128>>>(
            q.p+size_t(r)*6144,k.p,v.p,one.p,1,256,2,24,select.p+size_t(r)*512,0,pos.p,pacc.p,pml.p,0,paged?pt.p:nullptr);
          k_qsa_flash_combine<<<dim3(12,2),64>>>(pml.p,pacc.p,one.p,12,256,pos.p);
          CK(hipGetLastError());auto got=one.get();
          for(size_t i=0;i<got.size();i++)decode_error.check(got[i],expected[size_t(r)*6144+i]);
        }
      }
      cases++;
    }
    prefill_error.print("qsa_prefill_vs_independent_global_softmax");
    decode_error.print("qsa_split_decode_vs_independent_global_softmax");
    layout_identity.print("qsa_paging_and_shared_V_exact");
    printf("PASS independent FP32 QSA GPU check: %zu shapes/scales; no WMMA, no complete-model qualification\n",cases);
    return 0;
  }catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
