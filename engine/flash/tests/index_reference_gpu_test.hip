// Check actual indexer operators against the independent CPU oracle.
#define main gdec_original_main
#include "../upstream/src/gpu/gdec.cpp"
#undef main
#include "index_reference.hpp"
#include <iostream>
namespace R=flash_index_reference;
#include "reference_gpu.hpp"
int main(){int devices=0;if(hipGetDeviceCount(&devices)!=hipSuccess||!devices){fprintf(stderr,"SKIP: HIP device unavailable\n");return 77;}
 try {
  constexpr int length=4099,nb=length/4,seed=395;constexpr double theta=1e7;R::Stream ref;ref.theta=theta;ref.append(seed,0,length);
  auto projection=R::project(seed,0,length);RefBuffer<float> proj(projection);
  auto w=R::offset();RefBuffer<float> weight(std::vector<float>(w.begin(),w.end()));
  RefError norms,keys,ring_error,chunk_identity,scores_error;size_t ids_checked=0;
  std::vector<float> baseline;
  k_rope_tab_init<<<1,64>>>(theta,64);
  RefBuffer<float2> cs(length*32);
  k_rope_cs<<<(length*32+255)/256,256>>>(cs.p,0,length);
  for(bool table_rope:{false,true})for(bool paged:{false,true})for(int chunk:{1,3,65,256,2048,4099}) {
    const int pages=(nb+63)/64;std::vector<int> table(pages);for(int i=0;i<pages;i++)table[i]=pages-i;
    RefBuffer<int> ptab(table);RefBuffer<float> store((paged?(pages+1)*64:nb)*128),ring(512),gather(nb*128);
    for(int base=0;base<length;) {
      const int P=std::min(chunk,length-base),complete=(base+P)/4-base/4;
      if(complete)k_index_pool<<<complete,128>>>(proj.p+size_t(base)*640,weight.p,store.p,1e-6,theta,base,ring.p,table_rope?cs.p+size_t(base)*32:nullptr,paged?ptab.p:nullptr);
      k_index_ring<<<std::min(P,4),128>>>(proj.p+size_t(base)*640,ring.p,P,base);
      base+=P;
    }
    CK(hipGetLastError());
    if(paged)k_ik_gather<<<nb,128>>>(store.p,ptab.p,gather.p,nb);
    auto got=paged?gather.get():store.get();
    for(int b=0;b<nb;b++)for(int d=0;d<128;d++)keys.check(got[b*128+d],ref.keys[b][d]);
    auto r=ring.get();for(int s=0;s<4;s++)for(int d=0;d<128;d++)ring_error.check(r[s*128+d],ref.ring[s][d],true);
    if(baseline.empty())baseline=got;else for(size_t i=0;i<got.size();i++)chunk_identity.check(got[i],baseline[i],true);
    if(paged){auto physical=store.get();for(int d=0;d<64*128;d++)if(physical[d]!=0)throw std::runtime_error("page guard overwritten");}
  }
  RefBuffer<float> logical(baseline),query(512),score(4*nb);RefBuffer<int> position(1),selected(512);
  for(bool table_rope:{false,true})for(int p:{0,1,2,3,255,256,2046,2047,2048,2049,2050,2051,2052,2053,2054,2055,2056,4098}) {
    position.put({p});k_index_q<<<dim3(1,4),128>>>(proj.p+size_t(p)*640,weight.p,query.p,1,1e-6,theta,position.p,p,table_rope?cs.p+size_t(p)*32:nullptr);
    auto q=query.get();auto expected_q=R::queries(seed,p,theta);for(int h=0;h<4;h++)for(int d=0;d<128;d++)norms.check(q[h*128+d],expected_q[h][d]);
    const int n=(p+1)/4;
    if(p>=2051) {
      k_index_scores<<<(nb+3)/4,128>>>(query.p,logical.p,score.p,nb,position.p);
      auto got=score.get();
      for(int b=0;b<n;b++)for(int h=0;h<4;h++){double dot=0;for(int d=0;d<128;d++)dot+=double(expected_q[h][d])*ref.keys[b][d];scores_error.check(got[h*nb+b],float(dot));}
    }
    index_select(score.p,nb,selected.p,0,1,position.p);auto ids=selected.get();auto expected=ref.selected(seed,p);
    for(int i=0;i<512;i++){int want=i<int(expected.size())?expected[i]:-1;if(ids[i]!=want)throw std::runtime_error("CPU/device selected block mismatch");ids_checked++;}
  }
  // Independent sort validates every fast selector, exact ties, causal clamp,
  // uninitialized future scores, and the 8192-block implementation boundary.
  for(int n:{0,1,511,512,513,8192,8193,10001})for(int mode:{0,1,2}) {
    const int stride=n+17;std::vector<float> values(stride,1e20f);std::vector<double> oracle(n);
    for(int i=0;i<n;i++)values[i]=float(mode==0?0:mode==1?i%7:(i*107%1009))/64.f;
    for(int i=0;i<n;i++)oracle[i]=values[i];auto expected=R::select(oracle,n);
    RefBuffer<float> ds(values);const int first=n*4-1;
    for(int variant=0;variant<3;variant++) {
      if(variant==0&&n>8192)continue;
      if(variant==0)k_index_select_rs<true><<<1,256>>>(ds.p,stride,selected.p,first);
      else if(variant==1)k_index_select_stream<true><<<1,256>>>(ds.p,stride,selected.p,first);
      else k_index_select_2p<<<1,256>>>(ds.p,stride,selected.p,first,mode==2?0:ISEL_CAP);
      auto got=selected.get();for(int i=0;i<512;i++){int want=i<int(expected.size())?expected[i]:-1;if(got[i]!=want)throw std::runtime_error("GPU selector tie/causal mismatch");ids_checked++;}
    }
  }
  norms.print("index_query_vs_independent");keys.print("index_pool_vs_independent");scores_error.print("index_dots_vs_double");
  ring_error.print("index_ring_exact");chunk_identity.print("index_chunk_and_page_exact");
  printf("PASS independent index GPU operators selected_ids=%zu; no full-model qualification\n",ids_checked);return 0;
 }catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
