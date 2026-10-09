// Union attention (GDEC_QSA_UNION on the bf16 KV cache with the transposed V
// cache) at the edges the host partition decides: short unaligned
// continuations such as base=2053, P=2, and the native-context sentinel block
// (65535 = 0xffff, the union list's padding id). The launches are the ones
// qsa_flash_b issues; the inputs are synthetic. Each output row is checked
// against the independent double oracle (fed the bf16-rounded inputs the
// kernels see) and against the per-query k_qsa_wmma<true> kernel, and guard
// rows around the batch detect any write outside it. Rows the kernels must
// never read hold NaN, so a stray gather shows up as a nonfinite output.
#define main gdec_original_main
#include "../upstream/src/gpu/gdec.cpp"
#undef main
#include "../qsa_union_partition.hpp"
#include "attention_reference.hpp"
#include "reference_gpu.hpp"
namespace A=flash_attention_reference;
namespace I=flash_index_reference;

static float bf16_round(float x){const uint32_t u=uint32_t(gguf_map::f2bf(x))<<16;float r;memcpy(&r,&u,4);return r;}
struct Tolerance {
  double tol;size_t n=0,bits=0;double maxabs=0;
  explicit Tolerance(double t):tol(t){}
  void check(float a,float b){
    if(!std::isfinite(a)||!std::isfinite(b))throw std::runtime_error("nonfinite attention output");
    const double e=std::abs(double(a)-b);n++;bits+=memcmp(&a,&b,4)!=0;maxabs=std::max(maxabs,e);
    if(e>tol)throw std::runtime_error("attention output outside tolerance");
  }
  void print(const char* tag){printf("%s n=%zu maxabs=%.3g bit_diffs=%zu tol=%.1g\n",tag,n,maxabs,bits,tol);}
};
static bool is_nan(float v){return std::isnan(v);}

int main() {
  int devices=0;if(hipGetDeviceCount(&devices)!=hipSuccess||!devices){fprintf(stderr,"SKIP: HIP device unavailable\n");return 77;}
  try {
    constexpr int hq=24,hkv=2,dh=256,qrow=hq*dh,guard=8,seed=395;
    // bf16 P and WMMA accumulation against a double oracle on the same bf16 inputs:
    // measured 1.2e-4 on these shapes, so 1e-3 is a bound, not a budget.
    Tolerance oracle_union(1e-3),oracle_perquery(1e-3),union_vs_perquery(1e-3);
    RefError edge_identity;size_t cases=0,guard_rows=0,group_rows=0,edge_rows=0;
    const std::vector<std::pair<int,int>> shapes={
      {2053,2},{2051,1},{2051,2},{2051,3},{2052,4},{2051,9},{2053,7},{2049,6},{8190,5},{8192,16},
      {262120,24},{262136,8},{262138,6},{262140,4},{262141,3},{262143,1}};
    for(auto shape:shapes) {
      const int base=shape.first,P=shape.second,N=base+P,N4=(N+3)&~3;
      const int first=std::max(base,2051),dense=first-base;  // production keeps dense rows on k_qsa_flash
      if(first>=N)throw std::runtime_error("shape has no sparse rows");
      const auto ranges=chlorine_flash::qsa_union_partition(first,N);
      // Selections for every batch row (dense rows: none) and the rows the kernels may read.
      std::vector<int> selected(size_t(P)*512,-1);
      std::vector<char> visible(N4,0);
      for(int t=0;t<4;t++)visible[t]=1;  // block 0 stands in for padding ids and masked keys
      // Dense rows of the batch are not launched here (production keeps them on
      // k_qsa_flash), but the oracle still evaluates them over every earlier token.
      for(int t=0;t<std::min(N,2051)&&dense>0;t++)visible[t]=1;
      for(int r=dense;r<P;r++) {
        const auto ids=A::blocks(seed,base+r);std::copy(ids.begin(),ids.end(),selected.begin()+size_t(r)*512);
        for(int t:I::tokens(ids,base+r))visible[t]=1;
        const int own=(base+r)&~3;for(int s=0;s<4&&own+s<N4;s++)visible[own+s]=1;
      }
      // K and V: visible rows carry the oracle's values rounded to bf16, rows beyond
      // N in the trailing block are zero as production stores them, everything else NaN.
      std::vector<uint16_t> kc(size_t(N4)*hkv*dh,0x7FC0),vc(kc.size(),0x7FC0);
      std::vector<float> kf(kc.size(),std::numeric_limits<float>::quiet_NaN()),vf(kf);
      for(int t=0;t<N4;t++) {
        if(t>=N){for(int i=0;i<hkv*dh;i++){kc[size_t(t)*hkv*dh+i]=0;vc[size_t(t)*hkv*dh+i]=0;}continue;}
        if(!visible[t])continue;
        for(int h=0;h<hkv;h++)for(int d=0;d<dh;d++) {
          const size_t at=(size_t(t)*hkv+h)*dh+d;
          kc[at]=gguf_map::f2bf(I::input(seed+31,t,h*dh+d));vc[at]=gguf_map::f2bf(I::input(seed+97,t,h*dh+d));
          kf[at]=bf16_round(I::input(seed+31,t,h*dh+d));vf[at]=bf16_round(I::input(seed+97,t,h*dh+d));
        }
      }
      auto queries=A::query(seed,base,P);
      std::vector<float> q_bf(queries);for(auto& v:q_bf)v=bf16_round(v);
      const auto expected=A::mix(q_bf,kf,vf,seed,base,P);
      RefBuffer<float> q(queries);RefBuffer<uint16_t> k(kc),v(vc),vt(kc.size());RefBuffer<int> select(selected);
      k_bf16_rows_to_bt<<<(unsigned)((size_t(N4/4)*512+255)/256),256>>>(v.p,vt.p,0,N);
      CK(hipGetLastError());
      const int ngroups=std::max(1,ranges.groups.count/4);
      RefBuffer<uint16_t> ublk(size_t(ngroups)*QSA_UCAP),umask(size_t(ngroups)*QSA_UCAP);RefBuffer<int> ucount(ngroups);
      // Guard rows before and after the batch start as NaN and must stay NaN.
      const size_t outn=size_t(P+2*guard)*qrow;
      RefBuffer<float> out_part(outn),out_full(outn);
      CK(hipMemset(out_part.p,0xFF,outn*4));CK(hipMemset(out_full.p,0xFF,outn*4));
      float* part=out_part.p+size_t(guard)*qrow;float* full=out_full.p+size_t(guard)*qrow;
      const int* sel=select.p+size_t(first-base)*512;
      // The production sequence: union groups, then the per-query kernel on head and tail.
      if(ranges.groups.count) {
        const int ng=ranges.groups.count/4;
        k_qsa_union_merge<<<ng,128>>>(select.p,ranges.groups.first,ublk.p,umask.p,ucount.p,QSA_UCAP,base);
        k_qsa_wmma_u<<<dim3(ng,2),256>>>(q.p,k.p,vt.p,part,dh,hkv,hq,ublk.p,umask.p,ucount.p,QSA_UCAP,ranges.groups.first,base);
      }
      for(const auto edge:{ranges.head,ranges.tail})
        if(edge.count)k_qsa_wmma<true><<<dim3(edge.count,2),256>>>(q.p,k.p,v.p,vt.p,part,P,dh,hkv,hq,sel+size_t(edge.first-first)*512,edge.first,base);
      // The same rows on the per-query kernel alone.
      k_qsa_wmma<true><<<dim3(N-first,2),256>>>(q.p,k.p,v.p,vt.p,full,P,dh,hkv,hq,sel,first,base);
      CK(hipGetLastError());
      const auto got=out_part.get(),alone=out_full.get();
      if(ranges.groups.count) {
        // Every union list is 4-padded; 0xFFFF appears only as trailing padding,
        // never as a real block id (the sentinel block stays on the per-query kernel).
        const std::vector<int> counts=ucount.get();
        const std::vector<uint16_t> blocks=ublk.get();
        for(int g=0;g<ranges.groups.count/4;g++) {
          if(counts[g]<=0||counts[g]&3||counts[g]>QSA_UCAP)throw std::runtime_error("union count outside its capacity");
          const uint16_t* list=&blocks[size_t(g)*QSA_UCAP];int pad=-1;
          for(int i=0;i<counts[g];i++) {
            if(list[i]==0xFFFF){if(pad<0)pad=i;}
            else if(pad>=0)throw std::runtime_error("real block id after the padding of a union list");
          }
        }
      }
      for(size_t r=0;r<size_t(P+2*guard);r++) {
        const bool written=r>=size_t(guard+dense)&&r<size_t(guard+P);
        for(size_t i=0;i<size_t(qrow);i++) {
          const float a=got[r*qrow+i],b=alone[r*qrow+i];
          if(!written){if(!is_nan(a)||!is_nan(b))throw std::runtime_error("attention wrote outside its batch rows");continue;}
          const int row=int(r)-guard,token=base+row;const float ref=expected[size_t(row)*qrow+i];
          const bool in_group=token>=ranges.groups.first&&token<ranges.groups.first+ranges.groups.count;
          oracle_perquery.check(b,ref);
          if(in_group){oracle_union.check(a,ref);union_vs_perquery.check(a,b);}
          else edge_identity.check(a,b,true);
        }
        if(!written)guard_rows++;else if(int(r)-guard+base>=ranges.groups.first&&int(r)-guard+base<ranges.groups.first+ranges.groups.count)group_rows++;else edge_rows++;
      }
      printf("base=%d P=%d sparse=%d head=%d groups=%d tail=%d\n",base,P,N-first,ranges.head.count,ranges.groups.count,ranges.tail.count);
      cases++;
    }
    oracle_union.print("union_groups_vs_independent_oracle");
    oracle_perquery.print("perquery_wmma_vs_independent_oracle");
    union_vs_perquery.print("union_groups_vs_perquery_wmma");
    edge_identity.print("head_tail_rows_vs_perquery_wmma_exact");
    printf("PASS union attention edges: %zu shapes, %zu group rows, %zu head/tail rows, %zu guard rows untouched; no complete-model qualification\n",
           cases,group_rows,edge_rows,guard_rows);
    return 0;
  }catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
