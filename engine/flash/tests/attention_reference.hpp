#pragma once
// Independent sparse/dense GQA value-mix oracle. Inputs are post-norm/RoPE.
// Global double softmax over selected visible keys; no production reductions.
#include "index_reference.hpp"
#include <limits>
namespace flash_attention_reference {
constexpr int hq=24,hkv=2,dh=256;
inline std::vector<int> blocks(int seed,int position) {
  int n=(position+1)/4;std::vector<double> score(n);
  for(int b=0;b<n;b++)score[b]=flash_index_reference::input(seed,b,0)+1.;
  return flash_index_reference::select(score,n);
}
inline std::vector<int> sources(int seed,int position,bool sparse=true) {
  if(sparse && position>=2051)return flash_index_reference::tokens(blocks(seed,position),position);
  std::vector<int> ids(position+1);std::iota(ids.begin(),ids.end(),0);return ids;
}
inline std::vector<float> query(int seed,int base,int P,float scale=1.f) {
  std::vector<float> q(size_t(P)*hq*dh);
  for(int r=0;r<P;r++)for(int h=0;h<hq;h++)for(int d=0;d<dh;d++)
    q[(size_t(r)*hq+h)*dh+d]=scale*flash_index_reference::input(seed,base+r,h*dh+d);
  return q;
}
inline std::vector<float> kv(int seed,int rows,bool value,float scale=1.f) {
  std::vector<float> out(size_t(rows)*hkv*dh);
  for(int r=0;r<rows;r++)for(int h=0;h<hkv;h++)for(int d=0;d<dh;d++)
    out[(size_t(r)*hkv+h)*dh+d]=scale*flash_index_reference::input(seed+(value?97:31),r,h*dh+d);
  return out;
}
inline std::vector<float> mix(const std::vector<float>& q,const std::vector<float>& k,
                              const std::vector<float>& v,int seed,int base,int P,bool sparse=true) {
  if(base<0||P<1||q.size()!=size_t(P)*hq*dh||k.size()!=v.size()||k.size()<size_t(base+P)*hkv*dh)
    throw std::runtime_error("invalid attention shapes");
  std::vector<float> out(q.size());
  for(int r=0;r<P;r++) {
    const auto ids=sources(seed,base+r,sparse);
    for(int h=0;h<hq;h++) {
      const int kh=h/(hq/hkv);std::vector<double> score(ids.size());double peak=-std::numeric_limits<double>::infinity();
      for(size_t j=0;j<ids.size();j++) {
        double s=0;for(int d=0;d<dh;d++){
          float a=q[(size_t(r)*hq+h)*dh+d],b=k[(size_t(ids[j])*hkv+kh)*dh+d];
          if(!std::isfinite(a)||!std::isfinite(b))throw std::runtime_error("nonfinite visible attention input");
          s+=double(a)*b;
        }
        score[j]=s*.0625;peak=std::max(peak,score[j]);
      }
      double sum=0;for(double& s:score){s=std::exp(s-peak);sum+=s;}
      std::array<double,dh> acc{};
      for(size_t j=0;j<ids.size();j++)for(int d=0;d<dh;d++) {
        float value=v[(size_t(ids[j])*hkv+kh)*dh+d];
        if(!std::isfinite(value))throw std::runtime_error("nonfinite visible attention value");
        acc[d]+=score[j]/sum*value;
      }
      for(int d=0;d<dh;d++)out[(size_t(r)*hq+h)*dh+d]=float(acc[d]);
    }
  }
  return out;
}
}
