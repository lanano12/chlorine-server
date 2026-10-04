#pragma once
// Independent scalar QSA oracle: no production kernel/reduction/selector calls.
// Implements IMPLEMENTATION.md 5.4; PyTorch source comparisons pin its semantics.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace flash_index_reference {
constexpr int dim=128,heads=4,ratio=4,budget=512;
using Row=std::array<float,dim>;
inline float input(int seed,int row,int channel) {
  uint32_t u=uint32_t(seed) ^ uint32_t(row)*0x9e3779b9u ^ uint32_t(channel)*0x85ebca6bu;
  u^=u>>16;u*=0x7feb352du;u^=u>>15;u*=0x846ca68bu;u^=u>>16;
  return float(int((u>>16)&1023)-512)/512.f;
}
inline Row offset() {Row w{};for(int d=0;d<dim;d++)w[d]=float(d%7-3)/32.f;return w;}
inline Row norm_rope(const Row& raw,const Row& w,int position,double eps=1e-6,double theta=10000) {
  if(position<0 || eps<=0 || theta<=0)throw std::runtime_error("invalid norm/rope parameters");
  double sum=0;for(float v:raw){if(!std::isfinite(v))throw std::runtime_error("nonfinite index input");sum+=double(v)*v;}
  const double inv=1/std::sqrt(sum/dim+eps);std::array<double,dim> n{};
  for(int d=0;d<dim;d++)n[d]=double(raw[d])*inv*(1+double(w[d]));
  Row result{};for(int d=0;d<dim;d++)result[d]=float(n[d]);
  for(int d=0;d<32;d++) {
    const double a=position*std::pow(theta,-double(d)/32),c=std::cos(a),s=std::sin(a);
    result[d]=float(n[d]*c-n[d+32]*s);result[d+32]=float(n[d+32]*c+n[d]*s);
  }
  return result;
}
inline std::vector<float> project(int seed,int first,int count) {
  std::vector<float> v(size_t(count)*640);
  for(int r=0;r<count;r++)for(int d=0;d<640;d++)v[size_t(r)*640+d]=input(seed,first+r,d);
  return v;
}
inline std::array<Row,heads> queries(int seed,int position,double theta=10000) {
  std::array<Row,heads> out{};const Row w=offset();
  for(int h=0;h<heads;h++){Row r{};for(int d=0;d<dim;d++)r[d]=input(seed,position,h*dim+d);out[h]=norm_rope(r,w,position,1e-6,theta);}
  return out;
}
inline Row pool(int seed,int block) {
  Row mean{};for(int d=0;d<dim;d++){double s=0;for(int j=0;j<ratio;j++)s+=input(seed,block*ratio+j,512+d);mean[d]=float(s/ratio);}
  return norm_rope(mean,offset(),block*ratio);
}
inline std::vector<int> select(const std::vector<double>& score,int visible_blocks) {
  if(visible_blocks<0 || size_t(visible_blocks)>score.size())throw std::runtime_error("invalid completed block count");
  std::vector<int> ids(visible_blocks);std::iota(ids.begin(),ids.end(),0);
  for(int b:ids)if(!std::isfinite(score[b]) || score[b]<0)throw std::runtime_error("invalid index score");
  std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return score[a]>score[b] || (score[a]==score[b] && a<b);});
  if(ids.size()>budget)ids.resize(budget);std::sort(ids.begin(),ids.end());return ids;
}
inline std::vector<int> tokens(const std::vector<int>& blocks,int position) {
  if(position<0)throw std::runtime_error("negative query position");
  const int n=(position+1)/ratio;std::vector<int> out;int prev=-1;
  for(int b:blocks){if(b<0 || b>=n || b<=prev)throw std::runtime_error("invalid selected block");prev=b;for(int j=0;j<ratio;j++)out.push_back(b*ratio+j);}
  for(int p=n*ratio;p<=position;p++)out.push_back(p);return out;
}
struct Stream {
  double theta=10000;
  int next=0;std::array<Row,ratio> ring{};std::vector<Row> keys;
  void append(int seed,int first,int count) {
    if(first!=next || count<1)throw std::runtime_error("noncontiguous index append");
    for(int p=first;p<first+count;p++) {
      for(int d=0;d<dim;d++)ring[p%ratio][d]=input(seed,p,512+d);
      if(p%ratio==ratio-1){Row mean{};for(int d=0;d<dim;d++){double s=0;for(int j=0;j<ratio;j++)s+=ring[j][d];mean[d]=float(s/ratio);}keys.push_back(norm_rope(mean,offset(),p-3,1e-6,theta));}
    }
    next+=count;
  }
  std::vector<double> scores(int seed,int position)const {
    if(position<0 || position>=next)throw std::runtime_error("query outside history");
    const int n=(position+1)/ratio;const auto q=queries(seed,position,theta);std::vector<double> score(n);
    for(int b=0;b<n;b++)for(int h=0;h<heads;h++) {
      double dot=0;for(int d=0;d<dim;d++)dot+=double(q[h][d])*keys[b][d];
      score[b]+=std::max(0.,dot)/std::sqrt(double(dim));
    }
    return score;
  }
  std::vector<int> selected(int seed,int position)const {auto s=scores(seed,position);return select(s,int(s.size()));}
};
}
