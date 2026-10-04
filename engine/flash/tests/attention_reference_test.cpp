#include "attention_reference.hpp"
#include "../upstream/third_party/nlohmann/json.hpp"
#include <iostream>
namespace A=flash_attention_reference;using Json=nlohmann::json;
static Json run(const Json& in) {
  int seed=in.value("seed",395),base=in.at("base"),P=in.at("rows");float scale=in.value("scale",1.f);bool sparse=in.value("sparse",true);
  if(base<0||base>8192||P<1||P>33||base+P>8193||!std::isfinite(scale)||scale<=0||scale>8)
    throw std::runtime_error("attention reference request outside bound");
  auto q=A::query(seed,base,P,scale),k=A::kv(seed,base+P,false,scale),v=A::kv(seed,base+P,true);
  auto out=A::mix(q,k,v,seed,base,P,sparse);Json ids=Json::array();
  for(int r=0;r<P;r++)ids.push_back(A::sources(seed,base+r,sparse));
  return {{"output",out},{"sources",ids},{"gpu_executed",false}};
}
int main(int argc,char**) {
  try {
    if(argc>1){std::string line;while(std::getline(std::cin,line))try{std::cout<<run(Json::parse(line)).dump()<<'\n';}catch(const std::exception& e){std::cout<<Json({{"error",e.what()}}).dump()<<'\n';}return 0;}
    size_t values=0;
    for(auto shape:std::vector<std::pair<int,int>>{{0,1},{0,33},{5,4},{17,1},{2047,10},{2051,5},{8191,1}}) {
      int base=shape.first,P=shape.second;auto q=A::query(395,base,P),k=A::kv(395,base+P+8,false),v=A::kv(395,base+P+8,true);
      auto expected=A::mix(q,k,v,395,base,P);
      // Extra inaccessible rows are poisoned. Visibility is per query, so a
      // suffix may not influence an earlier row, even at the sparse crossover.
      for(size_t i=size_t(base+P)*A::hkv*A::dh;i<k.size();i++)k[i]=v[i]=std::numeric_limits<float>::quiet_NaN();
      if(A::mix(q,k,v,395,base,P)!=expected)throw std::runtime_error("future poison leak");
      for(int r=0;r<P;r++) {
        std::vector<float> qr(q.begin()+size_t(r)*A::hq*A::dh,q.begin()+size_t(r+1)*A::hq*A::dh);
        auto single=A::mix(qr,k,v,395,base+r,1);
        if(!std::equal(single.begin(),single.end(),expected.begin()+size_t(r)*A::hq*A::dh))
          throw std::runtime_error("attention row/chunk address mismatch");
      }
      values+=expected.size();
    }
    // Distinct constant value planes make the 12-query-to-one-KV map exact.
    auto q=A::query(395,0,4),k=A::kv(395,4,false),v=A::kv(395,4,true);
    for(int r=0;r<4;r++)for(int h=0;h<2;h++)for(int d=0;d<256;d++)v[(r*2+h)*256+d]=h?2.f:1.f;
    auto out=A::mix(q,k,v,395,0,4);
    for(int r=0;r<4;r++)for(int h=0;h<24;h++)for(int d=0;d<256;d++)
      if(out[(r*24+h)*256+d]!=(h<12?1.f:2.f))throw std::runtime_error("GQA head identity");
    std::cout<<"PASS scalar GQA attention: seven shapes, "<<values<<" values, per-row chunk identity, future poison and 24:2 head map\n";
  }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
