#include "index_reference.hpp"
#include "../upstream/third_party/nlohmann/json.hpp"
#include <cstring>
#include <iostream>
#include <limits>
using Json=nlohmann::json;namespace R=flash_index_reference;
static Json run(const Json& in) {
  const int seed=in.value("seed",395),length=in.at("length");
  if(length<1 || length>16384)throw std::runtime_error("reference length outside 1..16384");
  auto positions=in.at("positions").get<std::vector<int>>();if(positions.size()>64)throw std::runtime_error("too many reference queries");
  auto chunks=in.value("chunks",std::vector<int>{length});R::Stream st;st.theta=in.value("theta",10000.);
  if(!std::isfinite(st.theta)||st.theta<=0)throw std::runtime_error("invalid theta");
  for(int n:chunks){if(n<1 || n>length-st.next)throw std::runtime_error("invalid chunks");st.append(seed,st.next,n);}
  if(st.next!=length)throw std::runtime_error("incomplete chunks");
  Json rows=Json::array();
  for(int p:positions) {
    auto scores=st.scores(seed,p);auto selected=R::select(scores,int(scores.size()));auto q=R::queries(seed,p,st.theta);
    rows.push_back({{"position",p},{"visible_blocks",scores.size()},{"blocks",selected},{"tokens",R::tokens(selected,p)},
      {"query0",q[0]},{"last_completed_key",scores.empty()?Json::array():Json(st.keys[scores.size()-1])},
      {"selected_scores",[&]{std::vector<double> s;for(int b:selected)s.push_back(scores[b]);return s;}()}});
  }
  return {{"rows",rows},{"ring",st.ring},{"complete_blocks",st.keys.size()},{"gpu_executed",false}};
}
int main(int argc,char**) {
 try {
  if(argc>1){std::string line;while(std::getline(std::cin,line))try{std::cout<<run(Json::parse(line)).dump()<<'\n';}catch(const std::exception& e){std::cout<<Json({{"error",e.what()}}).dump()<<'\n';}return 0;}
  const std::vector<int> positions={0,1,2,3,63,255,256,2046,2047,2048,2049,2050,2051,2052,2053,2054,2055,2056,4098};
  R::Stream all;all.append(395,0,4099);size_t token_checks=0;
  for(int size:{1,3,4,7,31,64,65,255,256,1024,2048,4099}) {
    R::Stream split;while(split.next<4099)split.append(395,split.next,std::min(size,4099-split.next));
    if(split.keys!=all.keys || split.ring!=all.ring)throw std::runtime_error("chunk keys/ring mismatch");
    for(int p:positions){auto a=all.selected(395,p),b=split.selected(395,p);if(a!=b)throw std::runtime_error("chunk selection mismatch");
      auto ids=R::tokens(a,p);token_checks+=ids.size();for(int t:ids)if(t>p)throw std::runtime_error("future key visibility");
      if(p<2051 && ids.size()!=size_t(p+1))throw std::runtime_error("dense crossover");}
  }
  // Selection oracle: exact ties prefer smaller IDs, negatives/nonfinite reject.
  for(int n:{0,1,511,512,513,8192,8193,65536}) {
    std::vector<double> scores(n+7,1e20);for(int i=0;i<n;i++)scores[i]=i%5;
    auto a=R::select(scores,n);if(a.size()!=size_t(std::min(n,512)))throw std::runtime_error("top-k width");
    for(int i=0;i<n;i++)scores[i]=0;
    a=R::select(scores,n);for(size_t i=0;i<a.size();i++)if(a[i]!=int(i))throw std::runtime_error("zero tie rule");
  }
  for(double value:{-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected=false;try{R::select({value},1);}catch(const std::runtime_error&){rejected=true;}if(!rejected)throw std::runtime_error("invalid score accepted");
  }
  // Computing more future block keys cannot alter any earlier visible query.
  R::Stream prefix;prefix.append(395,0,2053);
  for(int p:{2047,2051,2052})if(prefix.selected(395,p)!=all.selected(395,p))throw std::runtime_error("future pooling leak");
  std::cout<<"PASS independent index reference: 12 chunk schedules, 19 query positions, "<<token_checks<<" token visibility checks; tie/8192 boundary/future isolation\n";
 }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
