#include "../upstream/src/gguf_map.h"
#include "../upstream/third_party/nlohmann/json.hpp"
#include <iostream>
#include <limits>
using Json=nlohmann::json;
struct Error {
  size_t n=0,diffs=0;double maxabs=0,maxrel=0,sum2=0;
  void add(float actual,float ref) {
    if(!std::isfinite(actual) || !std::isfinite(ref))throw std::runtime_error("nonfinite mapping result");
    double e=std::abs(double(actual)-ref);n++;diffs+=memcmp(&actual,&ref,4)!=0;
    maxabs=std::max(maxabs,e);maxrel=std::max(maxrel,e/std::max(1e-12,std::abs(double(ref))));sum2+=e*e;
  }
  Json json()const{return {{"values",n},{"bit_diffs",diffs},{"maxabs",maxabs},{"maxrel",maxrel},{"relative_floor",1e-12},{"rms",n?sqrt(sum2/n):0}};}
};
static Error direct,roundtrip,bias,normfold,gates;static Json vectors=Json::array();
static void check_decay(const std::vector<float>& a,const std::vector<float>& dt,int layer) {
  gguf_map::Opts opt;auto heads=gguf_map::map_heads(opt,48,1,0);hgn::Checkpoint ck;
  gguf_map::add_decay(ck,"L.",a,dt,heads);
  std::vector<float> c(48),l(48),d(48);ck.dequant(ck.at("L.A_coeff"),c.data());
  ck.dequant(ck.at("L.A_log"),l.data());ck.dequant(ck.at("L.dt_bias"),d.data());
  Error layer_roundtrip;
  for(int h=0;h<48;h++) {
    direct.add(c[h],a[heads[h]]);bias.add(d[h],dt[heads[h]]);roundtrip.add(-expf(l[h]),a[heads[h]]);
    layer_roundtrip.add(-expf(l[h]),a[heads[h]]);
    for(float input:{-80.f,-20.f,-1.f,0.f,1.f,20.f,80.f}) {
      const float s=input+d[h];const float sp=s>20.f?s:log1pf(expf(s));
      const float got=c[h]*sp;
      const float ref=float(double(a[heads[h]])*(double(s)>20?double(s):std::log1p(std::exp(double(s)))));
      gates.add(got,ref);
      if(std::abs(double(got)-ref)>1e-5+1e-4*std::abs(double(ref)))throw std::runtime_error("decay gate CPU budget");
    }
  }
  if(layer>=0)vectors.push_back({{"layer",layer},{"gguf_a",a},{"gguf_dt",dt},{"engine_coeff",c},{"engine_dt",d},{"legacy_log_exp_cpu",layer_roundtrip.json()}});
}
int main(int argc,char** argv) {
 try {
  int layers=0,norms=0;std::vector<float> a(48),dt(48);
  for(int h=0;h<48;h++){a[h]=-.017f*(h+1);dt[h]=float(h-24)*.031f;}check_decay(a,dt,-1);
  // Invalid coefficients reject before publishing any synthetic record.
  for(float bad:{0.f,1.f,std::numeric_limits<float>::quiet_NaN(),-std::numeric_limits<float>::infinity()}) {
    auto v=a;v[17]=bad;hgn::Checkpoint ck;bool rejected=false;
    try{gguf_map::add_decay(ck,"L.",v,dt,gguf_map::map_heads(gguf_map::Opts(),48,1,0));}catch(const std::runtime_error&){rejected=true;}
    if(!rejected || ck.tensor_count())throw std::runtime_error("invalid coefficient published");
  }
  if(argc==2) {
    gguf::File f(argv[1]);
    for(int l=0;l<48;l++)if(l%4!=3) {
      const auto prefix="blk."+std::to_string(l)+".";
      check_decay(gguf_map::rows_f32(f.at(prefix+"ssm_a")),gguf_map::rows_f32(f.at(prefix+"ssm_dt.bias")),l);layers++;
    }
    for(const auto& pair:f.tensors()) {
      const auto& t=pair.second;
      if(t.name.find("norm") == std::string::npos || t.name.find(".weight")==std::string::npos)continue;
      if(t.numel()>10240)throw std::runtime_error("unexpected norm size");
      const bool fold=t.name.find("ssm_norm.weight")==std::string::npos;
      auto original=gguf_map::rows_f32(t),mapped=gguf_map::norm_vec(t,fold);
      for(size_t i=0;i<original.size();i++) {
        const float recovered=fold?1.f+mapped[i]:mapped[i];normfold.add(recovered,original[i]);
        if(std::abs(double(recovered)-original[i])>1e-5+1e-4*std::abs(double(original[i])))throw std::runtime_error("norm mapping exceeds budget");
      }
      norms++;
    }
  }
  if(direct.diffs || bias.diffs)throw std::runtime_error("direct coefficient/bias mapping changed bits");
  std::cout<<Json({{"status","passed"},{"gpu_executed",false},{"model_loaded",false},{"real_gdn_layers",layers},
    {"norm_tensors",norms},{"direct_coefficient",direct.json()},{"bias_mapping",bias.json()},
    {"legacy_log_exp_cpu",roundtrip.json()},{"norm_fold_cpu",normfold.json()},{"direct_gate_cpu",gates.json()},
    {"vectors",vectors}}).dump(2)<<"\n";
 }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
