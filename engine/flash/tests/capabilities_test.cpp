#include "../capabilities.hpp"
#include <iostream>
using json=nlohmann::ordered_json;
int main() {
  json base={{"protocol","chlorine-flash-engine"},{"version",1},{"model_id","flash"},
    {"ple_semantics","causal-dilation3-v1"},{"kv_precision","f32"},{"manifest_sha256",""},{"binary_sha256",""},
    {"context",33024},{"prefill_chunk",2048},{"slots",1},{"gpu_qualified",false},{"model_qualified",false},{"vision_qualified",false}};
  auto parse=[](const json& j){return chlorine_flash::parse_capabilities("C "+j.dump());};
  if(parse(base).at("ple_semantics")!="causal-dilation3-v1")return 1;
  for(bool enabled:{false,true}) {
    auto j=base;j["diagnostic_trace"]=enabled;
    if(parse(j)["diagnostic_trace"]!=enabled)return 1;
  }
  auto bad_trace=base;bad_trace["diagnostic_trace"]=nullptr;
  bool trace_rejected=false;try{parse(bad_trace);}catch(const std::exception&){trace_rejected=true;}
  if(!trace_rejected)return 1;
  for(const char* semantic:{"causal-dilation3-v1","legacy-nine-slot-alias-v1"}) {
    auto j=base;j["ple_semantics"]=semantic;if(parse(j)["ple_semantics"]!=semantic)return 1;
  }
  int rejected=0;
  for(const char* field:{"version","protocol","context","slots","ple_semantics","model_qualified"}) {
    auto missing=base;missing.erase(field);
    try{parse(missing);}catch(const std::exception&){++rejected;}
    auto invalid=base;invalid[field]=nullptr;
    try{parse(invalid);}catch(const std::exception&){++rejected;}
  }
  auto future=base;future["version"]=2;
  try{parse(future);}catch(const std::exception&){++rejected;}
  try{chlorine_flash::parse_capabilities("CSTAT 0 0");}catch(const std::exception&){++rejected;}
  if(rejected!=14)return 1;
  std::cout<<"PASS two semantic modes, 14 malformed/unsupported capabilities rejected\n";
}
