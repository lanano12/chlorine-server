#include "model_config.hpp"
#include "upstream/src/hgn.h"
#include "upstream/third_party/nlohmann/json.hpp"
#include <iostream>
int main(int argc,char** argv) {
 try {
  if(argc<2 || argc>3) throw std::runtime_error("usage: chlorine-flash-inspect FILE [--model|--hgn]");
  if(argc==3 && std::string(argv[2])=="--hgn") {
    hgn::Checkpoint f(argv[1]); uint64_t n=0, bytes=0;
    for(const auto& item:f.tensors()) { n=gguf::checked_add(n,item.second.numel()); bytes=gguf::checked_add(bytes,item.second.data_size); }
    std::cout<<nlohmann::json({{"format","hgn"},{"tensor_count",f.tensor_count()},
      {"elements",n},{"payload_bytes",bytes},{"gpu_executed",false}}).dump(2)<<"\n";
    return 0;
  }
  if(argc==3 && std::string(argv[2])!="--model") throw std::runtime_error("unknown option");
  gguf::File f(argv[1]); if(argc==3) chlorine_flash::validate_model(f);
  nlohmann::json j={{"architecture",f.arch()},{"tensor_count",f.tensors().size()},{"gpu_executed",false}};
  uint64_t elements=0,bytes=0;
  j["shards"]=nlohmann::json::array();
  for(const auto& m:f.mappings()) j["shards"].push_back({{"path",m.path},{"file_bytes",m.len}});
  j["types"]=nlohmann::json::object();
  for(const auto& entry:f.tensors()) {
   const auto& t=entry.second; elements=gguf::checked_add(elements,t.numel()); bytes=gguf::checked_add(bytes,t.nbytes);
   auto& d=j["types"][gguf::type_name(t.type)]; if(d.is_null()) d={{"tensors",0},{"payload_bytes",uint64_t(0)}};
   d["tensors"]=d["tensors"].get<int>()+1; d["payload_bytes"]=d["payload_bytes"].get<uint64_t>()+t.nbytes;
  }
  j["elements"]=elements; j["payload_bytes"]=bytes;
  std::cout<<j.dump(2)<<"\n";
 } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
