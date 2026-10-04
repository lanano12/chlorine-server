#pragma once
// Bounded, opt-in diagnostics. Never captures full prompt x vocabulary arrays.
#include "upstream/third_party/nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace chlorine_flash {
class Trace {
  using Json = nlohmann::json;
  std::filesystem::path dir_;
  std::set<int> positions_, layers_;
  std::ofstream index_;
  size_t bytes_=0, records_=0;
  int request_=0;
  std::set<std::string> coordinates_;
  static constexpr size_t max_bytes=64u<<20, max_records=2048;
  static std::string env(const char* key) {const char* v=std::getenv(key);return v?v:"";}
  static std::set<int> list(const std::string& s, int limit, size_t cap) {
    std::set<int> out;std::istringstream in(s);std::string part;
    while(std::getline(in,part,',')) {
      if(part.empty() || !std::all_of(part.begin(),part.end(),[](char c){return c>='0'&&c<='9';}))
        throw std::runtime_error("trace selections must be comma-separated unsigned integers");
      size_t used=0;int n=std::stoi(part,&used);
      if(used!=part.size()||n>=limit||!out.insert(n).second||out.size()>cap)
        throw std::runtime_error("trace selection duplicate or outside limit");
    }
    if(out.empty() || (!s.empty() && s.back()==','))throw std::runtime_error("empty trace selection");
    return out;
  }
  void line(const Json& j) {index_<<j.dump()<<'\n';index_.flush();if(!index_)throw std::runtime_error("trace index write failed");}
public:
  Trace() = default;
  Trace(const std::filesystem::path& dir, const std::string& positions,
        const std::string& layers, const Json& identity) {
    positions_=list(positions,262144,32);layers_=list(layers,48,8);
    uint32_t endian=1;
    if(*reinterpret_cast<unsigned char*>(&endian)!=1 || sizeof(float)!=4)
      throw std::runtime_error("trace requires little-endian FP32");
    // create_directory is exclusive, so existing diagnostic evidence survives.
    if(!std::filesystem::create_directory(dir))throw std::runtime_error("trace directory already exists");
    dir_=dir;index_.open(dir_/"index.jsonl",std::ios::out|std::ios::trunc);
    if(!index_)throw std::runtime_error("cannot create trace index");
    line({{"schema_version",1},{"kind","chlorine-flash-diagnostic-trace"},
          {"identity",identity},{"positions",positions_},{"layers",layers_},
          {"max_bytes",max_bytes},{"max_records",max_records},{"graph_replay",false},
          {"timing_eligible",false},{"dtype","f32le"}});
  }
  static Trace from_environment() {
    auto dir=env("CHLORINE_FLASH_TRACE_DIR");
    if(dir.empty()) {
      if(!env("CHLORINE_FLASH_TRACE_POSITIONS").empty()||!env("CHLORINE_FLASH_TRACE_LAYERS").empty())
        throw std::runtime_error("trace selections require CHLORINE_FLASH_TRACE_DIR");
      return {};
    }
    for(const auto& item:std::vector<std::pair<const char*,const char*>>{{"GDEC_NOSPEC","1"},{"GDEC_DRAFTER","serial"},{"GDEC_PARALLEL","1"},{"GDEC_KVSNAP","0"},{"GDEC_RCKPT","0"}})
      if(env(item.first)!=item.second)throw std::runtime_error("trace requires serial one-slot execution with cache reuse disabled");
    for(const char* key:{"CHLORINE_FLASH_BINARY_SHA256","CHLORINE_FLASH_MANIFEST_SHA256"}) {
      std::string digest=env(key);
      if(digest.size()!=64 || digest.find_first_not_of("0123456789abcdef")!=std::string::npos)
        throw std::runtime_error("trace requires supervisor-supplied binary and checkpoint identities");
    }
    return Trace(dir,env("CHLORINE_FLASH_TRACE_POSITIONS"),env("CHLORINE_FLASH_TRACE_LAYERS"),
      {{"binary_sha256",env("CHLORINE_FLASH_BINARY_SHA256")},
       {"manifest_sha256",env("CHLORINE_FLASH_MANIFEST_SHA256")}});
  }
  bool active() const {return index_.is_open();}
  const std::set<int>& positions() const {return positions_;}
  bool wants(int layer,int position) const {
    return active() && positions_.count(position) && (layer==-1 || layers_.count(layer));
  }
  void new_request() {if(active()){
    if(request_>=int(max_records))throw std::runtime_error("trace request limit exceeded");
    ++request_;line({{"event","request"},{"request",request_}});
  }}
  void write(const char* phase,const char* stage,int layer,int position,const std::vector<float>& values) {
    if(!wants(layer,position))return;
    size_t bytes=values.size()*sizeof(float);
    if(values.empty()||values.size()>248320||records_>=max_records||bytes>max_bytes-bytes_)
      throw std::runtime_error("trace record or byte limit exceeded");
    if((std::string(phase)!="prefill" && std::string(phase)!="decode") || std::strlen(stage)>64)
      throw std::runtime_error("invalid trace phase or stage");
    for(float f:values)if(!std::isfinite(f))throw std::runtime_error("nonfinite trace tensor");
    std::string coordinate=std::to_string(request_)+":"+phase+":"+stage+":"+
      std::to_string(layer)+":"+std::to_string(position);
    if(coordinates_.count(coordinate))throw std::runtime_error("duplicate trace coordinate");
    std::string filename="tensor-"+std::to_string(records_)+".f32";
    const auto* data=reinterpret_cast<const unsigned char*>(values.data());
    uint64_t checksum=14695981039346656037ULL;
    for(size_t i=0;i<bytes;i++){checksum^=data[i];checksum*=1099511628211ULL;}
    std::ofstream file(dir_/filename,std::ios::binary|std::ios::out|std::ios::trunc);
    file.write(reinterpret_cast<const char*>(data),std::streamsize(bytes));file.close();
    if(!file)throw std::runtime_error("trace tensor write failed");
    line({{"event","tensor"},{"request",request_},{"phase",phase},{"stage",stage},{"layer",layer},
          {"position",position},{"shape",Json::array({values.size()})},{"dtype","f32le"},
          {"file",filename},{"bytes",bytes},{"fnv1a64",std::to_string(checksum)}});
    coordinates_.insert(coordinate);bytes_+=bytes;++records_;
  }
};
} // namespace chlorine_flash
