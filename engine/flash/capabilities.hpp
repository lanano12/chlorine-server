#pragma once
#include "upstream/third_party/nlohmann/json.hpp"
#include <stdexcept>
#include <string>

namespace chlorine_flash {
// Health obtains semantics from the running engine, never its own build flags.
inline nlohmann::ordered_json parse_capabilities(const std::string& line) {
  if(line.compare(0,2,"C ")!=0)throw std::runtime_error("invalid CAPS prefix");
  auto j=nlohmann::ordered_json::parse(line.substr(2));
  if(!j.is_object() || j.at("protocol")!="chlorine-flash-engine" ||
     !j.at("version").is_number_integer() || j.at("version")!=1)
    throw std::runtime_error("unsupported Flash capability protocol");
  for(const char* field:{"model_id","ple_semantics","kv_precision","manifest_sha256","binary_sha256"})
    if(!j.at(field).is_string())throw std::runtime_error("invalid CAPS string field");
  for(const char* field:{"context","prefill_chunk","slots"})
    if(!j.at(field).is_number_integer() || j.at(field).get<int>()<=0)
      throw std::runtime_error("invalid CAPS capacity");
  for(const char* field:{"gpu_qualified","model_qualified","vision_qualified"})
    if(!j.at(field).is_boolean())throw std::runtime_error("invalid CAPS qualification");
  return j;
}
}
