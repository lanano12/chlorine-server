#include "../upstream/src/gguf.h"
#include "../upstream/third_party/nlohmann/json.hpp"
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::runtime_error("usage: flash-tokenizer-metadata FIRST.gguf tokenizer.json");
    gguf::File f(argv[1]);std::ifstream in(argv[2]);nlohmann::json j;in>>j;
    const auto* vocab=f.kv("tokenizer.ggml.tokens"); const auto* merges=f.kv("tokenizer.ggml.merges");
    if(!vocab || !merges)throw std::runtime_error("GGUF tokenizer metadata missing");
    size_t compared=0;
    auto token=[&](size_t id,const std::string& text) {
      if(id>=vocab->as.size() || vocab->as[id]!=text)throw std::runtime_error("vocabulary differs at id "+std::to_string(id));
      compared++;
    };
    for(auto it=j["model"]["vocab"].begin();it!=j["model"]["vocab"].end();++it)token(it.value().get<size_t>(),it.key());
    for(const auto& t:j["added_tokens"])token(t["id"].get<size_t>(),t["content"]);
    const auto& jm=j["model"]["merges"];
    if(merges->as.size()!=jm.size())throw std::runtime_error("merge count differs");
    for(size_t i=0;i<jm.size();i++) {
      std::string m=jm[i].is_array()?jm[i][0].get<std::string>()+" "+jm[i][1].get<std::string>():jm[i].get<std::string>();
      if(m!=merges->as[i])throw std::runtime_error("merge rank differs at "+std::to_string(i));
    }
    std::cout<<nlohmann::json({{"status","passed"},{"token_entries_compared",compared},
      {"gguf_vocab_rows",vocab->as.size()},{"merges_compared",jm.size()},
      {"normalizer",j["normalizer"]},{"scope","metadata identity only; BPE behavior requires independent oracle"}}).dump(2)<<"\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
