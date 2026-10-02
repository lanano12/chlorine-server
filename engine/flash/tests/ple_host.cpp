#include "../ple_contract.hpp"
#include "../upstream/third_party/nlohmann/json.hpp"
#include <iostream>
#include <string>
#include <vector>
using json=nlohmann::json;
int main() {
  std::string line;
  while(std::getline(std::cin,line)) {
    try {
      auto q=json::parse(line);json out={{"mode",chlorine_flash::ple_semantics}};
      const auto op=q.at("op").get<std::string>();
      if(op=="conv") {
        const int n=q.at("width"),T=q.at("rows");
        auto x=q.at("input").get<std::vector<float>>(),w=q.at("weights").get<std::vector<float>>();
        if(n<=0 || T<=0 || x.size()!=size_t(n)*T || w.size()!=size_t(n)*4)throw std::runtime_error("invalid shape");
        auto chunks=q.value("chunks",std::vector<int>{T});
        std::vector<float> ring(chlorine_flash::ple_ring_slots*n,0.f),y(x.size());
        int base=0;
        for(int P:chunks) {
          if(P<1 || P>T-base)throw std::runtime_error("invalid chunk length");
          for(int t=0;t<P;t++) for(int c=0;c<n;c++)
            y[size_t(base+t)*n+c]=chlorine_flash::ple_conv_row(x.data()+size_t(base)*n,ring.data(),w.data(),t,c,n,base);
          for(int t=std::max(0,P-chlorine_flash::ple_ring_slots);t<P;t++)
            std::copy_n(x.data()+size_t(base+t)*n,n,ring.data()+size_t((base+t)%chlorine_flash::ple_ring_slots)*n);
          base+=P;
        }
        if(base!=T)throw std::runtime_error("incomplete chunks");
        out["output"]=y;out["ring"]=ring;
      } else if(op=="gate") {
        auto x=q.at("dots").get<std::vector<float>>();for(float& f:x)f=chlorine_flash::ple_gate(f);out["output"]=x;
      } else if(op=="history") {
        auto ids=q.at("ids").get<std::vector<int>>();int eos=q.at("eos");
        for(int back=0;back<3;back++) {
          std::vector<int> shifted;
          for(int t=0;t<(int)ids.size();t++)shifted.push_back(chlorine_flash::ple_history_token(ids.data(),t,back,eos));
          out["shifted"].push_back(shifted);
        }
      } else throw std::runtime_error("unknown op");
      std::cout<<out.dump()<<"\n"<<std::flush;
    }catch(const std::exception& e){std::cout<<json({{"error",e.what()}}).dump()<<"\n"<<std::flush;}
  }
}
