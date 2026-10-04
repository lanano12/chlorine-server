#include "../trace.hpp"
#include <chrono>
#include <iostream>
#include <limits>

using chlorine_flash::Trace;
namespace fs=std::filesystem;
static void emit(const fs::path& dir) {
  Trace t(dir,"0,3","0,3",{{"manifest_sha256",std::string(64,'a')},
        {"binary_sha256",std::string(64,'b')},{"scope","synthetic-host-test"}});
  t.new_request();
  t.write("prefill","attn-input",0,0,{1.f,-2.f,0.f});
  t.write("prefill","logits",-1,3,{0.f,1.f,2.f,-1.f});
  t.write("prefill","attn-input",1,0,{9.f}); // unselected layer must not write
  t.write("prefill","attn-input",0,1,{9.f}); // unselected position must not write
}
int main(int argc,char** argv) {
  try {
    if(argc==3 && std::string(argv[1])=="--emit"){emit(argv[2]);return 0;}
    auto dir=fs::temp_directory_path()/("flash-trace-host-"+std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup{fs::path dir;~Cleanup(){std::error_code ec;fs::remove_all(dir,ec);}}cleanup{dir};
    emit(dir);
    bool rejected=false;try{emit(dir);}catch(const std::exception&){rejected=true;}
    if(!rejected)throw std::runtime_error("trace overwrote existing evidence");
    auto guard=dir/"guards";
    Trace t(guard,"0","0",{});
    rejected=false;try{t.write("prefill","logits",-1,0,{std::numeric_limits<float>::infinity()});}catch(const std::exception&){rejected=true;}
    if(!rejected)throw std::runtime_error("nonfinite tensor accepted");
    rejected=false;try{t.write("prefill","logits",-1,0,std::vector<float>(248321,0.f));}catch(const std::exception&){rejected=true;}
    if(!rejected)throw std::runtime_error("unbounded tensor accepted");
    std::vector<float> head(248320,1.f);
    int written=0;
    for(;;)try{t.new_request();t.write("prefill","logits",-1,0,head);written++;}catch(const std::exception&){break;}
    if(written!=int((64u<<20)/(248320*4)))throw std::runtime_error("trace byte limit not enforced");
    for(const std::string& bad:{"","-1","0,0","0,","262144","0;1"}) {
      rejected=false;try{Trace x(dir/"invalid",bad,"0",{});}catch(const std::exception&){rejected=true;}
      if(!rejected)throw std::runtime_error("invalid position accepted");
    }
    for(const std::string& bad:{"48","0,0","0,1,2,3,4,5,6,7,8"}) {
      rejected=false;try{Trace x(dir/"invalid","0",bad,{});}catch(const std::exception&){rejected=true;}
      if(!rejected)throw std::runtime_error("invalid layer accepted");
    }
    std::cout<<"PASS bounded trace writer: selection, immutable directory, FP32 validity, 64 MiB budget and configuration rejection\n";
  }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
