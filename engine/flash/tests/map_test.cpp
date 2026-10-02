#include "../upstream/src/gguf_map.h"
#include <iostream>
#include <set>
int main() {
  try {
    hgn::Checkpoint ck;
    float values[8]={1.00001f,-2.003f,0.00000017f,8193.f,1.00002f,3.1415927f,-0.f,0.125001f};
    gguf::Tensor t; t.name="fixture"; t.type=gguf::F32; t.nd=2; t.ne[0]=4;t.ne[1]=2;
    t.data=(const uint8_t*)values;t.nbytes=sizeof values;
    gguf_map::add_bf16(ck,"router",t,{2,4},{1,0});
    const auto& out=ck.at("router");
    if(out.dtype!=1 || out.data_size!=32 || memcmp(out.data,values+4,16) || memcmp(out.data+16,values,16))
      throw std::runtime_error("F32 mapping rounded or permuted incorrectly");
    gguf_map::Opts o;
    const int prefix[]={0,16,32,1,17,33,2,18,34};
    std::set<uint32_t> seen;
    for(int h=0;h<48;h++)seen.insert(gguf_map::vhead(o,h));
    if(seen.size()!=48 || *seen.rbegin()!=47)throw std::runtime_error("head map not bijective");
    for(int h=0;h<9;h++) if(gguf_map::vhead(o,h)!=(uint32_t)prefix[h]) throw std::runtime_error("head grouping");
    auto m=gguf_map::map_heads(o,48,128,4096);
    if(m[4095]!=4095 || m[4096+128]!=6144 || m[4096+256+127]!=8319)
      throw std::runtime_error("Q/K prefix or V channels changed");
    std::cout<<"PASS F32 router bits, row permutation, GDN head map\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
