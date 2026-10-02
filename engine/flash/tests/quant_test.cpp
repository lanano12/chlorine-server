#include "../upstream/src/gguf.h"
#include <cassert>
#include <cstdio>
#include <random>
#include <stdexcept>
#include "iq3_oracle.hpp"
static uint64_t compared=0;
static void check(int type,const uint8_t* src,uint64_t cols) {
  std::vector<float> ref(cols),got(cols);
  if(type==gguf::IQ3_S) dequantize_row_iq3_s(reinterpret_cast<const block_iq3_s*>(src),ref.data(),cols);
  else gguf::dequant_row(type,src,ref.data(),cols);
  for(uint64_t k=0;k<cols;++k) {
    got[k]=flash_quant::value(type,src,k);
    if(std::memcmp(&got[k],&ref[k],sizeof(float))) throw std::runtime_error("decoded value differs: type="+std::to_string(type)+" col="+std::to_string(k));
  }
  compared+=cols;
}
int main(int argc,char** argv) {
 try {
  std::mt19937 rng(0x3951151);
  for(int type:{21,20,23,8}) for(int n=0;n<1024;++n) {
    uint32_t be,bb; gguf::block_info(type,be,bb);
    const int blocks=3; std::vector<uint8_t> bytes(bb*blocks);
    for(auto& b:bytes) b=uint8_t(rng());
    for(int b=0;b<blocks;++b) { // finite scales, including signed zero/subnormals
      uint16_t scale=uint16_t((n<64 ? n : rng()%0x7800) | ((n&1)<<15));
      std::memcpy(bytes.data()+b*bb,&scale,2);
    }
    check(type,bytes.data(),be*blocks);
  }
  // Enumerate every IQ3 grid id, each sign bit and all 16 subscale codes.
  for(int id=0;id<512;++id) for(int sc=0;sc<16;++sc) {
    block_iq3_s b{};b.d=0x2000;
    std::fill(b.qs,b.qs+64,uint8_t(id)); std::fill(b.qh,b.qh+8,id>=256?255:0);
    std::fill(b.scales,b.scales+4,uint8_t(sc*17));
    for(int j=0;j<32;++j)b.signs[j]=uint8_t(1u<<(j%8));
    check(21,reinterpret_cast<const uint8_t*>(&b),256);
  }
  int tensors=0;
  if(argc==2) {
    gguf::File f(argv[1]);
    for(const auto& entry:f.tensors()) {
      const auto& t=entry.second;
      if(t.name.find("_exps.weight")==std::string::npos)continue;
      for(uint64_t row:{uint64_t(0),t.rows()/2,t.rows()-1}) check(t.type,t.data+row*t.row_bytes(),t.ne[0]);
      ++tensors;
    }
  }
  printf("PASS compared=%llu real_expert_tensors=%d bit_diffs=0\n",(unsigned long long)compared,tensors);
 } catch(const std::exception& e) {fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
