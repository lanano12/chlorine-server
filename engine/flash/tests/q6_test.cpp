#include "../upstream/src/gguf_map.h"
#include "q6_oracle.hpp"
#include <iostream>
#include <random>
#include <set>

static size_t compared=0;
static void check(const uint8_t* data,size_t cols) {
  std::vector<float> a(cols),b(cols);
  dequantize_row_q6_K((const block_q6_K*)data,a.data(),cols);
  gguf::dequant_row(gguf::Q6_K,data,b.data(),cols);
  for(size_t i=0;i<cols;i++)if(memcmp(&a[i],&b[i],4))
    throw std::runtime_error("Q6_K decoder differs at column "+std::to_string(i));
  hgn::Checkpoint ck;auto t=gguf_map::desc("test",12,{1,cols});t.data=data;t.data_size=cols/256*210;
  ck.add_view(t);ck.dequant(ck.at("test"),b.data());
  if(memcmp(a.data(),b.data(),cols*4))throw std::runtime_error("Q6_K HGN view changed values");
  compared+=cols;
}
int main(int argc,char** argv) {
 try {
  std::mt19937 rng(0x613951);
  // Adjacent blocks, every 6-bit code and signed scale, both low/high halves.
  for(int test=0;test<4096;test++) {
    std::vector<block_q6_K> blocks(3);
    for(auto& b:blocks) {
      for(auto& v:b.ql)v=uint8_t(rng());for(auto& v:b.qh)v=uint8_t(rng());
      for(auto& v:b.scales)v=int8_t(test%256-128);
      b.d=uint16_t((test<64?test:rng()%0x7c00) | ((test&1)<<15));
    }
    check((const uint8_t*)blocks.data(),blocks.size()*256);
  }
  size_t real_rows=0,packed_bytes=0;
  if(argc==2) {
    gguf::File g(argv[1]);const auto& t=g.at("output.weight");
    if(t.type!=gguf::Q6_K || t.ne[0]!=2560 || t.ne[1]!=248320)throw std::runtime_error("actual head shape/type");
    hgn::Checkpoint ck;gguf_map::add_head(ck,t);const auto& head=ck.at("lm_head.weight");
    if(head.dtype!=12 || head.data!=t.data || head.data_size!=t.nbytes)throw std::runtime_error("head is not an unchanged packed view");
    packed_bytes=head.data_size;
    std::set<size_t> rows={0,1,255,256,257,248043,248044,248045,248046,248319};
    for(size_t i=0;i<32;i++)rows.insert(i*(t.rows()-1)/31);
    for(size_t r:rows){check(t.data+r*t.row_bytes(),t.ne[0]);real_rows++;}
  }
  // An unsupported head must reject before allocating or reading its payload.
  gguf::Tensor invalid;invalid.name="output.weight";invalid.nd=2;invalid.ne[0]=2560;invalid.ne[1]=248320;invalid.type=gguf::IQ3_S;
  hgn::Checkpoint ck;bool rejected=false;try{gguf_map::add_head(ck,invalid);}catch(const std::runtime_error&){rejected=true;}
  if(!rejected || ck.tensor_count())throw std::runtime_error("unsupported head was accepted");
  std::cout<<"PASS Q6_K compared="<<compared<<" bit_diffs=0 real_head_rows="<<real_rows<<" packed_bytes="<<packed_bytes<<"\n";
 }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
