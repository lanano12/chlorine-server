#pragma once
// Include after gdec.cpp: CK and the instrumented HIP allocation API are required.
template<class T>struct RefBuffer {
  T* p=nullptr;size_t n;
  explicit RefBuffer(size_t count):n(count){CK(hipMalloc(&p,n*sizeof(T)));CK(hipMemset(p,0,n*sizeof(T)));}
  explicit RefBuffer(const std::vector<T>& v):RefBuffer(v.size()){put(v);}
  ~RefBuffer(){if(p){auto e=hipFree(p);if(e!=hipSuccess)fprintf(stderr,"HIP cleanup failed: %s\n",hipGetErrorString(e));}}
  void put(const std::vector<T>& v){if(v.size()!=n)throw std::runtime_error("buffer shape");CK(hipMemcpy(p,v.data(),n*sizeof(T),hipMemcpyHostToDevice));}
  std::vector<T> get(){std::vector<T> v(n);CK(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost));return v;}
};
struct RefError {
  size_t n=0,bits=0;double maxabs=0,maxrel=0,sum2=0;
  void check(float a,float b,bool exact=false){double e=std::abs(double(a)-b);n++;bits+=memcmp(&a,&b,4)!=0;
    maxabs=std::max(maxabs,e);maxrel=std::max(maxrel,e/std::max(1e-12,std::abs(double(b))));sum2+=e*e;
    if(!std::isfinite(a)||!std::isfinite(b)||e>1e-5+1e-4*std::abs(double(b))||(exact&&memcmp(&a,&b,4)))throw std::runtime_error("FP32 reference operator contract failure");}
  void print(const char* tag){printf("%s n=%zu maxabs=%.9g maxrel=%.9g rms=%.9g bit_diffs=%zu relative_floor=1e-12\n",tag,n,maxabs,maxrel,n?sqrt(sum2/n):0,bits);}
};
