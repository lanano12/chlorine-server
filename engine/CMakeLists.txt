cmake_minimum_required(VERSION 3.16)
project(chlorine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CHLORINE_TMAX 1024 CACHE STRING "Maximum prefill rows")
set(CHLORINE_CXT 1088 CACHE STRING "Maximum prompt plus generation positions")

add_executable(chlorine
  src/main.cpp
  src/hgn.cpp
  src/serve.cpp)
target_compile_options(chlorine PRIVATE -Wall -Wextra -O2)

# HIP trunk backend (engine/src/generator.hip). Built with hipcc when available;
# without it the engine falls back to the stub generator (wire protocol only).
find_program(HIPCC_EXECUTABLE hipcc)
if(HIPCC_EXECUTABLE)
  find_library(HIP_RUNTIME amdhip64 HINTS /opt/rocm/lib /opt/rocm/lib64)
  execute_process(COMMAND ${HIPCC_EXECUTABLE} --rocm-path
                  OUTPUT_VARIABLE ROCM_PATH OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
  find_library(OMP_RUNTIME omp
               HINTS ${ROCM_PATH}/lib/llvm/lib ${ROCM_PATH}/lib /opt/rocm/lib/llvm/lib)
  if(NOT OMP_RUNTIME)
    find_library(OMP_RUNTIME gomp)
  endif()
  find_library(HIPBLASLT_LIB hipblaslt HINTS /opt/rocm/lib /opt/rocm/lib64)
  set(GEN_OBJ ${CMAKE_CURRENT_BINARY_DIR}/generator.hip.o)
  set(LT_OBJ ${CMAKE_CURRENT_BINARY_DIR}/gemm_lt.o)
  set(GDN_OBJ ${CMAKE_CURRENT_BINARY_DIR}/gdn_gpu.o)
  set(FLASH_OBJ ${CMAKE_CURRENT_BINARY_DIR}/attn_flash.o)
  add_custom_command(OUTPUT ${GEN_OBJ}
    COMMAND ${HIPCC_EXECUTABLE} --offload-arch=gfx1151 -O3 -fopenmp -fPIC
            -DCHLORINE_TMAX=${CHLORINE_TMAX} -DCHLORINE_CXT=${CHLORINE_CXT} -c
            ${CMAKE_CURRENT_SOURCE_DIR}/src/generator.hip -o ${GEN_OBJ}
    DEPENDS src/generator.hip
    COMMENT "hipcc generator.hip -> gfx1151")
  add_custom_command(OUTPUT ${LT_OBJ}
    COMMAND ${HIPCC_EXECUTABLE} --offload-arch=gfx1151 -O3 -fPIC -c
            ${CMAKE_CURRENT_SOURCE_DIR}/src/gemm_lt.cpp -o ${LT_OBJ}
    DEPENDS src/gemm_lt.cpp
    COMMENT "hipcc gemm_lt.cpp")
  add_custom_command(OUTPUT ${GDN_OBJ}
    COMMAND ${HIPCC_EXECUTABLE} --offload-arch=gfx1151 -O3 -fPIC -ffp-contract=off -c
            ${CMAKE_CURRENT_SOURCE_DIR}/src/gdn_gpu.hip -o ${GDN_OBJ}
    DEPENDS src/gdn_gpu.hip src/gdn_expf.h
    COMMENT "hipcc gdn_gpu.hip")
  add_custom_command(OUTPUT ${FLASH_OBJ}
    COMMAND ${HIPCC_EXECUTABLE} --offload-arch=gfx1151 -O3 -fPIC -c
            ${CMAKE_CURRENT_SOURCE_DIR}/src/attn_flash.hip -o ${FLASH_OBJ}
    DEPENDS src/attn_flash.hip
    COMMENT "hipcc causal f16 WMMA attention")
  target_sources(chlorine PRIVATE ${GEN_OBJ} ${LT_OBJ} ${GDN_OBJ} ${FLASH_OBJ})
  if(HIP_RUNTIME)
    target_link_libraries(chlorine PRIVATE ${HIP_RUNTIME})
  endif()
  if(HIPBLASLT_LIB)
    target_link_libraries(chlorine PRIVATE ${HIPBLASLT_LIB})
  endif()
  if(OMP_RUNTIME)
    target_link_libraries(chlorine PRIVATE ${OMP_RUNTIME})
  endif()
  target_compile_definitions(chlorine PRIVATE CHLORINE_HAS_TRUNK=1)
endif()
