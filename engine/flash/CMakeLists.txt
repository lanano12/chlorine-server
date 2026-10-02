cmake_minimum_required(VERSION 3.20)
project(chlorine_flash LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(FLASH_BUILD_GPU "Build gfx1151 HIP engine and GPU checks" ON)
option(FLASH_BUILD_API "Build the source Flash HTTP API" ON)
option(FLASH_PLE_CAUSAL "Use independently specified PLE back-nine history and EOS hash boundaries (GPU qualification required)" OFF)
option(FLASH_GGUF_DIRECT_DECAY "Consume GGUF's stored negative GDN coefficient without log/exp round trip" OFF)
set(FLASH_SEMANTIC_FLAGS -DFLASH_PLE_CAUSAL=$<BOOL:${FLASH_PLE_CAUSAL}> -DFLASH_GGUF_DIRECT_DECAY=$<BOOL:${FLASH_GGUF_DIRECT_DECAY}>)
include(CTest)
foreach(mode IN ITEMS legacy causal)
  add_executable(flash-ple-${mode} tests/ple_host.cpp)
  target_compile_options(flash-ple-${mode} PRIVATE -ffp-contract=off)
endforeach()
target_compile_definitions(flash-ple-causal PRIVATE FLASH_PLE_CAUSAL=1)
add_executable(chlorine-flash-inspect inspect.cpp)
add_executable(flash-quant-test tests/quant_test.cpp)
add_executable(flash-map-test tests/map_test.cpp)
add_executable(flash-q6-test tests/q6_test.cpp)
add_executable(flash-mapping-test tests/mapping_test.cpp)
target_compile_options(flash-mapping-test PRIVATE -ffp-contract=off)
add_test(NAME flash-mapping COMMAND flash-mapping-test)
target_compile_options(flash-q6-test PRIVATE -ffp-contract=off)
add_test(NAME flash-q6 COMMAND flash-q6-test)
add_executable(flash-tokenizer-metadata tests/tokenizer_metadata.cpp)
add_executable(flash-capabilities-test tests/capabilities_test.cpp)
add_test(NAME flash-capabilities COMMAND flash-capabilities-test)
find_package(Threads REQUIRED)
target_link_libraries(flash-map-test PRIVATE Threads::Threads)
target_link_libraries(flash-q6-test PRIVATE Threads::Threads)
target_link_libraries(flash-mapping-test PRIVATE Threads::Threads)
add_test(NAME flash-map COMMAND flash-map-test)
target_compile_options(flash-quant-test PRIVATE -ffp-contract=off)
add_test(NAME flash-quant COMMAND flash-quant-test)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME flash-ple-host COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/ple_fixture_test.py $<TARGET_FILE:flash-ple-causal>
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ple_reference.json)
add_test(NAME flash-loader COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/loader_test.py $<TARGET_FILE:chlorine-flash-inspect>)
add_test(NAME flash-hgn-loader COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/hgn_test.py $<TARGET_FILE:chlorine-flash-inspect>)
if(FLASH_BUILD_GPU)
  find_program(HIPCC_EXECUTABLE hipcc HINTS /opt/rocm/bin REQUIRED)
  file(GLOB_RECURSE FLASH_HIP_DEPS CONFIGURE_DEPENDS upstream/src/gpu/* upstream/src/*.h quant/*)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/chlorine-flash
    COMMAND ${HIPCC_EXECUTABLE} --offload-arch=gfx1151 -O3 -Werror ${FLASH_SEMANTIC_FLAGS}
      ${CMAKE_CURRENT_SOURCE_DIR}/upstream/src/gpu/gdec.cpp -lrocblas -lhipblaslt
      -o ${CMAKE_CURRENT_BINARY_DIR}/chlorine-flash
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/write_build_info.py
      --binary ${CMAKE_CURRENT_BINARY_DIR}/chlorine-flash --root ${CMAKE_CURRENT_SOURCE_DIR}
      --hipcc ${HIPCC_EXECUTABLE} --ple-causal $<BOOL:${FLASH_PLE_CAUSAL}>
      --gguf-direct-decay $<BOOL:${FLASH_GGUF_DIRECT_DECAY}>
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/chlorine-flash.build.json
    DEPENDS ${FLASH_HIP_DEPS} model_config.hpp ple_contract.hpp mapping_contract.hpp write_build_info.py VERBATIM)
  add_custom_target(flash-engine ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/chlorine-flash)
endif()
if(FLASH_BUILD_API)
  find_package(PNG REQUIRED)
  find_package(JPEG REQUIRED)
  find_library(WEBP_LIBRARY webp REQUIRED)
  find_package(Threads REQUIRED)
  set(API_SRC upstream/src/api)
  add_executable(chlorine-flash-api ${API_SRC}/http.cpp ${API_SRC}/engine_client.cpp
    ${API_SRC}/tokenizer.cpp ${API_SRC}/chat_template.cpp ${API_SRC}/json_py.cpp
    ${API_SRC}/toolparse.cpp ${API_SRC}/vision.cpp ${API_SRC}/reqstat.cpp
    ${API_SRC}/reqstat_read.cpp ${API_SRC}/main.cpp)
  target_include_directories(chlorine-flash-api PRIVATE upstream/src/api upstream/third_party)
  target_link_libraries(chlorine-flash-api PRIVATE PNG::PNG JPEG::JPEG ${WEBP_LIBRARY} Threads::Threads)
  add_executable(flash-toolparse-test ${API_SRC}/toolparse.cpp ${API_SRC}/json_py.cpp ${API_SRC}/toolparse_test.cpp)
  target_include_directories(flash-toolparse-test PRIVATE upstream/third_party)
  add_test(NAME flash-toolparse COMMAND flash-toolparse-test)
endif()
if(FLASH_BUILD_GPU)
  file(GLOB_RECURSE FLASH_TEST_DEPS CONFIGURE_DEPENDS tests/*.hpp)
  foreach(check IN ITEMS moe_test index_fast_test gdn_verify_test gemv_test ple_test mapping_gpu_test)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${check}
      COMMAND ${HIPCC_EXECUTABLE} --offload-arch=gfx1151 -O3 -Werror ${FLASH_SEMANTIC_FLAGS}
        -I${CMAKE_CURRENT_SOURCE_DIR}/upstream/src/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/${check}.hip -lrocblas -lhipblaslt
        -o ${CMAKE_CURRENT_BINARY_DIR}/${check}
      DEPENDS tests/${check}.hip ${FLASH_TEST_DEPS} ${FLASH_HIP_DEPS} model_config.hpp ple_contract.hpp mapping_contract.hpp VERBATIM)
    add_custom_target(${check}-build ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${check})
  endforeach()
  foreach(format IN ITEMS iq3s iq4xs iq4nl)
    foreach(rows IN ITEMS 1 2 8 17 65)
      add_test(NAME flash-moe-${format}-${rows} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/moe_test
        --synth ${format} --E 10 --P ${rows} --iters 1 --check-experts 2)
      set_tests_properties(flash-moe-${format}-${rows} PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)
    endforeach()
  endforeach()
  add_test(NAME flash-gemv COMMAND ${CMAKE_CURRENT_BINARY_DIR}/gemv_test)
  add_test(NAME flash-index COMMAND ${CMAKE_CURRENT_BINARY_DIR}/index_fast_test --quick)
  add_test(NAME flash-gdn COMMAND ${CMAKE_CURRENT_BINARY_DIR}/gdn_verify_test)
  add_test(NAME flash-mapping-gpu COMMAND ${CMAKE_CURRENT_BINARY_DIR}/mapping_gpu_test)
  add_test(NAME flash-ple-gpu COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ple_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ple_reference.json)
  set_tests_properties(flash-index flash-gdn flash-gemv flash-ple-gpu flash-mapping-gpu PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)
endif()
if(FLASH_BUILD_API)
  add_executable(flash-tok upstream/src/api/tokenizer.cpp upstream/src/api/tok_cli.cpp)
  target_include_directories(flash-tok PRIVATE upstream/third_party)
  add_executable(flash-template upstream/src/api/chat_template.cpp upstream/src/api/json_py.cpp upstream/src/api/tpl_cli.cpp)
  target_include_directories(flash-template PRIVATE upstream/third_party)
endif()
