cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_FLAGS "-fpermissive -fPIC -std=c++17")
set(LIB_NAME transcription_model)

set(
  SOURCES
    model.h
    model.cc
    translate.h
    translate.cc
)

add_library(${LIB_NAME} SHARED ${SOURCES})

# MLC-LLM specific
set(MLC_LLM_HOME ${PROJECT_SOURCE_DIR}/mlc-llm CACHE INTERNAL "")
set(MLC_LLM_3RDPARTY ${MLC_LLM_HOME}/3rdparty/tokenizers-cpp/include CACHE INTERNAL "")
set(MLC_LLM_TVM_HOME ${MLC_LLM_HOME}/3rdparty/tvm)

set(
  MLC_LLM_TVM_INCLUDE_DIRS
    ${MLC_LLM_TVM_HOME}/include
    ${MLC_LLM_TVM_HOME}/3rdparty/dlpack/include
    ${MLC_LLM_TVM_HOME}/3rdparty/dmlc-core/include
    ${MLC_LLM_TVM_HOME}/3rdparty/picojson
  CACHE INTERNAL ""
)

target_include_directories(
  ${LIB_NAME}
  PUBLIC
    ${MLC_LLM_HOME}
    ${MLC_LLM_3RDPARTY}
    ${MLC_LLM_TVM_INCLUDE_DIRS}
)

target_link_libraries(
  ${LIB_NAME}
  PRIVATE
    mlc_llm
)
