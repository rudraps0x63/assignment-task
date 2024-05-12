cmake_minimum_required(VERSION 3.25)

project(assignment-task CXX C)

set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC ${CMAKE_CXX_FLAGS}")

# MLC-LLM specific
if(APPLE)
  set(USE_METAL ON CACHE BOOL "Use the Metal API for TVM runtime")
else()
  set(USE_VULKAN ON CACHE BOOL "Use the Vulkan API for TVM runtime")
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/mlc-llm)
# --

add_subdirectory(${PROJECT_SOURCE_DIR}/lib)

add_executable(${PROJECT_NAME} main.cc)

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE
    ${MLC_LLM_HOME} # Cached
    ${MLC_LLM_3RDPARTY} # Cached
    ${MLC_LLM_TVM_INCLUDE_DIRS} # Cached
)

target_link_libraries(
  ${PROJECT_NAME}
    transcription_model
)
