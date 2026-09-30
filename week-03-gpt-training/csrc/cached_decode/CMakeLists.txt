cmake_minimum_required(VERSION 3.22)
project(TorchLeanGPTCache LANGUAGES CXX)

set(Torch_DIR "${TORCHLEAN_LIBTORCH_HOME}/share/cmake/Torch")
set(Caffe2_DIR "${TORCHLEAN_LIBTORCH_HOME}/share/cmake/Caffe2")
list(PREPEND CMAKE_PREFIX_PATH "${TORCHLEAN_LIBTORCH_HOME}")
find_package(Torch REQUIRED CONFIG PATHS "${Torch_DIR}" NO_DEFAULT_PATH)

add_library(torchlean_gpt_cached_decode SHARED torchlean_gpt_cached_decode.cpp)
target_compile_features(torchlean_gpt_cached_decode PRIVATE cxx_std_20)
target_compile_definitions(torchlean_gpt_cached_decode PRIVATE TORCHLEAN_LIBTORCH)
separate_arguments(torch_sdk_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(torchlean_gpt_cached_decode PRIVATE ${torch_sdk_cxx_flags})
target_include_directories(torchlean_gpt_cached_decode PRIVATE
  "${TORCHLEAN_LEAN_INCLUDE}" "${TORCHLEAN_SOURCE}/csrc/libtorch")
target_link_libraries(torchlean_gpt_cached_decode PRIVATE
  "${TORCHLEAN_BACKEND}" ${TORCH_LIBRARIES})
set_target_properties(torchlean_gpt_cached_decode PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  NO_SONAME ON
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH_USE_LINK_PATH ON
  LIBRARY_OUTPUT_DIRECTORY "${TORCHLEAN_OUTPUT_DIR}")
target_link_options(torchlean_gpt_cached_decode PRIVATE "LINKER:--disable-new-dtags")
