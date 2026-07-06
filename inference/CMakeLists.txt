cmake_minimum_required(VERSION 3.0)
project(cifar_cpp_inference)

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")

set(CMAKE_BUILD_TYPE Debug)

find_package(Torch REQUIRED)

add_executable(infer cpp_inference/main.cpp)

target_link_libraries(infer "${TORCH_LIBRARIES}")
set_property(TARGET infer PROPERTY CXX_STANDARD 17)