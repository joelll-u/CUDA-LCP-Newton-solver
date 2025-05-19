cmake_minimum_required(VERSION 3.10.0)
project(cuLCP_Newton VERSION 0.1.0 LANGUAGES C CXX CUDA)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

SET(CMAKE_BUILD_TYPE Debug)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
