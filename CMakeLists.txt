cmake_minimum_required(VERSION 3.10.0)
project(cuLCP_Newton VERSION 0.1.0 LANGUAGES C CXX CUDA)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0  # Use a specific release tag or branch (e.g., master)
)

FetchContent_MakeAvailable(eigen)

# This assumes Eigen3::Eigen is already defined and includes are available
get_target_property(EIGEN_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarking)