include(FetchContent)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.4
)
FetchContent_MakeAvailable(benchmark)

set(PATHLIB_LOCATION "" CACHE PATH "Direction to PATH")

add_executable(benchmarking benchmarking.cu)

target_link_directories(benchmarking PRIVATE ${PATHLIB_LOCATION}/lib)
target_link_libraries(benchmarking cuLCP_Newton Thrust cublas benchmark::benchmark curand Eigen3::Eigen path50)

target_include_directories(benchmarking PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(benchmarking PRIVATE ${PATHLIB_LOCATION}/examples/cpp)
target_include_directories(benchmarking PRIVATE ${PATHLIB_LOCATION}/include)
target_include_directories(benchmarking SYSTEM PRIVATE ${EIGEN_INCLUDE_DIR})

set_property(TARGET benchmarking PROPERTY CUDA_ARCHITECTURES 80)

add_executable(robustness robustness.cu)
target_include_directories(robustness PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(robustness cuLCP_Newton Thrust Eigen3::Eigen path50)

set_property(TARGET robustness PROPERTY CUDA_ARCHITECTURES 80)

add_executable(profiling profiling.cu)
target_include_directories(profiling PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(profiling cuLCP_Newton Thrust)


set_property(TARGET profiling PROPERTY CUDA_ARCHITECTURES 80)

add_executable(avg_iters avg_iters.cu)
target_include_directories(avg_iters PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(avg_iters cuLCP_Newton Thrust)


set_property(TARGET avg_iters PROPERTY CUDA_ARCHITECTURES 80)