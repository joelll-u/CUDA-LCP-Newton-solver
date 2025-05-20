include(FetchContent)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.4  # Or a specific release like v1.8.3
)
FetchContent_MakeAvailable(benchmark)

add_executable(benchmarking benchmarking.cu)

target_link_libraries(benchmarking cuLCP_Newton Thrust cublas benchmark::benchmark curand)
target_include_directories(benchmarking PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_property(TARGET benchmarking PROPERTY CUDA_ARCHITECTURES 80)