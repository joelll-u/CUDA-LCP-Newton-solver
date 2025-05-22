include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(
    BaseTest
    BaseTests.cu
)

target_link_libraries(BaseTest GTest::gtest_main Thrust cuLCP_Newton cublas)
target_include_directories(BaseTest PRIVATE ${PROJECT_SOURCE_DIR}/include)
# set_property(TARGET example PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET BaseTest PROPERTY CUDA_ARCHITECTURES 75)

include(GoogleTest)
gtest_discover_tests(BaseTest)