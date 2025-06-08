add_library(cuLCP_Newton SHARED LCP_newton.cu)

# set_property(TARGET cuLCP_Newton PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET cuLCP_Newton PROPERTY CUDA_ARCHITECTURES 80)

set(cuDSSDIR "" CACHE PATH "Path to cudss")

target_link_directories(cuLCP_Newton PRIVATE ${cuDSSDIR}/lib)

# find_library(CUDSS_LIB cudss PATHS ${cuDSSDIR}/lib REQUIRED)
target_link_libraries(cuLCP_Newton cublas Thrust cusolver cudss cusparse)

target_compile_options(cuLCP_Newton PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --extended-lambda
                       >)

target_include_directories(cuLCP_Newton 
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
        )