add_library(cuLCP_Newton SHARED LCP_newton.cu)

# set_property(TARGET cuLCP_Newton PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET cuLCP_Newton PROPERTY CUDA_ARCHITECTURES 80)

target_link_libraries(cuLCP_Newton cublas Thrust cusolver)

target_compile_options(cuLCP_Newton PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --extended-lambda
                       >)
target_include_directories(cuLCP_Newton 
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
        )