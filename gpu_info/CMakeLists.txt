cmake_minimum_required (VERSION 3.8)

set(PROJECT_NAME "gpu_info")

enable_language(CUDA)
add_executable (${PROJECT_NAME} 
  gpu_info.cu
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
endif()

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

