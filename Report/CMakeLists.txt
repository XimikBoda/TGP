cmake_minimum_required (VERSION 3.8)

set(PROJECT_NAME "Report")

enable_language(CUDA)
add_executable (${PROJECT_NAME} 
  main.cpp
  T1D.cu T1D.cuh
)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER ${PROJECT_NAME})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
endif()

target_link_libraries(${PROJECT_NAME} sfml-graphics)
target_link_libraries(${PROJECT_NAME} imgui implot)

target_include_directories(${PROJECT_NAME} PUBLIC "../include")