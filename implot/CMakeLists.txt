cmake_minimum_required (VERSION 3.8)

file(GLOB implot_SRC
     "*.h"
     "*.cpp"
)

add_library(implot ${implot_SRC})

target_link_libraries(implot imgui)

target_include_directories(implot PUBLIC ".")