# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project("TGP")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)


option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

add_subdirectory(imgui)
add_subdirectory(implot)
add_subdirectory(SFML)
add_subdirectory(gpu_info)
add_subdirectory(Lab2)
add_subdirectory(Report)