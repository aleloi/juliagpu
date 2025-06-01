cmake_minimum_required(VERSION 3.13)


project(hello_webgpu LANGUAGES CXX)

# ▸ Tell the compiler we need at least C++17 (swap with 20 if you like)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package(Dawn REQUIRED)
find_package(glfw3 REQUIRED)

# Define source files
set(SOURCE_FILES hello_webgpu.cpp)

# Platform-specific configuration
if(APPLE)
  # Add Objective-C++ file for Metal support on macOS
  list(APPEND SOURCE_FILES metal_layer_helper.mm)
  
  # Enable Objective-C++ compilation
  set_source_files_properties(metal_layer_helper.mm PROPERTIES
    COMPILE_FLAGS "-x objective-c++"
  )
  
  # Add required frameworks
  find_library(COCOA_LIBRARY Cocoa REQUIRED)
  find_library(QUARTZCORE_LIBRARY QuartzCore REQUIRED)
  set(PLATFORM_LIBS ${COCOA_LIBRARY} ${QUARTZCORE_LIBRARY})
endif()

add_executable(hello_webgpu ${SOURCE_FILES})

# Declare dependency on the dawn::webgpu_dawn library and GLFW
target_link_libraries(hello_webgpu dawn::webgpu_dawn glfw ${PLATFORM_LIBS})
