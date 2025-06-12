cmake_minimum_required(VERSION 3.13)

project(julia_fractal_webgpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# WebGPU configuration - use precompiled Dawn binaries
set(WEBGPU_BACKEND "DAWN" CACHE STRING "Backend to use")
set(WEBGPU_BUILD_FROM_SOURCE OFF CACHE BOOL "Build WebGPU from source")

# Add WebGPU distribution
add_subdirectory(../webgpu webgpu)

# Debug/Release configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -DDEBUG -D_DEBUG")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wunused -Wunreachable-code
        -Wcast-align -Wcast-qual
        -Wctor-dtor-privacy
        -Wdisabled-optimization
        -Wformat=2
        -Winit-self
        -Wmissing-declarations
        -Wmissing-include-dirs
        -Wold-style-cast
        -Woverloaded-virtual
        -Wredundant-decls
        -Wshadow -Wsign-conversion
        -Wsign-promo
        -Wstrict-overflow=5
        -Wswitch-default -Wundef
        -Wno-unused-parameter
    )
    
    add_compile_options(-fsanitize=address -fsanitize=undefined)
    add_link_options(-fsanitize=address -fsanitize=undefined)
    
    # Dawn-specific debug defines
    add_compile_definitions(
        DAWN_ENABLE_ASSERTS=1
        DAWN_ENABLE_BACKEND_VALIDATION=1
        DAWN_ENABLE_DEVICE_REMOVAL_TESTS=1
        DAWN_ENABLE_VULKAN_VALIDATION_LAYERS=1
        DAWN_ENABLE_WEBGPU_DEBUGGING=1
    )
    
    message(STATUS "Debug build configuration enabled with enhanced debugging")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    message(STATUS "Release build configuration")
endif()

if(UNIX)
    add_compile_definitions(ENABLE_STACK_TRACE=1)
endif()

find_package(glfw3 REQUIRED)

set(SOURCE_FILES 
    main.cc
    input_handler.cc
    webgpu_setup.cc
    compute_shaders.cc
)

# Attempts to be cross-platform, but we can only build on macOS.
if(APPLE)
  list(APPEND SOURCE_FILES metal_layer_helper.mm)
    set_source_files_properties(metal_layer_helper.mm PROPERTIES
    COMPILE_FLAGS "-x objective-c++"
  )
  
  find_library(COCOA_LIBRARY Cocoa REQUIRED)
  find_library(QUARTZCORE_LIBRARY QuartzCore REQUIRED)
  set(PLATFORM_LIBS ${COCOA_LIBRARY} ${QUARTZCORE_LIBRARY})
endif()

add_executable(julia_fractal_webgpu ${SOURCE_FILES})

# Link with webgpu target from WebGPU-distribution and GLFW
target_link_libraries(julia_fractal_webgpu webgpu glfw ${PLATFORM_LIBS})

# Debug prints
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
