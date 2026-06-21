cmake_minimum_required(VERSION 3.20)
project(tiramisu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

option(TIRAMISU_BUILD_TESTS "Build unit tests" ON)
option(TIRAMISU_BUILD_EXAMPLES "Build example programs" ON)
option(TIRAMISU_BUILD_BENCH "Build microbenchmarks" OFF)
option(TIRAMISU_ENABLE_SANITIZERS "Build with ASan/UBSan in Debug builds" ON)

# Shared interface targets

add_library(tiramisu_warnings INTERFACE)
target_compile_options(tiramisu_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow
)

add_library(tiramisu_sanitizers INTERFACE)
if (TIRAMISU_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(tiramisu_sanitizers INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer
  )
  target_link_options(tiramisu_sanitizers INTERFACE
    -fsanitize=address,undefined
  )
endif()

# Library modules

add_subdirectory(core)
add_subdirectory(autograd)
add_subdirectory(ops)

# Tests, examples, and bench

if (TIRAMISU_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
  )

  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(tests)
endif()

if (TIRAMISU_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if (TIRAMISU_BUILD_BENCH)
  add_subdirectory(bench)
endif()