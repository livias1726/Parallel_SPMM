project(SCPA_Project)
cmake_minimum_required(VERSION 2.8)

if (PERFORMANCE)
    add_definitions(-DPERFORMANCE=1)
endif (PERFORMANCE)

set(CMAKE_C_STANDARD 11)

add_subdirectory(parallel)
add_subdirectory(serial)
add_subdirectory(lib)
add_subdirectory(utils)

add_executable(test test.c)