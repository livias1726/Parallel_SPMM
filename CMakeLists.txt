project(SCPA_Project)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_C_STANDARD 11)

if (PERFORMANCE)
    add_definitions(-DPERFORMANCE=1)
endif (PERFORMANCE)

add_subdirectory(serial)
add_subdirectory(openmp)
#add_subdirectory(cuda)
add_subdirectory(lib)
add_subdirectory(utils)

add_executable(test test.c)