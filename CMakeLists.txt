project(SCPA_Project)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_C_STANDARD 11)

add_subdirectory(parallel)
add_subdirectory(serial)
add_subdirectory(lib)
add_subdirectory(utils)