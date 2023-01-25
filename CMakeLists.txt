project(SCPA_Project C)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_C_STANDARD 11)

#add_subdirectory(parallel)
add_subdirectory(serial)
add_subdirectory(utils)
add_subdirectory(lib)