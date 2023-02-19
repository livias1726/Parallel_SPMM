project(SCPA_Project)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_C_STANDARD 11)

if (OPENMP)
    add_subdirectory(openmp)
endif(OPENMP)

if (CUDA)
    add_subdirectory(cuda)
endif (CUDA)


