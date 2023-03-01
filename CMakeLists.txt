project(SCPA_Project)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_C_STANDARD 11)
set(UTILS_SRC ../lib/mmio.c ../utils/utils.c ../utils/serial.c ../utils/storage.c ../utils/error.c ../utils/io.c)

if (OPENMP)
    add_subdirectory(openmp)
else()
    add_subdirectory(cuda)
endif ()


