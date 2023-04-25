project(SCPA_Project)
cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_STANDARD 11)
set(UTILS_SRC ../lib/mmio.c ../utils/utils.c ../utils/serial.c ../utils/storage.c ../utils/io.c)

if (QUERY)
    add_definitions(-DQUERY=1)
endif()

if (SAVE)
    add_definitions(-DSAVE=1)
endif ()

if(ELLPACK)
    add_definitions(-DELLPACK=1)
endif()

if (OPENMP)
    add_subdirectory(openmp)
else()
    add_subdirectory(cuda)
endif ()


