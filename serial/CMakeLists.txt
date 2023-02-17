cmake_minimum_required(VERSION 2.8)
project(serial)

set(CMAKE_C_STANDARD 11)

if (AUDIT)
    add_definitions(-DAUDIT=1)
endif (AUDIT)

add_executable(s_spmm s_spmm.c ../lib/mmio.c ../utils/utils.c)