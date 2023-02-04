cmake_minimum_required(VERSION 2.8)
project(serial)

set(CMAKE_C_STANDARD 11)

add_executable(serial_mmp serial_mmp.c ../lib/mmio.c ../utils/utils.c)