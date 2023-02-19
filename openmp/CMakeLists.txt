project(openmp)

find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (PERFORMANCE)
    add_definitions(-DPERFORMANCE=1)
    add_executable(omp_test omp_test.c)
elseif(AUDIT)
    add_definitions(-DAUDIT=1)
endif ()

add_executable(omp_spmm omp_spmm.c ../lib/mmio.c ../utils/utils.c)
