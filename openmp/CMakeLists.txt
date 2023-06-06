find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O3 -mavx -mavx2 -mavx512f")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -fopenmp -mavx -mavx2 -mavx512f")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(UTILS_SRC ${UTILS_SRC} omp_utils.c)
if (ELLPACK)
    set(UTILS_SRC ${UTILS_SRC} omp_ell.c)
else()
    set(UTILS_SRC ${UTILS_SRC} omp_csr.c)
endif()

add_executable(omp_spmm omp_spmm.c ${UTILS_SRC})

if (PERFORMANCE)
    add_executable(omp_test omp_test.c)
endif ()

