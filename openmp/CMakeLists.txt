find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (PERFORMANCE)
    add_definitions(-DPERFORMANCE=1)
    add_executable(omp_test omp_test.c)
elseif(DEBUG)
    add_definitions(-DDEBUG=1)
endif ()

if(ELLPACK)
    add_definitions(-DELLPACK=1)
endif()

set(UTILS_SRC ${UTILS_SRC} omp_utils.c)

if (MPI)
    find_package(MPI REQUIRED)
    if(MPI_FOUND)
        include_directories(${MPI_INCLUDE_PATH})
        set(MPI_CXX_COMPILER ${MPI_BIN_PATH}/mpicxx)
        set(MPI_C_COMPILER ${MPI_BIN_PATH}/mpicc)
        set(MPIEXEC_EXECUTABLE ${MPI_BIN_PATH}/mpiexec)

        add_executable(ompi_spmm ompi_spmm.c ${UTILS_SRC})

        target_compile_options(ompi_spmm PRIVATE ${MPI_CXX_COMPILE_FLAGS})
        target_link_libraries(ompi_spmm ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
    else()
        message(SEND_ERROR "This application cannot compile without MPI")
    endif()
else()
    add_executable(omp_spmm omp_spmm.c ${UTILS_SRC})
endif()