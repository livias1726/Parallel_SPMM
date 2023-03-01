enable_language(CUDA)

set (CC 75)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)

set(CMAKE_CXX_FLAGS "-O3")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DSM_${CC};-arch=sm_${CC};-lineinfo;-Xcompiler=${CMAKE_CXX_FLAGS};-Xptxas=-v)

if (PERFORMANCE)
    add_definitions(-DPERFORMANCE=1)
    #add_executable(cu_test cu_test.c)
elseif(DEBUG)
    add_definitions(-DDEBUG=1)
endif ()

if(ELLPACK)
    add_definitions(-DELLPACK=1)
endif()

set(UTILS_SRC ${UTILS_SRC} cu_utils.cu)
add_executable(cu_spmm cu_spmm.cu ${UTILS_SRC})
