set (CC 75)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)

set(CMAKE_CXX_FLAGS "-O3")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DSM_${CC};-arch=sm_${CC};-lineinfo;-Xcompiler=${CMAKE_CXX_FLAGS};-Xptxas=-v)

enable_language(CUDA)

set(UTILS_SRC ${UTILS_SRC} cu_utils.cu)

if (QUERY)
    cuda_add_executable(deviceQuery deviceQuery.cpp)
elseif(PERFORMANCE)
    #add_executable(cu_test cu_test.cu ${UTILS_SRC})
else ()
    add_executable(cu_spmm cu_spmm.cu ${UTILS_SRC})
endif()