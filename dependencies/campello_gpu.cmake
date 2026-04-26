cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# Skip if campello_gpu target is already defined (e.g., by parent project)
if(TARGET campello_gpu)
    message(STATUS "campello_gpu target already exists, skipping FetchContent")
    return()
endif()

# Use local campello_gpu for rapid iteration, otherwise fetch from GitHub.
set(CAMPELLO_GPU_LOCAL_PATH "/Users/rubenleal/Projects/campello_gpu")

if(EXISTS "${CAMPELLO_GPU_LOCAL_PATH}/CMakeLists.txt")
    message(STATUS "Using local campello_gpu: ${CAMPELLO_GPU_LOCAL_PATH}")
    include(FetchContent)
    FetchContent_Declare(
            extern_campello_gpu
            SOURCE_DIR ${CAMPELLO_GPU_LOCAL_PATH}
    )
else()
    message(STATUS "Fetching campello_gpu v0.12.0 from GitHub")
    include(FetchContent)
    FetchContent_Declare(
            extern_campello_gpu
            GIT_REPOSITORY https://github.com/rusoleal/campello_gpu
            GIT_TAG        v0.12.0
    )
endif()

if(NOT extern_campello_gpu_POPULATED)
    FetchContent_GetProperties(extern_campello_gpu)

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    FetchContent_Populate(extern_campello_gpu)
    include_directories(${extern_campello_gpu_SOURCE_DIR}/inc)
    add_subdirectory(
            ${extern_campello_gpu_SOURCE_DIR}
            ${extern_campello_gpu_BINARY_DIR}
            EXCLUDE_FROM_ALL
    )
endif()
