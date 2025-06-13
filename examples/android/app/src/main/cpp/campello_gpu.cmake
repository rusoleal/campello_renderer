cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

include(FetchContent)

FetchContent_Declare(
        extern_campello_gpu
        SOURCE_DIR ../../../../../../../src/gpu/vulkan_android
        #GIT_REPOSITORY https://github.com/zeux/pugixml.git
        #GIT_TAG        v1.13
)

FetchContent_GetProperties(extern_campello_gpu)

if(NOT extern_campello_gpu_POPULATED)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    message(STATUS "Fetching campello_gpu...")
    FetchContent_Populate(extern_campello_gpu)
    message(STATUS ${extern_campello_gpu_SOURCE_DIR})
    include_directories(${extern_campello_gpu_SOURCE_DIR}/../../../inc)
    add_subdirectory(
            ${extern_campello_gpu_SOURCE_DIR}
            ${extern_campello_gpu_BINARY_DIR}
            EXCLUDE_FROM_ALL
    )
endif()
