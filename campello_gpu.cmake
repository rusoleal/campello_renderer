cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

include(FetchContent)

FetchContent_Declare(
        extern_campello_gpu
        #SOURCE_DIR ../gpu/vulkan_android
        GIT_REPOSITORY https://github.com/rusoleal/campello_gpu
        GIT_TAG        v0.0.13
)

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
