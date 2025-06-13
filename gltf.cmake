cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

include(FetchContent)

FetchContent_Declare(
        extern_gltf
        #SOURCE_DIR ../src/gpu/vulkan_android
        GIT_REPOSITORY https://github.com/rusoleal/gltf
        GIT_TAG        v0.0.6
)


if(NOT extern_gltf_POPULATED)
    FetchContent_GetProperties(extern_gltf)
    
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    message(STATUS "Fetching gltf...")
    message(STATUS ${extern_gltf_SOURCE_DIR})
    message(STATUS "pepito2")
    FetchContent_Populate(extern_gltf)
    include_directories(${extern_gltf_SOURCE_DIR}/inc)
    add_subdirectory(
            ${extern_gltf_SOURCE_DIR}
            ${extern_gltf_BINARY_DIR}
            EXCLUDE_FROM_ALL
    )
endif()
