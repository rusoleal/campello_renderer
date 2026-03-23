cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

include(FetchContent)

set(WEBP_BUILD_ANIM_UTILS OFF)
set(WEBP_BUILD_CWEBP      OFF)
set(WEBP_BUILD_DWEBP      OFF)
set(WEBP_BUILD_GIF2WEBP   OFF)
set(WEBP_BUILD_IMG2WEBP   OFF)
set(WEBP_BUILD_VWEBP      OFF)
set(WEBP_BUILD_WEBPINFO   OFF)
set(WEBP_BUILD_WEBPMUX    OFF)
set(WEBP_BUILD_EXTRAS     OFF)

FetchContent_Declare(
        extern_webp
        GIT_REPOSITORY https://chromium.googlesource.com/webm/libwebp
        GIT_TAG        v1.5.0
)

if(NOT extern_webp_POPULATED)
    FetchContent_GetProperties(extern_webp)

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    FetchContent_Populate(extern_webp)
    include_directories(${extern_webp_SOURCE_DIR}/src)
    add_subdirectory(
            ${extern_webp_SOURCE_DIR}
            ${extern_webp_BINARY_DIR}
            EXCLUDE_FROM_ALL
    )
endif()
