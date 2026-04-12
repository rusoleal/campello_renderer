cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# Skip if campello_image target is already defined (e.g., by parent project)
if(TARGET campello_image)
    message(STATUS "campello_image target already exists, skipping FetchContent")
    return()
endif()

include(FetchContent)

FetchContent_Declare(
        extern_campello_image
        GIT_REPOSITORY https://github.com/rusoleal/campello_image
        GIT_TAG        v0.3.1
)

if(NOT extern_campello_image_POPULATED)
    FetchContent_GetProperties(extern_campello_image)

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    FetchContent_Populate(extern_campello_image)
    include_directories(${extern_campello_image_SOURCE_DIR}/inc)
    add_subdirectory(
            ${extern_campello_image_SOURCE_DIR}
            ${extern_campello_image_BINARY_DIR}
            EXCLUDE_FROM_ALL
    )
endif()
