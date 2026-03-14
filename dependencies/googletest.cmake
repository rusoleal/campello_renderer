cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

include(FetchContent)

SET(BUILD_GMOCK OFF CACHE BOOL "DISABLE GMOCK")
SET(INSTALL_GTEST OFF CACHE BOOL "DISABLE INSTALL_GTEST")

FetchContent_Declare(
        extern_googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG        v1.17.0
)

if(NOT extern_googletest_POPULATED)
    FetchContent_GetProperties(extern_googletest)

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    FetchContent_Populate(extern_googletest)
    add_subdirectory(
            ${extern_googletest_SOURCE_DIR}
            ${extern_googletest_BINARY_DIR}
            EXCLUDE_FROM_ALL
    )
endif()
