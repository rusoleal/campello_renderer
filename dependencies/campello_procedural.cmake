cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# Skip if campello_procedural target is already defined (e.g., by parent project)
if(TARGET campello_procedural)
    message(STATUS "campello_procedural target already exists, skipping FetchContent")
    return()
endif()

include(FetchContent)
# CAMPELLO_PROCEDURAL_SOURCE=<checkout>: build against a local checkout.
if(DEFINED CAMPELLO_PROCEDURAL_SOURCE AND EXISTS "${CAMPELLO_PROCEDURAL_SOURCE}/CMakeLists.txt")
    message(STATUS "Using local campello_procedural: ${CAMPELLO_PROCEDURAL_SOURCE}")
    FetchContent_Declare(extern_campello_procedural SOURCE_DIR ${CAMPELLO_PROCEDURAL_SOURCE})
else()
    message(STATUS "Fetching campello_procedural v0.1.0 from GitHub")
    FetchContent_Declare(
            extern_campello_procedural
            GIT_REPOSITORY https://github.com/rusoleal/campello_procedural
            GIT_TAG        v0.1.0
    )
endif()

if(NOT extern_campello_procedural_POPULATED)
    FetchContent_GetProperties(extern_campello_procedural)

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    FetchContent_Populate(extern_campello_procedural)
    add_subdirectory(
            ${extern_campello_procedural_SOURCE_DIR}
            ${extern_campello_procedural_BINARY_DIR}
            EXCLUDE_FROM_ALL
    )
endif()
