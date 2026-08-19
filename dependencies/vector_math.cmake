cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# Skip if vector_math target is already defined (e.g., by parent project)
if(TARGET vector_math)
    message(STATUS "vector_math target already exists, skipping FetchContent")
    return()
endif()

include(FetchContent)

# Declared explicitly (and included before gltf.cmake, whose own transitive
# vector_math.cmake still pins the older v0.3.5 and guards on `if(TARGET
# vector_math)`) so campello_renderer picks up the Matrix4::lookAt() mirrored-
# camera fix from vector_math v0.6.0 instead of silently inheriting whatever
# older version gltf happens to pin. See Renderer::buildDefaultCameraView()'s
# doc comment in src/campello_renderer.cpp for the bug this fixes upstream.
message(STATUS "Fetching vector_math v0.6.0 from GitHub")
FetchContent_Declare(
        extern_vector_math
        GIT_REPOSITORY https://github.com/rusoleal/vector_math
        GIT_TAG        v0.6.0
)

if(NOT extern_vector_math_POPULATED)
    FetchContent_GetProperties(extern_vector_math)

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    FetchContent_Populate(extern_vector_math)
    include_directories(${extern_vector_math_SOURCE_DIR}/inc)
    add_subdirectory(
            ${extern_vector_math_SOURCE_DIR}
            ${extern_vector_math_BINARY_DIR}
            EXCLUDE_FROM_ALL
    )
endif()
