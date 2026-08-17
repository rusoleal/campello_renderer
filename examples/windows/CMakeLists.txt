cmake_minimum_required(VERSION 3.22.1)
project(campello_renderer_windows)

set(CMAKE_CXX_STANDARD 20)

# ---------------------------------------------------------------------------
# campello_renderer (parent project — always available when this CMakeLists
# is included via the root CMakeLists.txt)
# ---------------------------------------------------------------------------
if(NOT TARGET campello_renderer)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_BINARY_DIR}/campello_renderer)
endif()

include(${PROJECT_SOURCE_DIR}/../../dependencies/glfw.cmake)

# ---------------------------------------------------------------------------
# Application target
# ---------------------------------------------------------------------------
add_executable(campello_renderer_windows
    main.cpp
    Camera.hpp
)

target_link_libraries(campello_renderer_windows
    campello_renderer
    glfw
)

target_include_directories(campello_renderer_windows PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Auto-copy every shared-library dependency next to the exe on every build.
# Without this, cmake --build only updates DLLs inside their own target
# output directories (_deps/extern_campello_gpu-build/<config>/, etc.) —
# nothing puts a fresh copy next to campello_renderer_windows.exe itself, so
# running the exe after rebuilding just campello_gpu (or campello_renderer)
# silently launched against a stale DLL. This bit repeatedly during Windows/
# DirectX debugging (2026-08-16): a "reproduction" that was actually testing
# an already-fixed binary, and a shader change that appeared to do nothing
# because the old compiled shader was still embedded in the stale DLL.
# $<TARGET_FILE:...> resolves per-config automatically under multi-config
# generators (Debug/Release/RelWithDebInfo), so this works for all of them.
foreach(dep_target campello_renderer campello_gpu gltf campello_image)
    if(TARGET ${dep_target})
        add_custom_command(TARGET campello_renderer_windows POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:${dep_target}>
                $<TARGET_FILE_DIR:campello_renderer_windows>
            COMMENT "Copying ${dep_target} DLL next to campello_renderer_windows.exe"
        )
    endif()
endforeach()
