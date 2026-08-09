cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# Skip if glfw target is already defined (e.g., by parent project)
if(TARGET glfw)
    message(STATUS "glfw target already exists, skipping FetchContent")
    return()
endif()

include(FetchContent)

message(STATUS "Fetching glfw v3.4 from GitHub")
FetchContent_Declare(
        extern_glfw
        GIT_REPOSITORY https://github.com/glfw/glfw
        GIT_TAG        3.4
)

if(NOT extern_glfw_POPULATED)
    FetchContent_GetProperties(extern_glfw)

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)

    # This machine is a Wayland session without the full X11 extension dev
    # headers (Xrandr/Xinerama/Xcursor/Xi) GLFW's X11 backend needs to build —
    # only core libX11 is present. Building Wayland-only avoids that missing
    # dependency; GLFW vendors the protocol XML it needs (xdg-shell etc.), so
    # this needs no extra system package beyond the wayland-client/xkbcommon
    # dev headers already present. Flip these (and install the X11 extension
    # dev packages) for portability to X11-only Linux machines.
    set(GLFW_BUILD_WAYLAND ON  CACHE BOOL "" FORCE)
    set(GLFW_BUILD_X11     OFF CACHE BOOL "" FORCE)

    FetchContent_Populate(extern_glfw)
    add_subdirectory(
            ${extern_glfw_SOURCE_DIR}
            ${extern_glfw_BINARY_DIR}
            EXCLUDE_FROM_ALL
    )
endif()
