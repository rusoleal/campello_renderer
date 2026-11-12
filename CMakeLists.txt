cmake_minimum_required(VERSION 3.22.1)

project(campello_renderer VERSION 0.18.0)

configure_file(src/campello_renderer_config.h.in campello_renderer_config.h)

set(CMAKE_CXX_STANDARD 20)

include_directories(inc)

# UWP differs from a desktop Windows build in ways the dependencies don't expect:
# gltf only defines _USE_MATH_DEFINES for the "Windows" platform, and the /sdl
# default turns fopen and deprecation warnings into errors.
if(CMAKE_SYSTEM_NAME STREQUAL "WindowsStore")
    add_compile_definitions(_USE_MATH_DEFINES)
    add_compile_options(/sdl-)
endif()

message(STATUS ${PROJECT_NAME})

include(${PROJECT_SOURCE_DIR}/dependencies/vector_math.cmake)
set(DEPENDENCIES
    ${DEPENDENCIES}
    vector_math
)

include(${PROJECT_SOURCE_DIR}/dependencies/campello_gpu.cmake)
set(DEPENDENCIES
    ${DEPENDENCIES}
    campello_gpu
)

include(${PROJECT_SOURCE_DIR}/dependencies/gltf.cmake)
set(DEPENDENCIES
    ${DEPENDENCIES}
    gltf
)

include(${PROJECT_SOURCE_DIR}/dependencies/campello_image.cmake)
set(DEPENDENCIES
    ${DEPENDENCIES}
    campello_image
)

add_library(${PROJECT_NAME} SHARED
    src/campello_renderer.cpp
    src/path_tracer.cpp
    src/animation.cpp
    src/procedural_texture_baker.cpp
)

# Path tracer denoising (Intel Open Image Denoise, prebuilt): on by default
# where the path tracer runs (macOS).
if(APPLE AND NOT IOS)
    set(_campello_denoiser_default ON)
else()
    set(_campello_denoiser_default OFF)
endif()
option(CAMPELLO_RENDERER_DENOISER "PathTracer::denoise() with Intel Open Image Denoise (downloads its release)" ${_campello_denoiser_default})
if(CAMPELLO_RENDERER_DENOISER)
    include(${PROJECT_SOURCE_DIR}/dependencies/oidn.cmake)
    if(OIDN_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE OpenImageDenoise)
        target_compile_definitions(${PROJECT_NAME} PRIVATE CAMPELLO_RENDERER_DENOISER)
    endif()
endif()

# Enable Unity Build for faster compilation
set_target_properties(${PROJECT_NAME} PROPERTIES UNITY_BUILD ON)

# Windows: none of this library's symbols carry __declspec(dllexport), so
# without this MSVC's linker builds the .dll but never emits the companion
# import .lib, which fails every consumer (test target, example apps) at
# link time with LNK1104 — see CHANGELOG.md's "Windows CI build-only" entry.
if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# ---------------------------------------------------------------------------
# Metal shader compilation (Apple platforms only)
# ---------------------------------------------------------------------------
if(APPLE)
    set(METAL_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/metal/default.metal)
    set(METAL_SHADER_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/metal_default.h)
    set(METAL_SHADER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/build_metal_shaders.sh)

    add_custom_command(
        OUTPUT ${METAL_SHADER_HEADER}
        COMMAND ${METAL_SHADER_SCRIPT}
        DEPENDS ${METAL_SHADER_SOURCE}
        COMMENT "Compiling Metal shaders for macOS / iOS..."
        VERBATIM
    )

    add_custom_target(metal_shaders DEPENDS ${METAL_SHADER_HEADER})
    add_dependencies(${PROJECT_NAME} metal_shaders)

    # Ensure CMake tracks the generated header for incremental builds.
    target_sources(${PROJECT_NAME} PRIVATE ${METAL_SHADER_HEADER})
endif()

if(ANDROID)
    set(PLATFORM_LIBS log)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${DEPENDENCIES}
    ${PLATFORM_LIBS}
)

# On iOS, ensure all shared-library dependencies advertise @rpath install names
# so they can be embedded in an app bundle at @executable_path/Frameworks.
if(IOS)
    foreach(_dep campello_gpu gltf campello_image ${PROJECT_NAME})
        if(TARGET ${_dep})
            set_target_properties(${_dep} PROPERTIES
                INSTALL_NAME_DIR "@rpath"
                BUILD_WITH_INSTALL_NAME_DIR TRUE
            )
        endif()
    endforeach()
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           # Public headers, so consumers don't have to add inc/ by hand
                           # (the include_directories(inc) above is directory-scoped).
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
                           $<INSTALL_INTERFACE:include>
                           )

if(CAMPELLO_RENDERER_ECS_SUPPORT)
    # ecs.hpp is header-only; consumers must have campello_core in their include path.
    install(FILES inc/campello_renderer/ecs.hpp inc/campello_renderer/culling.hpp
            DESTINATION include/campello_renderer)
endif()

if(APPLE AND NOT IOS)
    option(CAMPELLO_RENDERER_BUILD_MACOS_EXAMPLE "Build macOS example" OFF)
    if(CAMPELLO_RENDERER_BUILD_MACOS_EXAMPLE)
        add_subdirectory(examples/macos)
    endif()
endif()

if(IOS)
    option(CAMPELLO_RENDERER_BUILD_IOS_EXAMPLE "Build iOS example" OFF)
    if(CAMPELLO_RENDERER_BUILD_IOS_EXAMPLE)
        add_subdirectory(examples/ios)
    endif()
endif()

if(UNIX AND NOT APPLE AND NOT ANDROID)
    option(CAMPELLO_RENDERER_BUILD_LINUX_EXAMPLE "Build Linux example" OFF)
    if(CAMPELLO_RENDERER_BUILD_LINUX_EXAMPLE)
        add_subdirectory(examples/linux)
    endif()
endif()

if(WIN32)
    option(CAMPELLO_RENDERER_BUILD_WINDOWS_EXAMPLE "Build Windows example" OFF)
    if(CAMPELLO_RENDERER_BUILD_WINDOWS_EXAMPLE)
        add_subdirectory(examples/windows)
    endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "WindowsStore")
    option(CAMPELLO_RENDERER_BUILD_UWP_EXAMPLE "Build UWP example (Windows PC and Xbox Developer Mode)" OFF)
    if(CAMPELLO_RENDERER_BUILD_UWP_EXAMPLE)
        add_subdirectory(examples/uwp)
    endif()
endif()

# campello_renderer_add_shader_program(): app shader programs (program.hpp).
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CampelloRendererShaders.cmake)

option(CAMPELLO_RENDERER_ECS_SUPPORT "Install ecs.hpp (requires campello_core at compile time)" ON)

option(CAMPELLO_RENDERER_BUILD_TEST "Build unit tests" OFF)

if(CAMPELLO_RENDERER_BUILD_TEST)
    include(CTest)
    include(${PROJECT_SOURCE_DIR}/dependencies/googletest.cmake)

    add_executable(${PROJECT_NAME}_test
        test/main.cpp
    )
    set_target_properties(${PROJECT_NAME}_test PROPERTIES UNITY_BUILD ON)
    target_link_libraries(${PROJECT_NAME}_test
        ${PROJECT_NAME}
        gtest
    )
    # App shader program used by the program tests (registerProgram()).
    campello_renderer_add_shader_program(
        TARGET ${PROJECT_NAME}_test
        NAME element_id
        SLANG test/shaders/element_id.slang
        METAL test/shaders/element_id.metal
        WGSL test/shaders/element_id.wgsl
        VULKAN_VERTEX test/shaders/element_id.vert
        VULKAN_FRAGMENT test/shaders/element_id.frag
        VERTEX_READS "semanticBit(VertexSemantic::position) | semanticBit(VertexSemantic::custom0)")
    if(EMSCRIPTEN)
        # WebGPU: the suite is a web page, run in headless Chrome by
        # campello_gpu's tests/webgpu/run_in_browser.mjs (see test/webgpu/).
        set_target_properties(${PROJECT_NAME}_test PROPERTIES SUFFIX ".html")
        target_link_options(${PROJECT_NAME}_test PRIVATE
            -sALLOW_MEMORY_GROWTH=1
            -sSTACK_SIZE=4MB
            -sASYNCIFY_STACK_SIZE=262144
            --shell-file ${CMAKE_CURRENT_SOURCE_DIR}/test/webgpu/shell.html)
        set_property(TARGET ${PROJECT_NAME}_test APPEND PROPERTY LINK_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/test/webgpu/shell.html)
    else()
        add_test(NAME ${PROJECT_NAME}_test COMMAND ${PROJECT_NAME}_test)
    endif()

    # Windows: cmake --build only updates DLLs inside their own target output
    # directories (_deps/extern_campello_gpu-build/<config>/, etc.) — nothing
    # puts a fresh copy next to campello_renderer_test.exe itself, so running
    # (or ctest-ing) it fails with STATUS_DLL_NOT_FOUND. Mirrors the identical
    # POST_BUILD copy step in examples/windows/CMakeLists.txt.
    if(WIN32)
        foreach(dep_target ${PROJECT_NAME} campello_gpu gltf campello_image)
            if(TARGET ${dep_target})
                add_custom_command(TARGET ${PROJECT_NAME}_test POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_FILE:${dep_target}>
                        $<TARGET_FILE_DIR:${PROJECT_NAME}_test>
                    COMMENT "Copying ${dep_target} DLL next to ${PROJECT_NAME}_test.exe"
                )
            endif()
        endforeach()
    endif()
endif()