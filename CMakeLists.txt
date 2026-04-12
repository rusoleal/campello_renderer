cmake_minimum_required(VERSION 3.22.1)

project(campello_renderer VERSION 0.1.4)

configure_file(src/campello_renderer_config.h.in campello_renderer_config.h)

set(CMAKE_CXX_STANDARD 20)

include_directories(inc)

message(STATUS ${PROJECT_NAME})

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
)

# Enable Unity Build for faster compilation
set_target_properties(${PROJECT_NAME} PROPERTIES UNITY_BUILD ON)


if(ANDROID)
    set(PLATFORM_LIBS log)
endif()

target_link_libraries(${PROJECT_NAME}
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
                           )

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
    add_test(NAME ${PROJECT_NAME}_test COMMAND ${PROJECT_NAME}_test)
endif()