cmake_minimum_required(VERSION 3.22.1)

project(campello_renderer VERSION 0.0.3)

configure_file(src/campello_renderer_config.h.in campello_renderer_config.h)

set(CMAKE_CXX_STANDARD 17)

include_directories(inc)

message(STATUS ${PROJECT_NAME})

include(${PROJECT_SOURCE_DIR}/campello_gpu.cmake)
set(DEPENDENCIES
    ${DEPENDENCIES}
    campello_gpu
)

include(${PROJECT_SOURCE_DIR}/gltf.cmake)
set(DEPENDENCIES
    ${DEPENDENCIES}
    gltf
)

add_library(${PROJECT_NAME} SHARED
    src/campello_renderer.cpp
)


target_link_libraries(${PROJECT_NAME}
    ${DEPENDENCIES}
    log
)

target_include_directories(${PROJECT_NAME} PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )