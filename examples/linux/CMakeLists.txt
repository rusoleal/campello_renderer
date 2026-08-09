cmake_minimum_required(VERSION 3.22.1)
project(campello_renderer_linux)

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
add_executable(campello_renderer_linux
    main.cpp
    Camera.hpp
)

target_link_libraries(campello_renderer_linux
    campello_renderer
    glfw
)

target_include_directories(campello_renderer_linux PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
