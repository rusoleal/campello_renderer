cmake_minimum_required(VERSION 3.22.1)
project(campello_renderer_macos)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD 20)

# ---------------------------------------------------------------------------
# campello_renderer (parent project — always available when this CMakeLists
# is included via the root CMakeLists.txt)
# ---------------------------------------------------------------------------
if(NOT TARGET campello_renderer)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_BINARY_DIR}/campello_renderer)
endif()

# ---------------------------------------------------------------------------
# Application target
# ---------------------------------------------------------------------------
add_executable(campello_renderer_macos MACOSX_BUNDLE
    main.mm
    AppDelegate.h
    AppDelegate.mm
    ViewController.h
    ViewController.mm
    Camera.hpp
)

set_target_properties(campello_renderer_macos PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist"
    MACOSX_BUNDLE_EXECUTABLE_NAME "campello_renderer_macos"
)

target_link_libraries(campello_renderer_macos
    campello_renderer
    "-framework Cocoa"
    "-framework Metal"
    "-framework MetalKit"
    "-framework QuartzCore"
)

target_include_directories(campello_renderer_macos PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
