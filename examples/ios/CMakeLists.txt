cmake_minimum_required(VERSION 3.22.1)
project(campello_renderer_ios)

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
add_executable(campello_renderer_ios MACOSX_BUNDLE
    main.mm
    AppDelegate.h
    AppDelegate.mm
    ViewController.h
    ViewController.mm
    Camera.hpp
)

set_target_properties(campello_renderer_ios PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist"
    MACOSX_BUNDLE_EXECUTABLE_NAME "campello_renderer_ios"
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "systems.leal.campello-renderer-ios"
    XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "16.0"
    XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY ""
    XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO"
    XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    XCODE_ATTRIBUTE_ENABLE_BITCODE "NO"
    # Add Frameworks/ to the runtime rpath so the embedded dylibs are found.
    XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "@executable_path/Frameworks"
)

# Post-build: copy all shared-library dependencies into the app bundle's
# Frameworks/ directory.  The dylibs already carry @rpath install names
# (CMAKE_MACOSX_RPATH is ON by default) so no install_name_tool fixup needed.
foreach(_dep campello_renderer campello_gpu gltf campello_image)
    add_custom_command(TARGET campello_renderer_ios POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_BUNDLE_DIR:campello_renderer_ios>/Frameworks"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:${_dep}>"
            "$<TARGET_BUNDLE_DIR:campello_renderer_ios>/Frameworks/$<TARGET_FILE_NAME:${_dep}>"
        COMMENT "Embedding ${_dep} into app bundle"
    )
endforeach()

target_link_libraries(campello_renderer_ios
    campello_renderer
    "-framework UIKit"
    "-framework Metal"
    "-framework MetalKit"
    "-framework QuartzCore"
    "-framework UniformTypeIdentifiers"
)

target_include_directories(campello_renderer_ios PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
