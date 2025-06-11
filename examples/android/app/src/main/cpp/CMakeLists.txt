# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

cmake_minimum_required(VERSION 3.22.1)

project("test")

#include(ExternalProject)

#set(campello_DIR "../../../../../../src/gpu/vulkan")
#find_package(campello)
#include(../../../../../../src/gpu/vulkan/CMakeLists.txt)

#include_directories(../../../../../../inc)

#find_package(Vulkan)
#message(STATUS "Vulkan FOUND = ${Vulkan_FOUND}")
#message(STATUS "Vulkan Include = ${Vulkan_INCLUDE_DIR}")
#message(STATUS "Vulkan Lib = ${Vulkan_LIBRARY}")

include(${PROJECT_SOURCE_DIR}/campello.cmake)

# Creates your game shared library. The name must be the same as the
# one used for loading in your Kotlin/Java or AndroidManifest.txt files.
add_library(test SHARED
        main.cpp
        AndroidOut.cpp
        Renderer.cpp
        Shader.cpp
        TextureAsset.cpp
        Utility.cpp
        #../../../../../../src/gpu/vulkan/device.cpp
)

# Searches for a package provided by the game activity dependency
find_package(game-activity REQUIRED CONFIG)



# Configure libraries CMake uses to link your target library.
target_link_libraries(test
        # The game activity
        game-activity::game-activity

        #Vulkan::Vulkan
        campello

        # EGL and other dependent libraries required for drawing
        # and interacting with Android system
        EGL
        GLESv3
        jnigraphics
        android
        log)