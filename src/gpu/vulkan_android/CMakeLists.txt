cmake_minimum_required(VERSION 3.22.1)

project("campello_gpu")

set(CMAKE_CXX_STANDARD 17)

include_directories(../../../inc)

find_package(Vulkan)
message(STATUS "Vulkan FOUND = ${Vulkan_FOUND}")
message(STATUS "Vulkan Include = ${Vulkan_INCLUDE_DIR}")
message(STATUS "Vulkan Lib = ${Vulkan_LIBRARY}")

message(STATUS ${PROJECT_NAME})

add_library(${PROJECT_NAME} SHARED
    context.cpp
    device.cpp
)

target_link_libraries(${PROJECT_NAME}
        Vulkan::Vulkan
        log)
