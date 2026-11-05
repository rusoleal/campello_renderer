cmake_minimum_required(VERSION 3.22.1)

set(LOGO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../xbox_probe/Assets)
set(PACKAGE_ASSETS
    ${LOGO_DIR}/Logo150.png
    ${LOGO_DIR}/Logo44.png
    ${LOGO_DIR}/StoreLogo.png
    ${LOGO_DIR}/SplashScreen.png
    ${CMAKE_CURRENT_SOURCE_DIR}/../android/app/src/main/assets/DamagedHelmet.glb
    ${CMAKE_CURRENT_SOURCE_DIR}/../../resources/environments/default_environment.hdr
)

add_executable(campello_renderer_uwp WIN32
    main.cpp
    Package.appxmanifest
    ${PACKAGE_ASSETS}
)

set_source_files_properties(${PACKAGE_ASSETS} PROPERTIES
    VS_DEPLOYMENT_CONTENT 1
    VS_DEPLOYMENT_LOCATION "Assets"
)

target_link_libraries(campello_renderer_uwp PRIVATE campello_renderer WindowsApp.lib)

target_include_directories(campello_renderer_uwp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../windows # Camera.hpp
)

target_compile_definitions(campello_renderer_uwp PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

set_target_properties(campello_renderer_uwp PROPERTIES
    VS_GLOBAL_AppxPackageSigningEnabled false
    VS_GLOBAL_GenerateAppxPackageOnBuild true
    VS_GLOBAL_AppxBundle Never
)
