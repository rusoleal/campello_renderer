# Intel Open Image Denoise (Apache-2.0), the path tracer's denoiser
# (PathTracer::denoise()): the official prebuilt release for the target,
# downloaded at configure time. Defines OIDN_FOUND and the OpenImageDenoise
# target. macOS (x86_64, arm64), Linux x86_64 and Windows x64 have builds.

set(OIDN_VERSION 2.5.1)
set(OIDN_FOUND FALSE)

set(_oidn_arch "${CMAKE_OSX_ARCHITECTURES}")
if(NOT _oidn_arch)
    set(_oidn_arch "${CMAKE_SYSTEM_PROCESSOR}")
endif()
if(APPLE AND NOT IOS AND _oidn_arch MATCHES "^(arm64|aarch64)$")
    set(_oidn_package arm64.macos.tar.gz)
    set(_oidn_hash 98e0aca8e7ab69e9f4f0191582a500fd8b0d9085d68662f18ab6469e934a6efd)
elseif(APPLE AND NOT IOS AND _oidn_arch MATCHES "^(x86_64|AMD64)$")
    set(_oidn_package x86_64.macos.tar.gz)
    set(_oidn_hash a4341639005a33ce0d944bd7c771ce34de1465be7eb3189b7c3b5b0064429832)
else()
    message(STATUS "Open Image Denoise: no prebuilt package for this target (${CMAKE_SYSTEM_NAME} ${_oidn_arch}); no denoiser")
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    extern_oidn
    URL https://github.com/RenderKit/oidn/releases/download/v${OIDN_VERSION}/oidn-${OIDN_VERSION}.${_oidn_package}
    URL_HASH SHA256=${_oidn_hash}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_GetProperties(extern_oidn)
if(NOT extern_oidn_POPULATED)
    FetchContent_Populate(extern_oidn)
endif()

find_package(OpenImageDenoise ${OIDN_VERSION} CONFIG
    PATHS ${extern_oidn_SOURCE_DIR}/lib/cmake
    NO_DEFAULT_PATH)
if(TARGET OpenImageDenoise)
    set(OIDN_FOUND TRUE)
    # Where the libraries are, for consumers bundling them with an app
    # (the core loads its device modules from its own directory).
    set(CAMPELLO_RENDERER_DENOISER_LIBRARY_DIR ${extern_oidn_SOURCE_DIR}/lib CACHE INTERNAL "")
    message(STATUS "Open Image Denoise ${OIDN_VERSION}: ${extern_oidn_SOURCE_DIR}")
endif()
