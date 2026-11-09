# campello_renderer_add_shader_program(): compiles an app shader program for
# the backends whose compilers are available and embeds the binaries in a
# generated header with a ready ShaderProgramDesc
# (docs/design/programmable-pipelines.md, phase 3).
#
#   campello_renderer_add_shader_program(
#       TARGET my_app                       # gets the dependency + include dir
#       NAME   element_id                   # namespace and file names
#       METAL  shaders/element_id.metal     # vertexMain + fragmentMain (Apple)
#       VULKAN_VERTEX   shaders/element_id.vert   # SPIR-V via glslc or glslangValidator
#       VULKAN_FRAGMENT shaders/element_id.frag
#       DIRECTX shaders/element_id.hlsl     # vertexMain + pixelMain via dxc, if found
#       SLANG  shaders/element_id.slang     # one source for every backend (preferred):
#                                           # vertexMain + fragmentMain, compiled with slangc
#                                           # (CAMPELLO_SLANGC, or $SLANG_DIR/bin) to Metal,
#                                           # SPIR-V, HLSL (then dxc) and WGSL; overrides the
#                                           # per-backend sources when slangc is found
#       VERTEX_READS "semanticBit(VertexSemantic::position) | semanticBit(VertexSemantic::custom0)")
#
# Generates <build>/campello_shaders/<NAME>_shaders.h:
#   #include "<NAME>_shaders.h"
#   auto program = renderer.registerProgram(<NAME>_shaders::desc());
# Backends without a compiled binary get an empty one (registerProgram()
# returns kInvalidProgram there). Shaders include the interface headers as
# "campello_renderer/shaders/program_interface.*" (-I <campello_renderer>/inc).

set(CAMPELLO_RENDERER_SHADER_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "")
set(CAMPELLO_RENDERER_SHADER_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../inc" CACHE INTERNAL "")

function(campello_renderer_add_shader_program)
    cmake_parse_arguments(ARG "" "TARGET;NAME;METAL;VULKAN_VERTEX;VULKAN_FRAGMENT;DIRECTX;SLANG;VERTEX_READS" "" ${ARGN})
    if(NOT ARG_TARGET OR NOT ARG_NAME)
        message(FATAL_ERROR "campello_renderer_add_shader_program: TARGET and NAME are required")
    endif()
    if(NOT ARG_VERTEX_READS)
        set(ARG_VERTEX_READS "semanticBit(VertexSemantic::position)")
    endif()
    set(out_dir "${CMAKE_CURRENT_BINARY_DIR}/campello_shaders")
    file(MAKE_DIRECTORY "${out_dir}")
    set(inc "${CAMPELLO_RENDERER_SHADER_INCLUDE_DIR}")
    file(GLOB interface_headers "${inc}/campello_renderer/shaders/program_interface.*")
    set(binaries "")
    set(embed_args "")
    set(dx_pixel_entry pixelMain)

    # Slang: one source -> Metal source, SPIR-V, HLSL, WGSL (then the
    # per-backend steps below compile the Metal/HLSL as usual).
    set(use_slang OFF)
    if(ARG_SLANG)
        find_program(CAMPELLO_SLANGC slangc HINTS "$ENV{SLANG_DIR}/bin")
        if(CAMPELLO_SLANGC)
            set(use_slang ON)
        else()
            message(STATUS "campello_renderer_add_shader_program(${ARG_NAME}): slangc not found "
                           "(set CAMPELLO_SLANGC); using the per-backend sources")
        endif()
    endif()
    if(use_slang)
        get_filename_component(slang_src "${ARG_SLANG}" ABSOLUTE)
        set(slang_deps "${slang_src}" ${interface_headers})
        set(slang_cmd "${CAMPELLO_SLANGC}" "${slang_src}" -I "${inc}")
        if(APPLE)
            set(generated_metal "${out_dir}/${ARG_NAME}.slang.metal")
            add_custom_command(OUTPUT "${generated_metal}"
                COMMAND ${slang_cmd} -target metal -o "${generated_metal}"
                DEPENDS ${slang_deps}
                COMMENT "Slang -> Metal: ${ARG_NAME}"
                VERBATIM)
            set(ARG_METAL "${generated_metal}")
        endif()
        foreach(stage vertex fragment)
            if(stage STREQUAL "vertex")
                set(entry vertexMain)
                set(var VULKAN_VERTEX)
                set(flip -fvk-invert-y)  # Vulkan clip-space Y (see program_interface.slang)
            else()
                set(entry fragmentMain)
                set(var VULKAN_FRAGMENT)
                set(flip "")
            endif()
            set(spv "${out_dir}/${ARG_NAME}_${stage}.slang.spv")
            add_custom_command(OUTPUT "${spv}"
                COMMAND ${slang_cmd} -target spirv ${flip} -entry ${entry} -stage ${stage} -o "${spv}"
                DEPENDS ${slang_deps}
                COMMENT "Slang -> SPIR-V: ${ARG_NAME} (${stage})"
                VERBATIM)
            list(APPEND binaries "${spv}")
            list(APPEND embed_args "-D${var}=${spv}")
        endforeach()
        set(ARG_VULKAN_VERTEX "")
        set(ARG_VULKAN_FRAGMENT "")
        set(generated_hlsl "${out_dir}/${ARG_NAME}.slang.hlsl")
        add_custom_command(OUTPUT "${generated_hlsl}"
            COMMAND ${slang_cmd} -target hlsl -o "${generated_hlsl}"
            DEPENDS ${slang_deps}
            COMMENT "Slang -> HLSL: ${ARG_NAME}"
            VERBATIM)
        set(ARG_DIRECTX "${generated_hlsl}")
        set(dx_pixel_entry fragmentMain)
        set(wgsl "${out_dir}/${ARG_NAME}.wgsl")
        add_custom_command(OUTPUT "${wgsl}"
            COMMAND ${slang_cmd} -target wgsl -o "${wgsl}"
            DEPENDS ${slang_deps}
            COMMENT "Slang -> WGSL: ${ARG_NAME}"
            VERBATIM)
        list(APPEND binaries "${wgsl}")
        list(APPEND embed_args "-DWGSL=${wgsl}" "-DDIRECTX_FRAGMENT_ENTRY=fragmentMain")
    endif()

    # Metal (Apple): .metal -> .air -> .metallib
    if(APPLE AND ARG_METAL)
        if(IOS)
            set(sdk iphoneos)
        else()
            set(sdk macosx)
        endif()
        get_filename_component(src "${ARG_METAL}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        set(air "${out_dir}/${ARG_NAME}.air")
        set(lib "${out_dir}/${ARG_NAME}.metallib")
        add_custom_command(OUTPUT "${lib}"
            COMMAND xcrun -sdk ${sdk} metal -c "${src}" -I "${inc}" -o "${air}"
            COMMAND xcrun -sdk ${sdk} metallib "${air}" -o "${lib}"
            DEPENDS "${src}" ${interface_headers}
            COMMENT "Compiling Metal shader program ${ARG_NAME}"
            VERBATIM)
        list(APPEND binaries "${lib}")
        list(APPEND embed_args "-DMETAL=${lib}")
    endif()

    # Vulkan: GLSL -> SPIR-V (glslc, else glslangValidator)
    if(ARG_VULKAN_VERTEX AND ARG_VULKAN_FRAGMENT)
        find_program(CAMPELLO_GLSLC glslc)
        find_program(CAMPELLO_GLSLANG glslangValidator)
        foreach(stage vert frag)
            if(stage STREQUAL "vert")
                get_filename_component(src "${ARG_VULKAN_VERTEX}" ABSOLUTE)
                set(var VULKAN_VERTEX)
            else()
                get_filename_component(src "${ARG_VULKAN_FRAGMENT}" ABSOLUTE)
                set(var VULKAN_FRAGMENT)
            endif()
            set(spv "${out_dir}/${ARG_NAME}_${stage}.spv")
            if(CAMPELLO_GLSLC)
                set(cmd "${CAMPELLO_GLSLC}" -fshader-stage=${stage} "${src}" -I "${inc}" -o "${spv}")
            elseif(CAMPELLO_GLSLANG)
                set(cmd "${CAMPELLO_GLSLANG}" -V -S ${stage} "${src}" "-I${inc}" -o "${spv}")
            else()
                break()
            endif()
            add_custom_command(OUTPUT "${spv}"
                COMMAND ${cmd}
                DEPENDS "${src}" ${interface_headers}
                COMMENT "Compiling SPIR-V shader ${ARG_NAME} (${stage})"
                VERBATIM)
            list(APPEND binaries "${spv}")
            list(APPEND embed_args "-D${var}=${spv}")
        endforeach()
    endif()

    # DirectX: HLSL -> DXIL (dxc, if found)
    if(ARG_DIRECTX)
        find_program(CAMPELLO_DXC dxc)
        if(CAMPELLO_DXC)
            get_filename_component(src "${ARG_DIRECTX}" ABSOLUTE)
            set(vs "${out_dir}/${ARG_NAME}_vs.dxil")
            set(ps "${out_dir}/${ARG_NAME}_ps.dxil")
            add_custom_command(OUTPUT "${vs}" "${ps}"
                COMMAND "${CAMPELLO_DXC}" -T vs_6_0 -E vertexMain -I "${inc}" -Fo "${vs}" "${src}"
                COMMAND "${CAMPELLO_DXC}" -T ps_6_0 -E ${dx_pixel_entry} -I "${inc}" -Fo "${ps}" "${src}"
                DEPENDS "${src}" ${interface_headers}
                COMMENT "Compiling DXIL shader program ${ARG_NAME}"
                VERBATIM)
            list(APPEND binaries "${vs}" "${ps}")
            list(APPEND embed_args "-DDIRECTX_VERTEX=${vs}" "-DDIRECTX_FRAGMENT=${ps}")
        endif()
    endif()

    set(header "${out_dir}/${ARG_NAME}_shaders.h")
    add_custom_command(OUTPUT "${header}"
        COMMAND "${CMAKE_COMMAND}" "-DNAME=${ARG_NAME}" "-DOUTPUT=${header}"
                "-DVERTEX_READS=${ARG_VERTEX_READS}" ${embed_args}
                -P "${CAMPELLO_RENDERER_SHADER_CMAKE_DIR}/campello_embed_shaders.cmake"
        DEPENDS ${binaries} "${CAMPELLO_RENDERER_SHADER_CMAKE_DIR}/campello_embed_shaders.cmake"
        COMMENT "Embedding shader program ${ARG_NAME}"
        VERBATIM)
    add_custom_target(${ARG_NAME}_shaders DEPENDS "${header}")
    add_dependencies(${ARG_TARGET} ${ARG_NAME}_shaders)
    target_include_directories(${ARG_TARGET} PRIVATE "${out_dir}")
endfunction()
