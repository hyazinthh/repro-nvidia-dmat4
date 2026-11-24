cmake_minimum_required(VERSION 3.19)
project(dmat4_repro CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

find_program(SHADER_COMPILER
    NAMES glslc glslangValidator
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" "${Vulkan_INCLUDE_DIR}/../Bin" "${Vulkan_INCLUDE_DIR}/../bin")
if(NOT SHADER_COMPILER)
    message(FATAL_ERROR "glslc or glslangValidator not found; install the Vulkan SDK")
endif()

get_filename_component(SHADER_COMPILER_NAME "${SHADER_COMPILER}" NAME_WE)
set(SPV "${CMAKE_CURRENT_BINARY_DIR}/copy.spv")
if(SHADER_COMPILER_NAME STREQUAL "glslc")
    set(SHADER_ARGS -fshader-stage=comp "${CMAKE_CURRENT_SOURCE_DIR}/copy.comp" -o "${SPV}")
else()
    set(SHADER_ARGS -V "${CMAKE_CURRENT_SOURCE_DIR}/copy.comp" -o "${SPV}")
endif()

add_custom_command(
    OUTPUT "${SPV}"
    COMMAND "${SHADER_COMPILER}" ${SHADER_ARGS}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/copy.comp"
    COMMENT "Compiling copy.comp")

add_executable(dmat4_repro main.cpp "${SPV}")
target_link_libraries(dmat4_repro PRIVATE Vulkan::Vulkan)
target_compile_definitions(dmat4_repro PRIVATE SPV_PATH="${SPV}")
