cmake_minimum_required(VERSION 3.20)
project(VkTutorial)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Vulkan REQUIRED)


add_executable(VkTutorial main.cpp)


set(GLFW_DIR "glfw")

target_include_directories(VkTutorial PUBLIC ${GLFW_DIR}/include)
target_include_directories(VkTutorial PUBLIC $ENV{VULKAN_SDK}/Include)

target_link_directories(VkTutorial PUBLIC ${GLFW_DIR}/lib-mingw-w64)

target_link_libraries(
        VkTutorial PUBLIC glfw3 Vulkan::Vulkan
)


# -------- Shader build --------
find_program(GLSLC glslc HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/Bin32"
)
if (NOT GLSLC)
    message(FATAL_ERROR "glslc not found. Install Vulkan SDK and ensure glslc is in PATH (or set VULKAN_SDK).")
endif()

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SPV_DIR    ${CMAKE_CURRENT_BINARY_DIR}/shaders)

file(MAKE_DIRECTORY ${SPV_DIR})

set(SHADERS
        vert.vert
        frag.frag
)

set(SPV_OUTPUTS "")
foreach(S ${SHADERS})
    set(SRC ${SHADER_DIR}/${S})
    set(SPV ${SPV_DIR}/${S}.spv)

    add_custom_command(
            OUTPUT ${SPV}
            COMMAND ${GLSLC} -O ${SRC} -o ${SPV}
            DEPENDS ${SRC}
            COMMENT "Compiling shader: ${S}"
            VERBATIM
    )
    list(APPEND SPV_OUTPUTS ${SPV})
endforeach()

add_custom_target(shaders DEPENDS ${SPV_OUTPUTS})
add_dependencies(VkTutorial shaders)
