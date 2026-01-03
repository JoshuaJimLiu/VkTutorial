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