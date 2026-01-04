#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>

// =====================================================================================
// 0) 全局配置与常量
// =====================================================================================

// 初始窗口尺寸（注意：真正 swapchain extent 取决于 framebuffer 像素尺寸，可能 != 逻辑窗口尺寸）
const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;

// “同时在 GPU 上飞行的帧数”
// - CPU 可以提前录制下一帧并提交，而 GPU 正在执行上一帧，提升吞吐
// - 数值越大，吞吐可能更高，但输入->显示延迟也更大；2 常见折中
const int MAX_FRAMES_IN_FLIGHT = 2;

// 验证层：开发期开启，能抓住大量 Vulkan 的用法错误（同步/资源生命周期/参数合法性等）
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// 设备扩展：WSI 必需 VK_KHR_swapchain，否则无法创建 VkSwapchainKHR
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// NDEBUG 通常在 Release 构建定义；这里用它控制 validation layers 开关
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// =====================================================================================
// 1) Debug Utils 扩展函数封装
// =====================================================================================

// 这些函数属于 VK_EXT_debug_utils 扩展：不是 core API，需要用 vkGetInstanceProcAddr 动态拿函数指针。
// 好处：即使扩展不存在也能编译运行（会返回 VK_ERROR_EXTENSION_NOT_PRESENT 或 func==nullptr）。
VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    // 关键点：vkGetInstanceProcAddr 按 instance 查找扩展入口（loader/驱动支持才会返回非空）
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

    if (func != nullptr) {
        // 这里调用的其实是“扩展函数指针”，不是直接链接到的符号
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        // 未启用扩展 / 平台不支持：返回“扩展不存在”
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator)
{
    // destroy 同理：动态获取
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

// =====================================================================================
// 2) 一些辅助结构体：队列族与 swapchain 支持信息
// =====================================================================================

struct QueueFamilyIndices {
    // 用 optional：因为合法索引可能是 0，不能用 0/负数当“没找到”的哨兵值
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        // 对本教程而言：要能绘制（graphics queue）并能呈现（present queue）
        // 注意：present 支持必须通过 vkGetPhysicalDeviceSurfaceSupportKHR 查询，不能只看 flags
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    // capabilities：extent、min/max imageCount、transform、compositeAlpha 等硬约束
    VkSurfaceCapabilitiesKHR capabilities{};
    // formats：可用的 (format, colorspace) 组合
    std::vector<VkSurfaceFormatKHR> formats;
    // presentModes：FIFO / MAILBOX / IMMEDIATE 等
    std::vector<VkPresentModeKHR> presentModes;
};

// =====================================================================================
// 3) 主应用类
// =====================================================================================

class HelloTriangleApplication {
public:
    void run() {
        // 生命周期骨架：
        // 1) initWindow: 创建窗口、设置 resize 回调
        // 2) initVulkan: 创建 instance/device/swapchain/pipeline/command/sync
        // 3) mainLoop: 每帧 drawFrame（acquire→submit→present）
        // 4) cleanup: 释放所有 Vulkan/GLFW 资源
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    // -------------------- GLFW --------------------
    GLFWwindow* window = nullptr;

    // -------------------- Vulkan 基础对象 --------------------
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue  = VK_NULL_HANDLE;

    // -------------------- Swapchain “子树”资源（resize 时要重建） --------------------
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat{};
    VkExtent2D swapChainExtent{};
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    // -------------------- 渲染管线（本教程里不随 swapchain 重建；工程上可能需要） --------------------
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;

    // -------------------- 命令系统 --------------------
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;

    // -------------------- 同步：frames-in-flight --------------------
    std::vector<VkSemaphore> imageAvailableSemaphores; // acquire 完成 -> 渲染可写
    std::vector<VkSemaphore> renderFinishedSemaphores; // 渲染完成 -> present 可用
    std::vector<VkFence>     inFlightFences;           // CPU 等待 GPU 完成该帧
    uint32_t currentFrame = 0;

    // -------------------- resize 标志 --------------------
    bool framebufferResized = false;

    // =================================================================================
    // 3.1 Window 初始化与 resize 回调
    // =================================================================================
    void initWindow() {
        // GLFW：只做窗口与 surface 创建，不做 Vulkan 管线/同步
        // - GLFW_NO_API：不创建 OpenGL context
        // - glfwSetWindowUserPointer：把 this 指针塞进去，让 C 回调能访问对象成员
        // - framebufferSizeCallback：只置位 flag，不要在回调里直接重建 swapchain（线程/重入风险）
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int /*width*/, int /*height*/) {
        // GLFW 回调不能直接调用成员函数（缺少 this）
        // 解决：window user pointer 存放 this；回调里取回并设置 flag
        // 仅设置 flag：真正重建放在 drawFrame / recreateSwapChain，确保时序与同步正确
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    // =================================================================================
    // 3.2 Vulkan 初始化总入口
    // =================================================================================
    void initVulkan() {
        // 依赖关系（非常重要）：
        // instance -> (debug messenger) -> surface -> physical device -> logical device
        // logical device -> swapchain -> imageViews -> renderPass/pipeline -> framebuffers
        // device/queue family -> command pool -> command buffers
        // 最后创建同步对象（因为 drawFrame 需要它们）
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();

        createSwapChain();
        createImageViews();

        createRenderPass();
        createGraphicsPipeline();

        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects();
    }

    // =================================================================================
    // 3.3 主循环：每帧 drawFrame
    // =================================================================================
    void mainLoop() {
        // 核心：glfwPollEvents + drawFrame
        // drawFrame 内包含 acquire/submit/present，并处理 OUT_OF_DATE/SUBOPTIMAL/resize flag
        // 退出时 vkDeviceWaitIdle：避免 cleanup 时资源仍被 GPU 使用造成未定义行为
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }

    // =================================================================================
    // 3.4 Swapchain 子树清理（用于 resize 重建）
    // =================================================================================
    void cleanupSwapChain() {
        // 之所以拆出来：swapchain refresh 时要重复 “销毁旧资源 → 创建新资源”
        // 销毁顺序遵循依赖：framebuffer 依赖 imageView / swapchain image，因此先销 framebuffer
        // 再销 imageView，最后销 swapchain；并清空 vector 防止误用旧 handle
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        swapChainFramebuffers.clear();

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }
        swapChainImageViews.clear();

        vkDestroySwapchainKHR(device, swapChain, nullptr);
        swapChain = VK_NULL_HANDLE;
    }

    // =================================================================================
    // 3.5 全量清理
    // =================================================================================
    void cleanup() {
        // 顺序要点：
        // - 先清 swapchain 子树（依赖 device）
        // - 再销 pipeline/renderPass/commandPool/sync（也都依赖 device）
        // - 再销 device
        // - 最后销 debugMessenger/surface/instance/GLFW window
        cleanupSwapChain();

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    // =================================================================================
    // 3.6 Swapchain 重建（resize/minimize/out_of_date/suboptimal）
    // =================================================================================
    void recreateSwapChain() {
        // 为什么要处理 width/height==0：
        // - 窗口最小化时 framebuffer 尺寸可能为 0
        // - 0 尺寸创建 swapchain 会失败（很多驱动直接返回 VK_ERROR_INITIALIZATION_FAILED）
        // - glfwWaitEvents 是阻塞等待事件，比 while+poll 更省 CPU
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        // 最简单的同步方案：vkDeviceWaitIdle
        // - 保证旧 swapchain 的图像不再被 GPU 使用
        // - 代价：会暂停渲染；更高级做法是用 oldSwapchain 渐进切换（这里按教程简化）
        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        // 重建链：swapchain -> imageViews -> framebuffers
        // 注意：这里不重建 renderPass/pipeline（教程为简化）
        // 工程上：若 swapChainImageFormat 发生变化（如 SDR/HDR 切换），renderPass/pipeline 可能也要重建
        createSwapChain();
        createImageViews();
        createFramebuffers();
    }

    // =================================================================================
    // 3.7 Instance / Debug / Surface
    // =================================================================================
    void createInstance() {
        // createInstance 的关键点：
        // - validation layer 要先检查是否存在，否则 vkCreateInstance 可能 VK_ERROR_LAYER_NOT_PRESENT
        // - extensions 必须包含 GLFW 要求的 WSI 扩展；启用 debug 还要加 VK_EXT_debug_utils
        // - pNext 链上挂 DebugUtilsMessengerCreateInfoEXT：让 instance 创建/销毁阶段也能输出验证信息
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        // debug utils 回调的过滤策略：
        // - VERBOSE/INFO/WARNING/ERROR：你可以只保留 WARNING|ERROR 以减少输出
        // - GENERAL/VALIDATION/PERFORMANCE：性能提示也很有用（例如不合理的 barrier）
        // - pfnUserCallback：回调函数必须是静态/自由函数，或用 pUserData 传上下文
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger() {
        // debug messenger 依赖 instance
        // - 只在启用 validation layers 时创建
        // - 失败通常说明：未启用 VK_EXT_debug_utils 扩展 或 loader/驱动不支持
        // - 销毁必须早于 vkDestroyInstance
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void createSurface() {
        // surface 是 WSI 抽象：连接窗口系统与 Vulkan
        // - 它影响物理设备选择：不是所有 GPU/队列族都支持对该 surface present
        // - 必须在 instance 创建后创建，在 instance 销毁前销毁
        // - GLFW 内部调用平台相关的 vkCreate*SurfaceKHR
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    // =================================================================================
    // 3.8 物理设备选择 / 逻辑设备创建
    // =================================================================================
    void pickPhysicalDevice() {
        // Vulkan 先枚举 physical devices，再挑一个“满足需求”的
        // 这里的需求：graphics + present queue family，支持 VK_KHR_swapchain 且 swapchain 细节可用
        // 实务上可加入“打分策略”（独显优先/更高 limits/更大 VRAM 等），教程为简单起见选第一个可用
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& dev : devices) {
            if (isDeviceSuitable(dev)) {
                physicalDevice = dev;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        // 逻辑设备 VkDevice：
        // - 显式声明需要的队列（queue family + queueCount + priorities）
        // - 启用设备扩展（VK_KHR_swapchain）
        // - 可选启用 features（本教程不开任何特性）
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{}; // 全 false

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos    = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        // 设备层字段主要是兼容旧实现（新版一般忽略），教程仍保留
        if (enableValidationLayers) {
            createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        // 拿到队列句柄：queueIndex=0 因为每个 family 只创建 1 条队列
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(),  0, &presentQueue);
    }

    // =================================================================================
    // 3.9 Swapchain 创建与相关资源
    // =================================================================================
    void createSwapChain() {
        // swapchain 创建的关键参数：
        // - surfaceFormat：颜色格式与色彩空间（影响 renderpass attachment format）
        // - presentMode：FIFO 必有；MAILBOX 低延迟但不一定支持
        // - extent：最终像素尺寸（通常来自 framebuffer size）
        // - imageCount：min+1 常用；不超过 max
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR   presentMode   = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D         extent        = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 &&
            imageCount > swapChainSupport.capabilities.maxImageCount)
        {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType   = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount    = imageCount;
        createInfo.imageFormat      = surfaceFormat.format;
        createInfo.imageColorSpace  = surfaceFormat.colorSpace;
        createInfo.imageExtent      = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        // 如果 graphics 与 present 不同 family：
        // - CONCURRENT 简单：两边都能用，不需要 ownership transfer
        // - 代价：可能略慢（驱动难优化）
        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices   = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform   = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode    = presentMode;
        createInfo.clipped        = VK_TRUE;

        // 教程完整版：resize 时可以使用 oldSwapchain 实现“平滑切换”
        // - createInfo.oldSwapchain = oldSwapchain
        // - 创建新 swapchain 后再销毁旧 swapchain
        // 这里为了注释清晰/代码简化未启用（我们用 vkDeviceWaitIdle 直接停一停）
        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        // 获取 swapchain images（两阶段调用）
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent      = extent;
    }

    void createImageViews() {
        // imageView = “如何解释 VkImage 的视图”
        // - swapchain image 要作为 color attachment 使用：aspect=COLOR、mip=0、layer=1
        // - resize 时 swapchain images 变了，必须重建 imageViews
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image    = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format   = swapChainImageFormat;

            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            createInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel   = 0;
            createInfo.subresourceRange.levelCount     = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount     = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    void createRenderPass() {
        // render pass 定义 attachment 的 load/store、布局转换、subpass 结构与依赖关系
        // - initialLayout=UNDEFINED：不关心上一帧内容（配合 CLEAR）
        // - finalLayout=PRESENT_SRC_KHR：呈现之前必须转到可呈现布局
        // - subpass dependency：解决 EXTERNAL→subpass 的同步/布局转换顺序问题（教程推荐）
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format         = swapChainImageFormat;
        colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments    = &colorAttachmentRef;

        // 外部 -> subpass0 的依赖
        // 直觉理解：确保我们开始写 color attachment 之前，相关的布局转换/等待已经完成
        VkSubpassDependency dependency{};
        dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass    = 0;
        dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments    = &colorAttachment;
        renderPassInfo.subpassCount    = 1;
        renderPassInfo.pSubpasses      = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies   = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createGraphicsPipeline() {
        // pipeline 是 Vulkan 里最“重量级”的状态对象：
        // - shader stages + 固定功能状态 + pipelineLayout + renderPass/subpass 一起决定 pipeline
        // - 大多数状态不可变；改 shader / renderpass / vertex layout 往往要重建 pipeline
        // - 这里启用了动态 viewport/scissor：所以 pipeline 里只声明“有 1 个 viewport/scissor”，具体值在 cmd 里设
        auto vertShaderCode = readFile("shaders/vert.vert.spv");
        auto fragShaderCode = readFile("shaders/frag.frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage  = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName  = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName  = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        // 本教程顶点在 shader 中硬编码：所以 vertex binding / attributes 都为 0
        vertexInputInfo.vertexBindingDescriptionCount   = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable        = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth               = 1.0f;
        rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace               = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable         = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable  = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable   = VK_FALSE;
        colorBlending.logicOp         = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments    = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates    = dynamicStates.data();

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount         = 0;
        pipelineLayoutInfo.pushConstantRangeCount = 0;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount          = 2;
        pipelineInfo.pStages             = shaderStages;
        pipelineInfo.pVertexInputState   = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState      = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState   = &multisampling;
        pipelineInfo.pColorBlendState    = &colorBlending;
        pipelineInfo.pDynamicState       = &dynamicState;
        pipelineInfo.layout              = pipelineLayout;
        pipelineInfo.renderPass          = renderPass;
        pipelineInfo.subpass             = 0;
        pipelineInfo.basePipelineHandle  = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        // shader module 仅用于创建 pipeline；之后即可销毁（pipeline 内部已完成必要编译/链接）
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createFramebuffers() {
        // framebuffer = renderPass + 实际 attachments（imageViews）的绑定实例
        // - 一般 swapchain 有 N 张 image，就有 N 个 framebuffer
        // - framebuffer 尺寸必须与 swapchain extent 一致，否则 render pass 会越界/验证层报错
        // - resize 时 swapchain imageViews 变了，所以 framebuffer 必须重建
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = { swapChainImageViews[i] };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass      = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments    = attachments;
            framebufferInfo.width           = swapChainExtent.width;
            framebufferInfo.height          = swapChainExtent.height;
            framebufferInfo.layers          = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    // =================================================================================
    // 3.10 Command pool/buffer 与录制
    // =================================================================================
    void createCommandPool() {
        // command pool 绑定到 queue family：
        // - 从该 pool 分配出来的 command buffer 适合提交到该 family 的队列
        // - RESET_COMMAND_BUFFER_BIT：允许单独 reset 某个 command buffer（每帧复用常用）
        // - 更复杂场景：多线程录制通常每线程一个 pool，减少锁竞争
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void createCommandBuffers() {
        // per-frame command buffers：
        // - 我们按 MAX_FRAMES_IN_FLIGHT 分配，复用时用 currentFrame 轮转
        // - 每帧 reset 后重录，避免维护多个静态命令缓冲（教程简化）
        // - PRIMARY：可直接提交到队列；SECONDARY 常用于多线程/子pass
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool        = commandPool;
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        // 录制命令缓冲的常见结构：
        // begin -> beginRenderPass -> bindPipeline -> setDynamicStates -> draw -> endRenderPass -> end
        // 注意：我们在 drawFrame 里 reset 了该 command buffer，所以这里可以“从头写一遍”
        // 关键：framebuffer 使用 imageIndex 对应的 swapchain framebuffer（acquire 得到哪张就画哪张）
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass        = renderPass;
        renderPassInfo.framebuffer       = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues    = &clearColor;

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            // 动态 viewport/scissor：pipeline 里声明了“它们是动态的”，这里给出具体值
            VkViewport viewport{};
            viewport.x        = 0.0f;
            viewport.y        = 0.0f;
            viewport.width    = (float)swapChainExtent.width;
            viewport.height   = (float)swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = { 0, 0 };
            scissor.extent = swapChainExtent;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            // 画 3 个顶点（shader 内硬编码）组成一个 triangle
            vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    // =================================================================================
    // 3.11 同步对象（semaphore/fence）与 drawFrame（swapchain recreation 重点）
    // =================================================================================
    void createSyncObjects() {
        // 每帧一套同步对象：
        // - imageAvailableSemaphore：vkAcquireNextImageKHR signal；queue submit wait
        // - renderFinishedSemaphore：queue submit signal；vkQueuePresentKHR wait
        // - inFlightFence：queue submit signal；CPU 下一次使用该帧资源前 wait
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // 初始 signaled：否则第一帧 vkWaitForFences 会永远等不到（因为还没 submit 过）
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    void drawFrame() {
        // 一帧典型时序：
        // 1) 等待上一轮使用 currentFrame 这套资源的 GPU 工作完成（fence）
        // 2) acquire swapchain image（可能 OUT_OF_DATE）
        // 3) reset fence（仅当确定会 submit；避免 deadlock）
        // 4) reset + record command buffer
        // 5) submit：wait imageAvailableSemaphore，signal renderFinishedSemaphore，绑定 fence
        // 6) present：wait renderFinishedSemaphore（可能 OUT_OF_DATE 或 SUBOPTIMAL）
        // 7) 根据 result / framebufferResized 决定是否重建 swapchain
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex = 0;
        VkResult result = vkAcquireNextImageKHR(
            device,
            swapChain,
            UINT64_MAX,
            imageAvailableSemaphores[currentFrame], // acquire 完成后 signal
            VK_NULL_HANDLE,
            &imageIndex
        );

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            // acquire 阶段 OUT_OF_DATE：这张 swapchain 已经不再可用，立即重建并 return
            // 关键：此时不要 reset fence（否则会造成“reset 了但没 submit → fence 永不 signal → 下次 wait 死锁”）
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            // VK_SUBOPTIMAL_KHR 在 acquire 也可能出现：表示还能用，但不完美匹配
            // 这里选择继续渲染（因为已经拿到了 image）；present 后再处理是否重建
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // 修复 deadlock 的关键点：
        // 只有当我们确定要提交工作（不会提前 return）时，才 reset fence
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // command buffer 复用策略：每帧 reset 后重录
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        // wait：图像可用后再进入 COLOR_ATTACHMENT_OUTPUT 阶段写颜色附件
        // pWaitDstStageMask 决定“从哪个 pipeline stage 开始等待”：
        // - 设为 COLOR_ATTACHMENT_OUTPUT：允许更早阶段（如 vertex shader）与 acquire 部分重叠（在某些实现上有益）
        VkSemaphore waitSemaphores[]      = { imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores    = waitSemaphores;
        submitInfo.pWaitDstStageMask  = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &commandBuffers[currentFrame];

        // signal：渲染完成后 signal，present 将等待它
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores    = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        // present：等待渲染完成 semaphore
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores    = signalSemaphores;

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains    = swapChains;
        presentInfo.pImageIndices  = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        // 何时需要重建：
        // - OUT_OF_DATE：swapchain 与 surface 不兼容（典型 resize）
        // - SUBOPTIMAL：还能 present，但不再完美匹配（比如窗口移动到不同色域/HDR 显示器，或尺寸不一致）
        // - framebufferResized：GLFW 回调置位（因为有的驱动不会自动报 OUT_OF_DATE）
        //
        // 注意：教程强调最好在 present 之后处理 framebufferResized，
        // 因为此时信号量状态更一致，避免出现“信号量 signal 了但永远没被 wait”的奇怪问题。
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        // 轮转 currentFrame：复用下一套 per-frame 资源
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // =================================================================================
    // 3.12 其他工具函数：shader、format/mode/extent、查询支持、设备适配、扩展/层检查、读文件、回调
    // =================================================================================
    VkShaderModule createShaderModule(const std::vector<char>& code) {
        // SPIR-V 是按 uint32_t words 存储；pCode 需要 4 字节对齐
        // vkCreateShaderModule 只是“装载字节码”；真正编译通常在创建 pipeline 时发生
        // 如果 shader 文件路径不对（工作目录问题），readFile 会先抛异常
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode    = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }
        return shaderModule;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        // 偏好：B8G8R8A8_SRGB + SRGB_NONLINEAR（常见默认）
        // SRGB 能让颜色在显示器上更符合人眼观感（gamma 校正），但具体要看你的渲染/资产管线
        // 找不到偏好就退回第一个（保证可运行；更严谨可做更复杂的候选策略）
        for (const auto& f : availableFormats) {
            if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return f;
            }
        }
        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        // MAILBOX：低延迟+无撕裂（若支持），类似“总是显示最新的一帧”
        // FIFO：一定支持，类似 vsync（更稳但延迟高）
        // IMMEDIATE：可能撕裂，但延迟低
        for (const auto& m : availablePresentModes) {
            if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        // currentExtent != UINT32_MAX：平台给定固定 extent，必须用它（常见于某些移动端/Wayland 等）
        // 否则需要自己选择：通常用 framebuffer 像素尺寸（glfwGetFramebufferSize）
        // 最后 clamp 到 min/maxImageExtent，避免越界导致 vkCreateSwapchainKHR 失败
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width = 0, height = 0;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width  = std::clamp(actualExtent.width,
                                             capabilities.minImageExtent.width,
                                             capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height,
                                             capabilities.minImageExtent.height,
                                             capabilities.maxImageExtent.height);
            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device_) {
        // swapchain 支持是 “device + surface” 的组合属性
        // - capabilities：硬约束（数量/尺寸/transform 等）
        // - formats/presentModes：可选集合（必须非空才可创建 swapchain）
        // 两阶段查询：先拿 count，再分配数组再获取数据
        SwapChainSupportDetails details{};

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_, surface, &details.capabilities);

        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, &formatCount, nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device_, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device_, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    bool isDeviceSuitable(VkPhysicalDevice device_) {
        // “适配”检查：
        // 1) queue families 是否完整（graphics + present）
        // 2) 设备扩展是否支持（VK_KHR_swapchain）
        // 3) swapchain 支持细节是否足够（formats/presentModes 非空）
        QueueFamilyIndices indices = findQueueFamilies(device_);

        bool extensionsSupported = checkDeviceExtensionSupport(device_);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            auto sc = querySwapChainSupport(device_);
            swapChainAdequate = !sc.formats.empty() && !sc.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device_) {
        // 枚举设备支持的扩展，再检查 requiredExtensions 是否都在其中
        // 用 set 做差集最简单：最终为空表示全部满足
        // 对 swapchain 来说：缺 VK_KHR_swapchain 就无法 present（本教程必需）
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(device_, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device_, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
        for (const auto& ext : availableExtensions) {
            requiredExtensions.erase(ext.extensionName);
        }
        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device_) {
        // 1) queue family properties 描述每个 family 的能力（graphics/compute/transfer 等 flags）
        // 2) present 支持需要单独查询：vkGetPhysicalDeviceSurfaceSupportKHR
        // 3) 允许 graphicsFamily == presentFamily（很多桌面 GPU 是这样），也允许不同（某些平台/驱动可能不同）
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device_, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device_, &queueFamilyCount, queueFamilies.data());

        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            const auto& q = queueFamilies[i];

            if (q.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device_, i, surface, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) break;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions() {
        // GLFW 负责给出创建 surface 所需的 instance 扩展（平台相关）
        // - Windows: VK_KHR_win32_surface + VK_KHR_surface
        // - X11/Wayland: 对应的 surface 扩展 + VK_KHR_surface
        // 开启 validation 时再加 VK_EXT_debug_utils，才能创建 debug messenger
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        return extensions;
    }

    bool checkValidationLayerSupport() {
        // validation layers 是可选组件：必须先枚举系统可用层再决定启用
        // VK_LAYER_KHRONOS_validation 是标准验证层（LunarG SDK 常见）
        // 若缺失：要么没装 SDK，要么运行环境没部署 layer（比如没有 VulkanSDK 或 layer JSON 路径没配置）
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool found = false;
            for (const auto& p : availableLayers) {
                if (strcmp(layerName, p.layerName) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return true;
    }

    static std::vector<char> readFile(const std::string& filename) {
        // 以二进制方式读取 SPIR-V：
        // ios::ate：初始定位到文件尾，tellg 得到大小，然后分配 buffer 再读
        // 常见坑：IDE 工作目录不对导致找不到 shaders/xxx.spv（需在运行配置里设置 working directory）
        // 如果你想更鲁棒：可以同时尝试从 exe 目录/资源目录读取（教程后续经常会做）
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
        VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* /*pUserData*/)
    {
        // validation layer 输出入口：
        // - pMessage 里通常包含“哪个对象/哪个函数/哪个 VUID”出错，非常有助于定位
        // - 这里简单打印到 stderr；工程里可接入日志系统并按 severity 过滤
        // - 返回 VK_FALSE 表示不终止 Vulkan 调用（只是报告）
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
};

// =====================================================================================
// 4) main：驱动应用生命周期
// =====================================================================================
int main() {
    // main 只做异常边界与返回码：
    // - Vulkan 初始化失败、shader 读不到、设备不支持等都会抛异常
    // - catch 后打印原因并返回失败码，方便脚本/CI 判断
    // - app.run 内部负责 init/loop/cleanup 的完整流程
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
