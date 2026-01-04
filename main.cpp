// ============================================================================
//  Vulkan Tutorial - Hello Triangle (GLFW + Vulkan C API)
//  目标：在尽量“教程式”的注释中解释每一步 Vulkan 初始化/Swapchain/管线搭建。
//  说明：Vulkan 的核心理念是“显式”——几乎所有状态/资源/同步都需要你自己声明。
// ============================================================================

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
#include <string>   // NOTE: 你原代码里用到了 std::string，但没 include，会编译失败

const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;

// Vulkan 的 validation layer（调试层）
// 作用：在开发期帮你检查 API 参数合法性、资源生命周期、同步错误等；Release 通常关闭以零开销运行。
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// 设备级扩展：swapchain 是 Vulkan 核心之外的 WSI（Window System Integration）能力
// 没启用 VK_KHR_swapchain 就无法创建 VkSwapchainKHR，也就无法显示到窗口。
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// ============================================================================
//  Debug Utils 扩展函数加载：vkCreateDebugUtilsMessengerEXT 不是 core 函数
//  Vulkan 扩展函数通常需要通过 vkGetInstanceProcAddr / vkGetDeviceProcAddr 动态获取。
// ============================================================================

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger
) {
    // [知识点1] 扩展函数不是一定存在：需要先启用对应 instance extension（VK_EXT_debug_utils）
    // [知识点2] vkGetInstanceProcAddr 返回的是 void* 风格函数指针，需要强转成 PFN_ 类型
    // [知识点3] 若返回 nullptr，说明实现不支持/未启用扩展；要给出可诊断的错误码
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator
) {
    // [知识点1] 销毁函数同样是扩展函数，也要动态获取
    // [知识点2] 释放顺序：通常先 Destroy debug messenger，再 Destroy instance（否则 instance 已没了）
    // [知识点3] Vulkan 的 pAllocator 一般传 nullptr（使用默认分配器），保持一致即可
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

// ============================================================================
//  Queue Family 选择：显卡通常有多个“队列族”(queue family)，每个族支持不同能力：graphics/compute/transfer/present...
//  我们要找到：
//    - graphicsFamily：能执行图形命令（VK_QUEUE_GRAPHICS_BIT）
//    - presentFamily：能把 swapchain image 提交到 surface 显示（vkGetPhysicalDeviceSurfaceSupportKHR）
// ============================================================================

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        // [知识点1] std::optional 表示“可能还没找到”：避免用 -1 这种魔法值，更类型安全
        // [知识点2] “完整”不是指 family 本身完整，而是“我们需要的那些 family 索引都已找到”
        // [知识点3] 后续如果你还需要 computeFamily/transferFamily，就把它们也加进来一起判断
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// ============================================================================
//  Swapchain 支持细节查询：选择 swapchain 时必须了解 surface 的能力（format/present mode/extent 等）
// ============================================================================

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;           // min/max image count、extent、变换、合成 alpha 等
    std::vector<VkSurfaceFormatKHR> formats;         // 像素格式 + 色彩空间（SRGB/Linear/等）
    std::vector<VkPresentModeKHR> presentModes;      // FIFO/MAILBOX/IMMEDIATE 等
};

class HelloTriangleApplication {
public:
    void run() {
        // [知识点1] Vulkan 初始化通常分“窗口系统”与“Vulkan 对象”两大块
        // [知识点2] 典型生命周期：init -> loop -> cleanup；cleanup 顺序要严格（先子资源后父资源）
        // [知识点3] 这里的 run() 是一个高层 orchestrator：把复杂过程拆成多个小函数更清晰、易 debug
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; // 物理设备句柄由 instance 管；无需 destroy
    VkDevice device = VK_NULL_HANDLE;                 // 逻辑设备：你创建的“虚拟设备”，持有队列/资源创建能力

    VkQueue graphicsQueue = VK_NULL_HANDLE;           // 从某个 queue family 拿到的具体队列句柄
    VkQueue presentQueue  = VK_NULL_HANDLE;

    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages;             // swapchain 内部图像（VkImage）
    VkFormat swapChainImageFormat{};
    VkExtent2D swapChainExtent{};
    std::vector<VkImageView> swapChainImageViews;     // VkImage 要进入渲染管线通常需要 ImageView

    VkRenderPass renderPass = VK_NULL_HANDLE;         // 传统 render pass（非 dynamic rendering）
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE; // 描述 descriptor set layouts + push constants
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;     // 真正可用于 draw 的 pipeline 对象

    void initWindow() {
        // [知识点1] GLFW 负责跨平台窗口与输入；我们用它创建窗口并获取 Vulkan 所需 instance extensions
        // [知识点2] GLFW_CLIENT_API=NO_API 表示不要创建 OpenGL Context（否则会多余/冲突）
        // [知识点3] 这里禁用 resize 简化教程；后续 swapchain recreation 章节会专门处理 resize/minimize
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        // [知识点1] 初始化顺序很关键：instance -> surface -> physical device -> device/queues -> swapchain -> views -> renderpass -> pipeline
        // [知识点2] surface 影响 physical device 选择（并不是所有 GPU/队列族都能 present 到该 surface）
        // [知识点3] swapchain 及其派生对象（image view/framebuffer）往往是“窗口相关资源”，resize 时要重建
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
    }

    void mainLoop() {
        // [知识点1] Vulkan 的真正渲染通常在 drawFrame()：acquire -> record -> submit -> present
        // [知识点2] 这里为了聚焦初始化，没有 drawFrame；真实项目里要加入同步与 frames-in-flight
        // [知识点3] glfwPollEvents 让 GLFW 处理系统事件队列（键鼠/窗口消息）
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        // [知识点1] 销毁顺序：先销毁依赖 device 的资源（pipeline/renderpass/imageviews/swapchain），再销毁 device
        // [知识点2] debug messenger 依赖 instance；所以 destroy device 之后，destroy debug messenger，再 destroy instance
        // [知识点3] surface 也是 instance 的 child；必须在 vkDestroyInstance 之前销毁
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void createInstance() {
        // [知识点1] VkInstance 是 Vulkan “全局上下文”：加载驱动、枚举物理设备、启用 instance 扩展/层
        // [知识点2] Validation layer 属于 instance 层（旧模型下），需在创建 instance 时启用
        // [知识点3] VkApplicationInfo 主要用于驱动统计/兼容性路径/工具诊断；不是“必须”，但强烈建议填
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";          // 主要用于诊断/工具显示/驱动统计（一般不会影响正确性）
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";                    // 引擎名更常用于统计与特定兼容/优化路径（依实现）
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;              // 告诉 loader/驱动：你期望的 Vulkan API 版本（兼容性关键）

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();            // GLFW + (可选) debug utils
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            // [知识点] 把 debug messenger create info 挂到 pNext：
            // 这样 vkCreateInstance / vkDestroyInstance 期间也能输出校验信息（否则你会错过早期报错）
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
        // [知识点1] Debug messenger 决定“哪些消息会回调给你”：severity + type
        // [知识点2] VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT 很吵；你可以根据需要筛掉
        // [知识点3] 回调函数不能抛异常（C ABI），一般只做 log/统计；严重错误可触发断点/标记
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
        // [知识点1] debug messenger 依赖 VK_EXT_debug_utils 扩展（需要在 instance extensions 中启用）
        // [知识点2] 这是“运行期”消息回调；而 createInstance 里 pNext 挂载的是“instance 创建期”的消息回调
        // [知识点3] 关闭 validation layers 时应跳过创建，避免无意义工作与潜在不兼容
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void createSurface() {
        // [知识点1] VkSurfaceKHR 是“可呈现目标”的抽象：把 Vulkan 与窗口系统连接起来（WSI）
        // [知识点2] surface 是 instance-level 对象：用 vkDestroySurfaceKHR(instance, surface, ...)
        // [知识点3] surface 影响“present 支持性”：需要 vkGetPhysicalDeviceSurfaceSupportKHR 检查队列族是否可 present
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void pickPhysicalDevice() {
        // [知识点1] VkPhysicalDevice 代表真实硬件（或软件实现）；句柄随 instance 生命周期有效
        // [知识点2] 选择标准通常包含：队列族支持、必要扩展（swapchain）、特性（samplerAnisotropy 等）、性能偏好
        // [知识点3] 教程选“第一个合适的”，工程里常做打分（独显优先/更高 limits/更大 VRAM 等）
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
        // [知识点1] VkDevice 是逻辑设备：你在其上创建 buffer/image/pipeline 等资源，并获取队列 VkQueue 提交工作
        // [知识点2] 队列从“队列族”创建：VkDeviceQueueCreateInfo 指定 queueFamilyIndex + queueCount + priority
        // [知识点3] graphicsFamily 与 presentFamily 可能相同，也可能不同；不同则要为两个 family 都创建队列
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        float queuePriority = 1.0f; // 0~1，影响同一设备内部对队列调度的“相对优先级”
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;                 // 一个 family 里可以创建多个队列（若该 family 的 queueCount > 1）
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};           // 教程先不启用额外特性

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        // 设备扩展：swapchain 是 device-level extension
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        // 旧实现可能仍要求填写 enabledLayerCount（新模型里 device layer 已废弃），教程保持兼容写法
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        // [知识点] vkGetDeviceQueue 的第 3 个参数 queueIndex：
        // 一个 queue family 可能暴露多个队列（比如 graphics queue 族里有 16 个队列）。
        // 你创建 device 时指定了 queueCount=1，所以这里只能取 index=0。
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(),  0, &presentQueue);
    }

    void createSwapChain() {
        // [知识点1] swapchain 是“待显示图像队列”：你渲染到其中某张 image，然后 present 给屏幕
        // [知识点2] swapchain 的参数必须匹配 surface 能力：format/presentMode/extent/min/max image count
        // [知识点3] sharingMode：若 graphics 与 present 是不同队列族，可能需要 CONCURRENT（更简单但可能慢一点）
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR   presentMode   = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D         extent        = chooseSwapExtent(swapChainSupport.capabilities);

        // 一般取 minImageCount+1 做“多缓冲”，减少等待；但不能超过 maxImageCount（max=0 表示无限制）
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 &&
            imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount    = imageCount;
        createInfo.imageFormat      = surfaceFormat.format;
        createInfo.imageColorSpace  = surfaceFormat.colorSpace;
        createInfo.imageExtent      = extent;
        createInfo.imageArrayLayers = 1;                          // VR/多视图才会 >1
        createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // 我们渲染到它，所以作为 color attachment

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        if (indices.graphicsFamily != indices.presentFamily) {
            // CONCURRENT：两个队列族都能访问 swapchain image，不用显式 ownership transfer
            createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices   = queueFamilyIndices;
        } else {
            // EXCLUSIVE：性能更好（实现更容易优化），但跨队列族需要 ownership transfer（本例无需）
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform   = swapChainSupport.capabilities.currentTransform; // 是否旋转/翻转
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;              // 与窗口系统合成时 alpha 如何处理
        createInfo.presentMode    = presentMode;
        createInfo.clipped        = VK_TRUE;                                        // 被遮挡像素可裁剪，省资源
        createInfo.oldSwapchain   = VK_NULL_HANDLE;                                 // resize/recreate 时会用到旧链

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        // swapchain 内部有若干 VkImage，需要取出来保存（两步：先拿 count，再拿数组）
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent      = extent;
    }

    void createImageViews() {
        // [知识点1] VkImage 本体只是“存储”，如何解释它（format/子资源范围/2D vs 3D）由 VkImageView 决定
        // [知识点2] 作为颜色附件：aspectMask=COLOR；mip/array 都是 1（swapchain image 通常无 mip）
        // [知识点3] ImageView 是 device 资源，必须在 vkDestroyDevice 前销毁；并且在 swapchain 重建时要重建
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;

            // swizzle：允许重映射通道；IDENTITY 表示不改
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
        // [知识点1] 传统 render pass 描述“attachment 的使用方式/布局转换/子通道关系”
        // [知识点2] attachment 是“渲染的输入/输出目标”：颜色、深度、resolve、输入附件等（不是模板形状）
        // [知识点3] initialLayout/finalLayout 用于自动 layout transition（这里：UNDEFINED -> PRESENT_SRC_KHR）
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format  = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

        // loadOp：开始 render pass 时如何处理 attachment（CLEAR/LOAD/DONT_CARE）
        colorAttachment.loadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
        // storeOp：结束 render pass 时是否保留结果（present 必须 STORE）
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        // 这里没有 stencil，因此不关心
        colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;        // 不关心旧内容
        colorAttachment.finalLayout   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;  // 结束后要能 present

        // subpass 里引用 attachment：attachment=0 指向 pAttachments[0]
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments    = &colorAttachmentRef;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments    = &colorAttachment;
        renderPassInfo.subpassCount    = 1;
        renderPassInfo.pSubpasses      = &subpass;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createGraphicsPipeline() {
        // [知识点1] pipeline 在 Vulkan 中是“巨大且基本不可变”的状态对象：shader + 固定功能状态 + layout + renderpass
        // [知识点2] dynamic state 只是一小部分可变：例如 viewport/scissor；其余大多要重建 pipeline
        // [知识点3] shader module 是 SPIR-V 字节码的薄封装；真正的编译/链接发生在 vkCreateGraphicsPipelines
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

        // ========== 固定功能状态（教程简化版）==========

        // 顶点输入：此处 shader 内硬编码顶点，因此不从 vertex buffer 读入
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount   = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

        // 图元装配：我们画 triangle list
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // viewport/scissor：这里设为“动态”，所以创建时只给 count，实际值在录 command buffer 时设置
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount  = 1;

        // 光栅化：填充、背面剔除、顺时针为正面（注意 Vulkan NDC 的 Y 方向与 OpenGL 不同）
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable        = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth               = 1.0f;
        rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace               = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable         = VK_FALSE;

        // MSAA：先关闭
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable  = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // 颜色混合：先不混合，直接覆盖写入 RGBA
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable   = VK_FALSE;          // 若启用 logicOp，会禁用常规 blend（类似按位操作）
        colorBlending.logicOp         = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments    = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        // Dynamic state：告诉 pipeline 这些状态将由命令缓冲在 draw 时设置
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates    = dynamicStates.data();

        // pipelineLayout：descriptor set layout + push constants 的“根布局”
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount         = 0;      // 暂时没有 descriptor sets
        pipelineLayoutInfo.pushConstantRangeCount = 0;      // 暂时没有 push constants

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // ========== 组装 VkGraphicsPipelineCreateInfo ==========

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
        pipelineInfo.renderPass          = renderPass;      // pipeline 需要与 render pass 兼容
        pipelineInfo.subpass             = 0;
        pipelineInfo.basePipelineHandle  = VK_NULL_HANDLE;  // 派生管线优化：此处不用

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        // [知识点] shader module 创建后只在 pipeline 创建期需要；pipeline 创建完成可立即销毁 module
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        // [知识点1] VkShaderModule 是 SPIR-V 的容器；pCode 要求 4 字节对齐并以 uint32_t 解释
        // [知识点2] codeSize 以“字节”为单位；SPIR-V 本质是 uint32_t words 的序列
        // [知识点3] 失败常见原因：SPIR-V 版本不匹配/文件读错/路径不对（建议打印绝对路径调试）
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
        // [知识点1] format + colorSpace 一起决定颜色意义；教程偏好 SRGB_NONLINEAR（显示器常用）
        // [知识点2] 常见组合：VK_FORMAT_B8G8R8A8_SRGB + VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
        // [知识点3] 若没有理想格式，通常退化为第一个可用（但工程里可能更精细：HDR/10bit 等）
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
                availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        // [知识点1] FIFO：必定支持，类似“垂直同步”，延迟较高但不撕裂
        // [知识点2] MAILBOX：常被视为“三缓冲”，低撕裂且延迟较低（若支持通常优先）
        // [知识点3] IMMEDIATE：可能撕裂，但延迟最低（部分场景可用）
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        // [知识点1] extent 决定 swapchain image 分辨率（像素级），不一定等于窗口逻辑大小
        // [知识点2] 若 currentExtent != UINT32_MAX，说明平台已经固定好了 extent（你必须用它）
        // [知识点3] 否则你可自己选，但必须 clamp 到 min/maxImageExtent（平台限制）
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height); // 注意：framebuffer size 是像素，不是窗口逻辑点

            VkExtent2D actualExtent = {
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

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice dev) {
        // [知识点1] swapchain 支持性完全依赖 surface：同一块 GPU 对不同 surface 可能结果不同
        // [知识点2] 查询一般分三组：capabilities（尺寸/数量/变换）、formats（像素格式）、present modes（呈现策略）
        // [知识点3] formats/presentModes 可能为空（表示不可用）；isDeviceSuitable 里要检查
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surface, &details.capabilities);

        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &formatCount, nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    bool isDeviceSuitable(VkPhysicalDevice dev) {
        // [知识点1] “合适”通常由多个条件组成：队列族齐全、扩展支持、swapchain 充足、所需特性支持
        // [知识点2] indices.isComplete() 表示我们找到 graphics+present family；如果你需要 compute 也要纳入
        // [知识点3] swapChainAdequate 检查 format 和 present mode 非空：否则即使支持扩展也无法真正创建 swapchain
        QueueFamilyIndices indices = findQueueFamilies(dev);

        bool extensionsSupported = checkDeviceExtensionSupport(dev);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(dev);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice dev) {
        // [知识点1] Vulkan 的扩展是“能力开关”：你必须显式启用它，才能调用/创建相关对象
        // [知识点2] 设备扩展与 instance 扩展分开枚举；swapchain 是 device 扩展
        // [知识点3] 常用做法：把 requiredExtensions 放入 set，然后从 availableExtensions 里逐个 erase，最后看是否为空
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev) {
        // [知识点1] queue family 是 Vulkan 并行执行的基础：你把 command buffer submit 到队列上执行
        // [知识点2] graphicsFamily 通过 queueFlags 找；presentFamily 需要对“特定 surface”调用 vkGetPhysicalDeviceSurfaceSupportKHR
        // [知识点3] indices.isComplete() 早停是优化：找到所需 family 就可以 break（避免无意义遍历）
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }
            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions() {
        // [知识点1] WSI 需要 instance extensions（例如 VK_KHR_surface + 平台相关 surface 扩展）
        // [知识点2] GLFW 已经帮你封装好了：glfwGetRequiredInstanceExtensions 返回创建 window surface 所需扩展
        // [知识点3] 开启 validation 时，还要加上 VK_EXT_debug_utils，否则 debug messenger 扩展函数会拿不到
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = nullptr;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        return extensions;
    }

    bool checkValidationLayerSupport() {
        // [知识点1] layer 名是字符串；如果请求了不存在的 layer，vkCreateInstance 会失败（VK_ERROR_LAYER_NOT_PRESENT）
        // [知识点2] 标准校验层通常是 VK_LAYER_KHRONOS_validation；你也可以列多个 layer
        // [知识点3] 这里的逻辑：对每个 requested layer，去 availableLayers 里找同名项
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound) {
                return false;
            }
        }
        return true;
    }

    static std::vector<char> readFile(const std::string& filename) {
        // [知识点1] SPIR-V 是二进制；用 ios::binary 打开，且常用 ios::ate 先定位到末尾拿文件大小
        // [知识点2] 一次性读入 vector<char>，便于传给 createShaderModule；注意路径问题（cwd vs 可执行文件目录）
        // [知识点3] 文件打不开时尽早抛异常；工程里建议输出 filename/绝对路径/工作目录辅助定位
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
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData
    ) {
        // [知识点1] 这是从 validation layers 发出的消息回调（发生在 Vulkan API 调用附近）
        // [知识点2] messageSeverity/type 可用于过滤（比如只打印 WARNING/ERROR）
        // [知识点3] 返回 VK_FALSE 表示“不终止”触发该消息的 Vulkan 调用（大多数情况下应返回 VK_FALSE）
        (void)messageSeverity;
        (void)messageType;
        (void)pUserData;

        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
};

int main() {
    // [知识点1] main 只负责异常边界与进程返回码；把 Vulkan 逻辑放到类里更清晰
    // [知识点2] Vulkan 初始化/创建资源失败很常见（驱动/扩展/权限/路径等），用异常能快速退出并打印原因
    // [知识点3] 真实项目里你可能还会做：命令行参数、日志系统初始化、GPU 选择策略、崩溃转储等
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
