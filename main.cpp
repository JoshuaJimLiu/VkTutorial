// =====================================================================================
// Vulkan Tutorial - Vertex/Index Buffer + Staging Buffer + Swapchain Recreation
// 在原教程基础上：
//   - 使用 interleaved Vertex（pos+color）
//   - 使用 Index Buffer（矩形）
//   - 使用 Staging Buffer 上传到 DEVICE_LOCAL
//   - 支持 swapchain recreation（窗口 resize / out-of-date）
//   - 使用动态 viewport/scissor（pipeline 里启用 dynamic states）
//
// 注意：这份代码仍然保持“教程风格”：
//   - 没引入 VMA / 内存池（每个 buffer 单独 vkAllocateMemory）
//   - copyBuffer 用 graphicsQueue + vkQueueWaitIdle（简单但不高效）
// =====================================================================================

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>

// -------------------------------------------------------------------------------------
// 基本窗口参数
// -------------------------------------------------------------------------------------
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// “同时在飞的帧数”（frames-in-flight）
// 典型双缓冲：CPU 可以提前准备下一帧命令，而 GPU 在执行上一帧
const int MAX_FRAMES_IN_FLIGHT = 2;

// -------------------------------------------------------------------------------------
// Validation Layers（调试层）
// -------------------------------------------------------------------------------------
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// -------------------------------------------------------------------------------------
// Device Extensions（设备扩展）
// 这里必须启用 swapchain 扩展才能呈现到窗口
// -------------------------------------------------------------------------------------
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// Debug/Release 下是否启用 validation layers
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// =====================================================================================
// Debug Utils Messenger: 扩展函数加载（因为它不是 core 函数）
// vkCreateDebugUtilsMessengerEXT / vkDestroyDebugUtilsMessengerEXT 属于 VK_EXT_debug_utils
// 需要通过 vkGetInstanceProcAddr 动态取地址（类似 dlsym/GetProcAddress）
// =====================================================================================
VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger
) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        // 说明实例没启用 VK_EXT_debug_utils 扩展（或驱动不支持）
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator
) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

// =====================================================================================
// Queue Families（队列族）索引：需要 graphics + present
// 说明：graphicsQueue 用来提交渲染命令；presentQueue 用来呈现 swapchain image
// 它们可能在同一个 queue family，也可能不同（因此要兼容两者）
// =====================================================================================
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// =====================================================================================
// Swapchain 支持信息：format / present mode / capabilities
// 用于选择合适的 swapchain 配置
// =====================================================================================
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

// =====================================================================================
// Vertex 定义（与 shader 输入匹配）
// shader: layout(location=0) in vec2 inPosition;
//         layout(location=1) in vec3 inColor;
// interleaved: 一个数组里每个元素 Vertex = {pos, color}
// =====================================================================================
struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;

    // BindingDescription 描述：从哪个 binding 读取、每个元素间隔 stride、按 vertex/instance 推进
    // - binding=0：我们只用一个 vertex buffer，所以绑定点为 0
    // - stride=sizeof(Vertex)：从一个 Vertex 跳到下一个 Vertex 的字节跨度
    // - inputRate=VERTEX：每处理一个“顶点”推进一次（不是每个 instance）
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // per-vertex

        return bindingDescription;
    }

    // AttributeDescriptions 描述：location->从 Vertex 里怎么取（format + offset）
    // location 必须和 shader 的 layout(location=...) 对齐
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        // 位置属性：vec2 -> VK_FORMAT_R32G32_SFLOAT
        attributeDescriptions[0].binding = 0;                       // 来自 binding 0
        attributeDescriptions[0].location = 0;                      // shader location=0
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;  // vec2(float,float)
        attributeDescriptions[0].offset = offsetof(Vertex, pos);    // Vertex 内 pos 的字节偏移

        // 颜色属性：vec3 -> VK_FORMAT_R32G32B32_SFLOAT
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT; // vec3(float,float,float)
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

// -------------------------------------------------------------------------------------
// 顶点数据：矩形 4 个角点（颜色只是演示）
// 注意：这只是 CPU 侧数组，最终会被上传到 GPU 的 VkBuffer
// -------------------------------------------------------------------------------------
const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{ 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{ 0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f,  0.5f}, {1.0f, 1.0f, 1.0f}}
};

// -------------------------------------------------------------------------------------
// 索引数据：两个三角形拼成矩形
// 用 uint16_t 是因为顶点数 < 65535（也可以用 uint32_t）
// -------------------------------------------------------------------------------------
const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0
};

// =====================================================================================
// HelloTriangleApplication：教程式“把所有 Vulkan 对象当成员变量管理生命周期”
// 重要：Vulkan 的大部分对象需要你显式 Destroy/Free；顺序要正确：
//   - 依赖 device 的对象必须在 vkDestroyDevice 前销毁
//   - surface 必须在 instance 销毁前销毁
// =====================================================================================
class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    // ---------------------------------------------------------------------------------
    // Window / Instance / Surface
    // ---------------------------------------------------------------------------------
    GLFWwindow* window = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    // ---------------------------------------------------------------------------------
    // Physical / Logical device & queues
    // ---------------------------------------------------------------------------------
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;

    // ---------------------------------------------------------------------------------
    // Swapchain + ImageViews + Framebuffers
    // swapchain 相关资源：窗口 resize/格式变化时需要重建（recreateSwapChain）
    // ---------------------------------------------------------------------------------
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages;          // swapchain image 由驱动创建，我们只是拿句柄
    VkFormat swapChainImageFormat{};
    VkExtent2D swapChainExtent{};
    std::vector<VkImageView> swapChainImageViews;  // 我们自己创建
    std::vector<VkFramebuffer> swapChainFramebuffers; // 依赖 renderPass + imageView

    // ---------------------------------------------------------------------------------
    // Render pass / Pipeline
    // 注意：本教程里 pipeline 与 swapchain format 强相关（renderPass.format）。
    // 严格来讲：如果 swapchain format 发生变化，renderPass/pipeline 也应重建。
    // 这份代码的 recreateSwapChain 只重建 swapchain/imageViews/framebuffers，
    // 依赖“format 不变”的常见情况，简化逻辑。
    // ---------------------------------------------------------------------------------
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;

    // ---------------------------------------------------------------------------------
    // Command pool & command buffers
    // - commandPool 绑定在 graphics queue family 上
    // - commandBuffers：每帧一个（MAX_FRAMES_IN_FLIGHT 个）
    // ---------------------------------------------------------------------------------
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;

    // ---------------------------------------------------------------------------------
    // Vertex/Index buffers（GPU 侧资源）
    // vertexBuffer/indexBuffer：DEVICE_LOCAL（不能 map）
    // stagingBuffer：临时 HOST_VISIBLE/COHERENT，用完销毁
    // ---------------------------------------------------------------------------------
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;

    // ---------------------------------------------------------------------------------
    // Sync objects（每帧一套）
    // imageAvailableSemaphore: acquire 完成后 signal（图像可用）
    // renderFinishedSemaphore: 渲染提交完成后 signal（可 present）
    // inFlightFence: CPU 等 GPU 完成该帧，避免覆盖 commandBuffer/资源
    // ---------------------------------------------------------------------------------
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;

    // GLFW resize callback 触发的标志位：提示 swapchain 需要重建
    bool framebufferResized = false;

    // =================================================================================
    // 1) Window init
    // =================================================================================
    void initWindow() {
        glfwInit();

        // 告诉 GLFW：我们不用 OpenGL Context（否则它会为你创建 GL 上下文）
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

        // 把 this 指针挂到 GLFW window 上，便于 callback 拿到 app 对象
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    // GLFW 的 callback 必须是静态函数/自由函数
    static void framebufferResizeCallback(GLFWwindow* window, int /*width*/, int /*height*/) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        // 这里只设置一个 flag：真正重建 swapchain 在 drawFrame 里做（更安全）
        app->framebufferResized = true;
    }

    // =================================================================================
    // 2) Vulkan init - 顺序非常重要（依赖关系）
    // =================================================================================
    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();         // surface 影响 physical device 是否支持 present
        pickPhysicalDevice();
        createLogicalDevice();

        createSwapChain();       // swapchain format/extent 依赖 surface capabilities
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();

        createCommandPool();     // commandPool 要绑定到 graphics queue family
        createVertexBuffer();    // 上传顶点数据（staging -> device local）
        createIndexBuffer();     // 上传索引数据（staging -> device local）
        createCommandBuffers();  // 分配每帧命令缓冲（录制在 drawFrame 里）
        createSyncObjects();
    }

    // =================================================================================
    // 3) Main loop
    // =================================================================================
    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        // 等 GPU 完全空闲再退出：确保不再使用任何将要销毁的资源
        vkDeviceWaitIdle(device);
    }

    // =================================================================================
    // 4) Swapchain cleanup / full cleanup
    // =================================================================================
    void cleanupSwapChain() {
        // Framebuffer 依赖 imageView + renderPass
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        // ImageView 依赖 swapchain image
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        // 最后销毁 swapchain
        vkDestroySwapchainKHR(device, swapChain, nullptr);

        swapChainFramebuffers.clear();
        swapChainImageViews.clear();
        swapChainImages.clear();
    }

    void cleanup() {
        cleanupSwapChain();

        // pipeline/layout/renderPass 都是 device 级对象
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        // index/vertex buffer：先 destroy buffer 再 free memory（教程写法）
        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        // 每帧同步对象
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        // Command pool 必须在 device 销毁前销毁
        vkDestroyCommandPool(device, commandPool, nullptr);

        // device 销毁会隐式释放 queues
        vkDestroyDevice(device, nullptr);

        // debug messenger 依赖 instance（必须在 instance 销毁前销毁）
        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        // surface 依赖 instance
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    // =================================================================================
    // 5) Swapchain recreation（窗口 resize / out-of-date）
    // =================================================================================
    void recreateSwapChain() {
        // 如果窗口被最小化，framebuffer 可能变成 0x0。
        // Vulkan 不允许 extent 为 0，因此需要等恢复。
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        // 这里用最简单做法：等待 GPU 完全空闲再重建 swapchain
        // 更高性能方式：等相关 in-flight fences / per-image fences（更复杂）
        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        // 依赖顺序：swapchain -> imageViews -> framebuffers
        createSwapChain();
        createImageViews();
        createFramebuffers();

        // ⚠️严格来说：如果 swapchain format 改变，renderPass/pipeline 也应重建。
        // 教程在很多平台上 format 基本不变，所以这里省略。
    }

    // =================================================================================
    // 6) Instance + Debug messenger + Surface
    // =================================================================================
    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        // ApplicationInfo：可选，但建议填，让驱动知道你的应用信息
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        // InstanceCreateInfo：必须
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // GLFW 要求的 instance extensions（不同平台不同，比如 Win32 surface / XCB / etc.）
        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        // 把 DebugUtilsMessengerCreateInfoEXT 挂到 pNext：
        // 这样 vkCreateInstance / vkDestroyInstance 期间也能收到 validation 消息
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

    // 填充 debug messenger 的配置：消息级别 + 类型 + 回调函数
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

        // 你可以只开 WARNING/ERROR，VERBOSE 会非常多
        createInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

        // 消息类型：一般/验证/性能
        createInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void createSurface() {
        // GLFW 封装了平台相关 vkCreateXxxSurfaceKHR
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    // =================================================================================
    // 7) Physical device selection
    // =================================================================================
    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // 教程：选第一个满足需求的 device
        // 工程：可以打分（独显优先、feature/limits 优先）
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

    // =================================================================================
    // 8) Logical device & queues
    // =================================================================================
    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        // 如果 graphics/present 在不同 family，需要创建两个 queue
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        float queuePriority = 1.0f; // 0~1，影响调度优先级
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority; // 必填
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // 设备特性：此 demo 不用特殊 feature，保持默认 false
        VkPhysicalDeviceFeatures deviceFeatures{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        // 启用 swapchain 扩展
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        // 旧实现可能需要 device layer（现在通常忽略，但教程保留字段）
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        // 从 device 中取出 queue handle（queue 由 device 拥有，无需手动销毁）
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    // =================================================================================
    // 9) Swapchain
    // =================================================================================
    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        // 选择 format / present mode / extent
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        // imageCount 通常 = min+1（给 GPU 多一张缓冲减少等待）
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        // maxImageCount=0 表示不限制，否则要 clamp
        if (swapChainSupport.capabilities.maxImageCount > 0 &&
            imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;

        // 2D 图像，layer=1（VR/立体渲染可能 >1）
        createInfo.imageArrayLayers = 1;

        // 作为 color attachment 渲染目标
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        // 处理 graphics/present 不同 queue family 的情况
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        if (indices.graphicsFamily != indices.presentFamily) {
            // CONCURRENT：资源可同时被多个 queue family 使用（更方便，但可能性能差一些）
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            // EXCLUSIVE：同一时间只归一个 family 所有（通常性能更好）
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        // 变换：通常用当前变换（比如旋转屏幕）
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

        // 合成 alpha：通常 opaque
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        createInfo.presentMode = presentMode;

        // clipped=VK_TRUE：被遮挡的像素可以不计算（省性能）
        createInfo.clipped = VK_TRUE;

        // oldSwapchain：重建时可填旧 swapchain（教程省略，填 null）
        // createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        // 获取 swapchain images（由驱动创建）
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        // 记录 format/extent，后续创建 image view、framebuffer、viewport/scissor 需要
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    // =================================================================================
    // 10) Image views
    // =================================================================================
    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;

            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;

            // swizzle：保持原样
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            // 子资源范围：color、mip=0、level=1、layer=0、layerCount=1
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    // =================================================================================
    // 11) Render pass
    // =================================================================================
    void createRenderPass() {
        // 一个颜色 attachment（swapchain image）
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

        // loadOp=CLEAR：每帧开始清屏
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;

        // storeOp=STORE：渲染结果要保留给 present
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        // 不用 stencil
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        // 初始 layout 未定义（我们不关心旧内容）
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        // 最终 layout：present 引擎期望的 layout
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        // subpass 引用 attachment 0
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        // 一个 subpass：graphics
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        // subpass dependency：确保外部到 subpass 的同步（layout transition + 写入）
        // 这里是教程经典写法：让 color attachment output 阶段正确等待
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL; // “render pass 之外”
        dependency.dstSubpass = 0;                   // subpass 0
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    // =================================================================================
    // 12) Graphics pipeline
    // =================================================================================
    void createGraphicsPipeline() {
        // 读取 SPIR-V（由 glslc 编译出来）
        auto vertShaderCode = readFile("shaders/vert.vert.spv");
        auto fragShaderCode = readFile("shaders/frag.frag.spv");

        // 创建 shader module（只是 SPIR-V 的薄封装）
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        // shader stage：vertex
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main"; // entry point

        // shader stage：fragment
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {
            vertShaderStageInfo,
            fragShaderStageInfo
        };

        // ---- Vertex input：绑定描述 + 属性描述 ----
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        // ---- Input assembly：顶点如何组成 primitive ----
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;

        // TRIANGLE_LIST：每 3 个顶点一组，组成一个三角形（不共享边）
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        // primitiveRestart：用于 strip（如 triangle strip）插入特殊 index 重启
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // ---- Viewport / Scissor：这里做成 dynamic（运行时 vkCmdSetViewport/Scissor） ----
        // 注意：viewportState 里只填 “数量”，具体值在 command buffer 里 set
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        // ---- Rasterizer：图元光栅化配置 ----
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;       // 超出深度范围是否 clamp（需要 feature）
        rasterizer.rasterizerDiscardEnable = VK_FALSE;// true 则不输出任何 fragment（相当于关闭渲染）
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;// 填充（也可线框/点，需要 feature）
        rasterizer.lineWidth = 1.0f;                  // 线宽（>1 需要 wideLines feature）
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;  // 背面剔除
        // 注意：Vulkan 的 NDC Y 轴与 OpenGL 不同，教程里通常用 CLOCKWISE
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;        // 阴影贴图常用

        // ---- MSAA：本 demo 不开 ----
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // ---- Color blend：本 demo 直接覆盖，不做 alpha blending ----
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        // blendConstants 只有 blendEnable 时可能用到；这里保留默认
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        // ---- Dynamic states：告诉 pipeline 哪些状态由 command buffer 动态指定 ----
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        // ---- Pipeline layout：descriptor sets / push constants 的入口 ----
        // 目前没有 descriptor set / push constants，所以 layout 是空的
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0;
        pipelineLayoutInfo.pushConstantRangeCount = 0;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // ---- 最终创建 Graphics Pipeline ----
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;

        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;

        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        // basePipelineHandle 用于派生 pipeline（这里不用）
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        // shader module 在 pipeline 创建后即可销毁（机器码已被 pipeline “吸收/编译/链接”）
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    // =================================================================================
    // 13) Framebuffers：每个 swapchain image view 一个 framebuffer
    // =================================================================================
    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            // 本 demo 只有一个 color attachment
            VkImageView attachments[] = { swapChainImageViews[i] };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width  = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    // =================================================================================
    // 14) Command pool：绑定 graphics queue family
    // =================================================================================
    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

        // RESET_COMMAND_BUFFER_BIT：允许单独 reset 某个 command buffer（更灵活）
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics command pool!");
        }
    }

    // =================================================================================
    // 15) Vertex buffer（staging -> device local）
    // =================================================================================
    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        // ---- 1) staging buffer：HOST_VISIBLE|COHERENT，CPU 可 map 写入 ----
        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingBufferMemory = VK_NULL_HANDLE;

        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, // 作为 copy 源
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingBufferMemory
        );

        // map -> memcpy -> unmap
        // COHERENT 的好处：不需要手动 vkFlushMappedMemoryRanges
        void* data = nullptr;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // ---- 2) device local vertex buffer：GPU 读最快，但通常不能 map ----
        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, // 既能作为 copy 目的，又能作为 vertex buffer
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            vertexBuffer,
            vertexBufferMemory
        );

        // ---- 3) copy staging -> vertexBuffer ----
        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        // ---- 4) staging buffer 用完即销毁（只负责上传） ----
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // =================================================================================
    // 16) Index buffer（staging -> device local）
    // =================================================================================
    void createIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingBufferMemory = VK_NULL_HANDLE;

        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingBufferMemory
        );

        void* data = nullptr;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // 注意 usage：INDEX_BUFFER_BIT
        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            indexBuffer,
            indexBufferMemory
        );

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // =================================================================================
    // 17) createBuffer：把“创建 VkBuffer + 分配内存 + bind”抽成工具函数
    // =================================================================================
    void createBuffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkBuffer& buffer,
        VkDeviceMemory& bufferMemory
    ) {
        // VkBuffer 只是一个“资源描述/句柄”，不会自动分配内存
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;

        // EXCLUSIVE：只在一个 queue family 使用（本 demo 用 graphics queue）
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        // 查询内存需求：size/alignment/memoryTypeBits
        VkMemoryRequirements memRequirements{};
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        // 分配内存：allocationSize 可能 >= size（包含对齐/实现细节）
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;

        // memoryTypeIndex：必须同时满足
        //   - memRequirements.memoryTypeBits（buffer 允许的类型）
        //   - properties（我们期望的属性，比如 HOST_VISIBLE/DEVICE_LOCAL）
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        // 把这段内存绑定给 buffer（offset=0）
        // offset 必须是 memRequirements.alignment 的倍数
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    // =================================================================================
    // 18) copyBuffer：用一个一次性 command buffer 提交 vkCmdCopyBuffer
    // =================================================================================
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        // 分配一个临时 primary command buffer
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        // ONE_TIME_SUBMIT：提示驱动该命令缓冲只提交一次，有利于优化
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        // 定义复制区域：srcOffset/dstOffset/size
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0; // 可省略，默认 0
        copyRegion.dstOffset = 0;
        copyRegion.size = size;

        // 录制拷贝命令
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        vkEndCommandBuffer(commandBuffer);

        // 提交到队列执行
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);

        // 简化同步：等待队列空闲，确保 copy 完成后再返回
        // ⚠️性能提示：真实项目更倾向用 fence 等待，或者批量提交多个 copy 再等待
        vkQueueWaitIdle(graphicsQueue);

        // 释放临时 command buffer
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    // =================================================================================
    // 19) findMemoryType：在物理设备提供的内存类型中找一个满足要求的
    // =================================================================================
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        // 遍历 memoryTypes：最多通常几十个
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            // typeFilter 的第 i 位为 1 表示 buffer 允许使用该 memory type
            // propertyFlags 必须包含我们要求的所有 properties
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    // =================================================================================
    // 20) Command buffers：只分配，不在这里录制（每帧 drawFrame 里 reset+record）
    // =================================================================================
    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    // =================================================================================
    // 21) recordCommandBuffer：录制一帧的渲染命令
    // =================================================================================
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        // Begin command buffer recording
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        // 开始 render pass（指定 framebuffer / clear color）
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        // 清屏色：黑色
        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            // 绑定 pipeline（图形管线状态）
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            // 动态 viewport：与 swapchain extent 对齐
            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width  = (float)swapChainExtent.width;
            viewport.height = (float)swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            // 动态 scissor：裁剪区域（通常等于全屏）
            VkRect2D scissor{};
            scissor.offset = {0, 0};
            scissor.extent = swapChainExtent;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            // 绑定 vertex buffer（binding=0，对应 Vertex::getBindingDescription().binding）
            VkBuffer vertexBuffers[] = {vertexBuffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

            // 绑定 index buffer（只能绑定一个）
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

            // 按索引绘制：
            // indexCount = indices.size()
            // instanceCount = 1（不使用实例化）
            // firstIndex = 0（从第 0 个 index 开始）
            // vertexOffset = 0（索引加偏移，常用于大 buffer 子分配）
            // firstInstance = 0
            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        // End command buffer recording
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    // =================================================================================
    // 22) Sync objects：每帧一套（semaphore + fence）
    // =================================================================================
    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        // Fence 初始设为 signaled：
        // 这样第一帧 vkWaitForFences 不会卡住（否则第一帧还没 submit 过）
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    // =================================================================================
    // 23) drawFrame：一帧的典型流程
    //     wait fence -> acquire image -> record cmd -> submit -> present
    // =================================================================================
    void drawFrame() {
        // 1) CPU 等待该帧 fence（确保上一轮使用该 command buffer 的 GPU 工作已完成）
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // 2) 从 swapchain 获取一张可渲染的 image
        uint32_t imageIndex = 0;
        VkResult result = vkAcquireNextImageKHR(
            device,
            swapChain,
            UINT64_MAX,
            imageAvailableSemaphores[currentFrame], // acquire 完成后 signal
            VK_NULL_HANDLE,
            &imageIndex
        );

        // 可能发生：窗口 resize / surface 变化导致 swapchain 过期
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return; // 注意：这里 return 前我们没有 reset fence（因为还没 submit 新工作）
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // 3) 只有当我们确定要提交新工作时，才 reset fence
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // 4) 重录当前帧 command buffer
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        // 5) 提交：等待 imageAvailableSemaphore（图像可用），执行渲染，signal renderFinishedSemaphore
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };

        // waitStages：告诉 GPU “从哪个 pipeline stage 开始等”
        // 这里用 COLOR_ATTACHMENT_OUTPUT：让前面的顶点/装配等能与 acquire 重叠（更高并行）
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        // 提交到 graphicsQueue，提交完成后 signal fence（CPU 可等）
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        // 6) Present：等待 renderFinishedSemaphore，确保渲染完成后再 present
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        // present 也可能返回 out-of-date/suboptimal 或窗口 resize 触发重建
        if (result == VK_ERROR_OUT_OF_DATE_KHR ||
            result == VK_SUBOPTIMAL_KHR ||
            framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        // 7) 下一帧
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // =================================================================================
    // 24) Shader module helper
    // =================================================================================
    VkShaderModule createShaderModule(const std::vector<char>& code) {
        // 注意：pCode 需要 4 字节对齐，vector<char> 的 data 通常可满足（教程默认）
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }
        return shaderModule;
    }

    // =================================================================================
    // 25) Swapchain choose helpers
    // =================================================================================
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        // 优先选择：SRGB + 非线性 SRGB 色彩空间（教程推荐）
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
                availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }
        // 否则退回第一个
        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        // MAILBOX：低延迟、接近三缓冲（若可用就选）
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }
        // FIFO：保证存在（类似 vsync）
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        // 如果 currentExtent != UINT32_MAX，说明窗口系统已经给了固定 extent（必须用它）
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            // 否则需要自己从 GLFW 取 framebuffer size（注意是像素，不是逻辑窗口大小）
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            // clamp 到 allowed range
            actualExtent.width  = std::clamp(actualExtent.width,
                                             capabilities.minImageExtent.width,
                                             capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height,
                                             capabilities.minImageExtent.height,
                                             capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    // =================================================================================
    // 26) Query swapchain support
    // =================================================================================
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice dev) {
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

    // =================================================================================
    // 27) Device suitability: queue families + extensions + swapchain adequate
    // =================================================================================
    bool isDeviceSuitable(VkPhysicalDevice dev) {
        QueueFamilyIndices indices = findQueueFamilies(dev);

        bool extensionsSupported = checkDeviceExtensionSupport(dev);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(dev);
            swapChainAdequate = !swapChainSupport.formats.empty() &&
                                !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice dev) {
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

    // =================================================================================
    // 28) Find queue families: graphics + present
    // =================================================================================
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            // graphics 支持
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            // present 支持：需要针对 surface 查询
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

    // =================================================================================
    // 29) Required instance extensions
    // =================================================================================
    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        // 开启 validation 时需要 VK_EXT_debug_utils 扩展来创建 debug messenger
        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        return extensions;
    }

    // =================================================================================
    // 30) Validation layer support
    // =================================================================================
    bool checkValidationLayerSupport() {
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        // 逐个检查 validationLayers 是否存在
        for (const char* layerName : validationLayers) {
            bool layerFound = false;
            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound) return false;
        }
        return true;
    }

    // =================================================================================
    // 31) Read SPIR-V binary file
    // =================================================================================
    static std::vector<char> readFile(const std::string& filename) {
        // ate：打开后把读指针放到文件末尾，便于 tellg 获取大小
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            // 常见坑：工作目录不对（CLion/VS/命令行）
            throw std::runtime_error("failed to open file! (" + filename + ")");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }

    // =================================================================================
    // 32) Debug callback：validation layer 会把消息发到这里
    // =================================================================================
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
        VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* /*pUserData*/
    ) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        // 返回 VK_FALSE 表示不终止该 Vulkan 调用
        return VK_FALSE;
    }
};

// =====================================================================================
// main：创建 app 并运行；捕获异常打印错误
// =====================================================================================
int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
