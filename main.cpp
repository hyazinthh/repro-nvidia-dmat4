// Reproduces wrong dmat4 loads from a storage buffer whose bound base address is
// 16-byte but not 32-byte aligned (observed on NVIDIA, correct on AMD).
//
// The buffer holds 8 padding doubles (-1 .. -8) followed by 128 values 3k + 0.25.
// A compute shader loads dmat4 i from the bound range and stores its 16 components
// as doubles, so the output must equal the bound range verbatim.
//
// Two ways of producing a misaligned base are exercised:
//   1. VkDescriptorBufferInfo::offset = 8 .. 64 on a buffer bound at memory offset 0
//      (every offset that is a multiple of minStorageBufferOffsetAlignment is legal).
//   2. vkBindBufferMemory memoryOffset = 16 .. 48 with descriptor offset 0
//      (legal when memoryOffset is a multiple of VkMemoryRequirements::alignment).
//
// Usage: dmat4_repro [--gpu N] [--host] [--validate] [--list]
//   --gpu N     use physical device N (default: first with shaderFloat64)
//   --host      put the input buffers in host-visible memory instead of device-local
//   --validate  enable VK_LAYER_KHRONOS_validation and print its messages
//   --list      list physical devices and exit

#include <vulkan/vulkan.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/utsname.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(call)                                                                 \
    do {                                                                            \
        VkResult r_ = (call);                                                       \
        if (r_ != VK_SUCCESS) {                                                     \
            fprintf(stderr, "%s failed with VkResult %d (line %d)\n", #call, (int)r_, __LINE__); \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

static const int kPad      = 8;              // padding doubles in front of the values
static const int kMatrices = 8;              // dmat4 count copied by one dispatch
static const int kValues   = 16 * kMatrices; // doubles compared per case
static const int kDoubles  = kPad + kValues;

struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};
    VkPhysicalDeviceProperties props{};
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize memoryOffset = 0;
    VkMemoryRequirements requirements{};
};

static int g_passed = 0, g_failed = 0, g_skipped = 0;

// Outcome of the cases with a 16 mod 32 base, for the summary row.
static std::string g_a16 = "n/a", g_a48 = "n/a", g_b16 = "n/a", g_b48 = "n/a";

static std::string osName()
{
#ifdef _WIN32
    // RtlGetVersion reports the real version regardless of the manifest.
    typedef LONG(WINAPI* RtlGetVersionFn)(PRTL_OSVERSIONINFOW);
    RTL_OSVERSIONINFOW info{};
    info.dwOSVersionInfoSize = sizeof info;
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    RtlGetVersionFn fn = ntdll ? (RtlGetVersionFn)GetProcAddress(ntdll, "RtlGetVersion") : nullptr;
    if (fn && fn(&info) == 0) {
        char buf[64];
        snprintf(buf, sizeof buf, "Windows %lu.%lu build %lu",
                 (unsigned long)info.dwMajorVersion, (unsigned long)info.dwMinorVersion, (unsigned long)info.dwBuildNumber);
        return buf;
    }
    return "Windows";
#else
    utsname u{};
    if (uname(&u) == 0) return std::string(u.sysname) + " " + u.release;
    return "unknown";
#endif
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void*)
{
    const char* level = severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "ERROR" : "WARNING";
    printf("  [validation %s] %s\n", level, data->pMessage);
    return VK_FALSE;
}

static bool hasLayer(const char* name)
{
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (auto& l : layers) if (strcmp(l.layerName, name) == 0) return true;
    return false;
}

static bool hasInstanceExtension(const char* name)
{
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, exts.data());
    for (auto& e : exts) if (strcmp(e.extensionName, name) == 0) return true;
    return false;
}

static std::string driverVersionString(const VkPhysicalDeviceProperties& p)
{
    char buf[64];
    uint32_t v = p.driverVersion;
    if (p.vendorID == 0x10DE) // NVIDIA: 10.8.8.6 bits
        snprintf(buf, sizeof buf, "%u.%u.%u.%u", v >> 22, (v >> 14) & 0xFF, (v >> 6) & 0xFF, v & 0x3F);
#ifdef _WIN32
    else if (p.vendorID == 0x8086) // Intel on Windows: 18.14 bits
        snprintf(buf, sizeof buf, "%u.%u", v >> 14, v & 0x3FFF);
#endif
    else
        snprintf(buf, sizeof buf, "%u.%u.%u", VK_VERSION_MAJOR(v), VK_VERSION_MINOR(v), VK_VERSION_PATCH(v));
    return buf;
}

static void printDevice(VkPhysicalDevice pd, int index)
{
    VkPhysicalDeviceDriverProperties driver{};
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    bool haveDriverProps = props.apiVersion >= VK_API_VERSION_1_2;
    if (haveDriverProps) {
        props2.pNext = &driver;
        vkGetPhysicalDeviceProperties2(pd, &props2);
    }

    VkPhysicalDeviceFeatures features{};
    vkGetPhysicalDeviceFeatures(pd, &features);

    printf("device %d: %s\n", index, props.deviceName);
    printf("  vendor 0x%04X, device 0x%04X, type %d\n", props.vendorID, props.deviceID, (int)props.deviceType);
    printf("  Vulkan %u.%u.%u, driver version %s\n",
           VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion),
           driverVersionString(props).c_str());
    if (haveDriverProps)
        printf("  driver: %s (%s)\n", driver.driverName, driver.driverInfo);
    printf("  shaderFloat64: %s\n", features.shaderFloat64 ? "yes" : "no");
    printf("  minStorageBufferOffsetAlignment: %llu\n", (unsigned long long)props.limits.minStorageBufferOffsetAlignment);
}

static uint32_t findMemoryType(const Context& ctx, uint32_t typeBits, VkMemoryPropertyFlags required)
{
    for (uint32_t i = 0; i < ctx.memory.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (ctx.memory.memoryTypes[i].propertyFlags & required) == required)
            return i;
    return UINT32_MAX;
}

// Creates a buffer and binds it to fresh memory at `memoryOffset`. Returns false
// (and creates nothing) when the offset violates the buffer's alignment requirement.
static bool createBuffer(const Context& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags props, VkDeviceSize memoryOffset, Buffer& out)
{
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK(vkCreateBuffer(ctx.device, &info, nullptr, &out.buffer));

    vkGetBufferMemoryRequirements(ctx.device, out.buffer, &out.requirements);
    if (memoryOffset % out.requirements.alignment != 0) {
        vkDestroyBuffer(ctx.device, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return false;
    }

    uint32_t type = findMemoryType(ctx, out.requirements.memoryTypeBits, props);
    if (type == UINT32_MAX) {
        fprintf(stderr, "no memory type with properties 0x%X for buffer\n", props);
        exit(1);
    }

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = out.requirements.size + memoryOffset;
    alloc.memoryTypeIndex = type;
    CHECK(vkAllocateMemory(ctx.device, &alloc, nullptr, &out.memory));
    CHECK(vkBindBufferMemory(ctx.device, out.buffer, out.memory, memoryOffset));
    out.size = size;
    out.memoryOffset = memoryOffset;
    return true;
}

static void destroyBuffer(const Context& ctx, Buffer& b)
{
    if (b.buffer) vkDestroyBuffer(ctx.device, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(ctx.device, b.memory, nullptr);
    b = Buffer{};
}

static void submitAndWait(const Context& ctx)
{
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &ctx.cmd;
    CHECK(vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(ctx.queue));
}

static void beginCommands(const Context& ctx)
{
    CHECK(vkResetCommandBuffer(ctx.cmd, 0));
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkBeginCommandBuffer(ctx.cmd, &begin));
}

// Copies `data` into `dst` through a staging buffer.
static void upload(const Context& ctx, const Buffer& dst, const void* data, VkDeviceSize size)
{
    Buffer staging;
    createBuffer(ctx, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, staging);
    void* mapped = nullptr;
    CHECK(vkMapMemory(ctx.device, staging.memory, 0, size, 0, &mapped));
    memcpy(mapped, data, (size_t)size);
    vkUnmapMemory(ctx.device, staging.memory);

    beginCommands(ctx);
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(ctx.cmd, staging.buffer, dst.buffer, 1, &region);
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = dst.buffer;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(ctx.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 1, &barrier, 0, nullptr);
    CHECK(vkEndCommandBuffer(ctx.cmd));
    submitAndWait(ctx);
    destroyBuffer(ctx, staging);
}

static std::vector<uint32_t> readSpirv()
{
    const char* candidates[] = { SPV_PATH, "copy.spv" };
    for (const char* path : candidates) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) continue;
        size_t size = (size_t)f.tellg();
        std::vector<uint32_t> words((size + 3) / 4);
        f.seekg(0);
        f.read(reinterpret_cast<char*>(words.data()), size);
        return words;
    }
    fprintf(stderr, "copy.spv not found (looked at %s and the working directory)\n", SPV_PATH);
    exit(1);
}

static void createPipeline(Context& ctx)
{
    VkDescriptorSetLayoutBinding bindings[2]{};
    for (int i = 0; i < 2; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    CHECK(vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr, &ctx.setLayout));

    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t)};
    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &ctx.setLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &push;
    CHECK(vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &ctx.pipelineLayout));

    std::vector<uint32_t> code = readSpirv();
    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = code.size() * sizeof(uint32_t);
    moduleInfo.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    CHECK(vkCreateShaderModule(ctx.device, &moduleInfo, nullptr, &module));

    VkComputePipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.module = module;
    pipeInfo.stage.pName = "main";
    pipeInfo.layout = ctx.pipelineLayout;
    CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &ctx.pipeline));
    vkDestroyShaderModule(ctx.device, module, nullptr);

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    CHECK(vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &ctx.descriptorPool));

    VkDescriptorSetAllocateInfo setInfo{};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setInfo.descriptorPool = ctx.descriptorPool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &ctx.setLayout;
    CHECK(vkAllocateDescriptorSets(ctx.device, &setInfo, &ctx.set));
}

// Dispatches the copy over `input` bound at `descriptorOffset` and compares the
// output with the buffer contents from that offset on.
static std::string runCase(const Context& ctx, const char* label, const Buffer& input, VkDeviceSize descriptorOffset,
                           const Buffer& output, const double* data)
{
    // zero the output
    void* mapped = nullptr;
    CHECK(vkMapMemory(ctx.device, output.memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    memset(mapped, 0, (size_t)output.size);
    vkUnmapMemory(ctx.device, output.memory);

    VkDescriptorBufferInfo infos[2]{};
    infos[0] = {input.buffer, descriptorOffset, VK_WHOLE_SIZE};
    infos[1] = {output.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet writes[2]{};
    for (int i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ctx.set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);

    beginCommands(ctx);
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipelineLayout, 0, 1, &ctx.set, 0, nullptr);
    int32_t count = kMatrices;
    vkCmdPushConstants(ctx.cmd, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof count, &count);
    vkCmdDispatch(ctx.cmd, 1, 1, 1);
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = output.buffer;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(ctx.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, nullptr, 1, &barrier, 0, nullptr);
    CHECK(vkEndCommandBuffer(ctx.cmd));
    submitAndWait(ctx);

    std::vector<double> result(kValues);
    CHECK(vkMapMemory(ctx.device, output.memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    memcpy(result.data(), mapped, kValues * sizeof(double));
    vkUnmapMemory(ctx.device, output.memory);

    const int base = (int)(descriptorOffset / sizeof(double));
    const double* expected = data + base;
    int firstMismatch = -1;
    for (int k = 0; k < kValues; ++k)
        if (result[k] != expected[k]) { firstMismatch = k; break; }

    if (firstMismatch < 0) {
        printf("  PASS  %s\n", label);
        ++g_passed;
        return "pass";
    }

    ++g_failed;
    printf("  FAIL  %s\n", label);
    printf("        first mismatch at double %d: expected %g, got %g\n",
           firstMismatch, expected[firstMismatch], result[firstMismatch]);
    printf("        expected: ");
    for (int k = 0; k < 8; ++k) printf("%g ", expected[k]);
    printf("...\n        output:   ");
    for (int k = 0; k < 8; ++k) printf("%g ", result[k]);
    printf("...\n");

    // Does the output equal the buffer contents read from a shifted address?
    for (int shift = -8; shift <= 8; ++shift) {
        if (shift == 0) continue;
        bool match = true;
        for (int k = 0; k < kValues && match; ++k) {
            int idx = base + k - shift;
            if (idx < 0 || idx >= kDoubles) continue;   // beyond the buffer: skip
            if (result[k] != data[idx]) match = false;
        }
        if (match) {
            printf("        output equals the buffer contents read %d bytes %s\n",
                   abs(shift) * 8, shift > 0 ? "EARLY (before the bound base)" : "late");
            char buf[48];
            snprintf(buf, sizeof buf, "FAIL (%d B %s)", abs(shift) * 8, shift > 0 ? "early" : "late");
            return buf;
        }
    }
    return "FAIL";
}

int main(int argc, char** argv)
{
    int gpuIndex = -1;
    bool hostInput = false, validate = false, list = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gpu" && i + 1 < argc) gpuIndex = atoi(argv[++i]);
        else if (a == "--host") hostInput = true;
        else if (a == "--validate") validate = true;
        else if (a == "--list") list = true;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
    }

    Context ctx;

    // ---- instance ----
    std::vector<const char*> layers, extensions;
    bool debugUtils = false;
    if (validate) {
        if (hasLayer("VK_LAYER_KHRONOS_validation")) layers.push_back("VK_LAYER_KHRONOS_validation");
        else printf("VK_LAYER_KHRONOS_validation not available, running without it\n");
        if (hasInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            debugUtils = true;
        }
    }
    uint32_t loaderVersion = VK_API_VERSION_1_0;
    auto enumerateVersion = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion");
    if (enumerateVersion) enumerateVersion(&loaderVersion);
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "dmat4_repro";
    app.apiVersion = loaderVersion >= VK_API_VERSION_1_2 ? VK_API_VERSION_1_2 : VK_API_VERSION_1_1;
    VkInstanceCreateInfo instInfo{};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &app;
    instInfo.enabledLayerCount = (uint32_t)layers.size();
    instInfo.ppEnabledLayerNames = layers.data();
    instInfo.enabledExtensionCount = (uint32_t)extensions.size();
    instInfo.ppEnabledExtensionNames = extensions.data();
    CHECK(vkCreateInstance(&instInfo, nullptr, &ctx.instance));

    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    if (debugUtils) {
        auto create = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(ctx.instance, "vkCreateDebugUtilsMessengerEXT");
        VkDebugUtilsMessengerCreateInfoEXT dbg{};
        dbg.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dbg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dbg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
        dbg.pfnUserCallback = debugCallback;
        if (create) CHECK(create(ctx.instance, &dbg, nullptr, &messenger));
    }

    // ---- physical device ----
    uint32_t count = 0;
    CHECK(vkEnumeratePhysicalDevices(ctx.instance, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    CHECK(vkEnumeratePhysicalDevices(ctx.instance, &count, devices.data()));
    if (devices.empty()) { fprintf(stderr, "no Vulkan devices\n"); return 1; }

    if (list) {
        for (uint32_t i = 0; i < count; ++i) printDevice(devices[i], (int)i);
        return 0;
    }

    if (gpuIndex < 0) {
        for (uint32_t i = 0; i < count; ++i) {
            VkPhysicalDeviceFeatures f{};
            vkGetPhysicalDeviceFeatures(devices[i], &f);
            if (f.shaderFloat64) { gpuIndex = (int)i; break; }
        }
        if (gpuIndex < 0) { fprintf(stderr, "no device supports shaderFloat64\n"); return 2; }
    }
    if (gpuIndex >= (int)count) { fprintf(stderr, "no device %d\n", gpuIndex); return 1; }
    ctx.physical = devices[gpuIndex];
    printDevice(ctx.physical, gpuIndex);
    vkGetPhysicalDeviceProperties(ctx.physical, &ctx.props);
    vkGetPhysicalDeviceMemoryProperties(ctx.physical, &ctx.memory);

    VkPhysicalDeviceFeatures features{};
    vkGetPhysicalDeviceFeatures(ctx.physical, &features);
    if (!features.shaderFloat64) { fprintf(stderr, "device %d does not support shaderFloat64\n", gpuIndex); return 2; }

    // ---- device and queue ----
    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical, &familyCount, families.data());
    ctx.queueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < familyCount; ++i)
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { ctx.queueFamily = i; break; }
    if (ctx.queueFamily == UINT32_MAX) { fprintf(stderr, "no compute queue\n"); return 1; }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = ctx.queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures enabled{};
    enabled.shaderFloat64 = VK_TRUE;
    VkDeviceCreateInfo devInfo{};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &queueInfo;
    devInfo.pEnabledFeatures = &enabled;
    CHECK(vkCreateDevice(ctx.physical, &devInfo, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queueFamily, 0, &ctx.queue);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = ctx.queueFamily;
    CHECK(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.pool));
    VkCommandBufferAllocateInfo cmdInfo{};
    cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdInfo.commandPool = ctx.pool;
    cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdInfo.commandBufferCount = 1;
    CHECK(vkAllocateCommandBuffers(ctx.device, &cmdInfo, &ctx.cmd));

    createPipeline(ctx);

    // ---- data and buffers ----
    std::vector<double> data(kDoubles);
    for (int i = 0; i < kPad; ++i) data[i] = -1.0 - i;
    for (int k = 0; k < kValues; ++k) data[kPad + k] = 3.0 * k + 0.25;
    const VkDeviceSize dataBytes = kDoubles * sizeof(double);

    const VkBufferUsageFlags inputUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkMemoryPropertyFlags inputMemory = hostInput
        ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    Buffer output;
    createBuffer(ctx, kValues * sizeof(double), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, output);

    Buffer input;
    createBuffer(ctx, dataBytes, inputUsage, inputMemory, 0, input);
    upload(ctx, input, data.data(), dataBytes);
    printf("  input buffer: %llu bytes, usage STORAGE|TRANSFER_DST, VkMemoryRequirements.alignment = %llu, memory %s\n",
           (unsigned long long)dataBytes, (unsigned long long)input.requirements.alignment,
           hostInput ? "host-visible" : "device-local");

    const VkDeviceSize storageAlignment = ctx.props.limits.minStorageBufferOffsetAlignment;

    printf("\nA. descriptor offset on a buffer bound at memory offset 0\n");
    for (VkDeviceSize offset : {0, 8, 16, 24, 32, 48, 64}) {
        char label[96];
        snprintf(label, sizeof label, "descriptor offset %2llu (base %s 32-byte aligned)",
                 (unsigned long long)offset, offset % 32 == 0 ? "is" : "NOT");
        std::string outcome;
        if (offset % storageAlignment != 0) {
            printf("  SKIP  %s: not a multiple of minStorageBufferOffsetAlignment\n", label);
            ++g_skipped;
            outcome = "skip";
        } else {
            outcome = runCase(ctx, label, input, offset, output, data.data());
        }
        if (offset == 16) g_a16 = outcome;
        if (offset == 48) g_a48 = outcome;
    }

    printf("\nB. descriptor offset 0 on a buffer bound at a memory offset\n");
    for (VkDeviceSize memoryOffset : {16, 32, 48}) {
        char label[96];
        snprintf(label, sizeof label, "memory offset %2llu (base %s 32-byte aligned)",
                 (unsigned long long)memoryOffset, memoryOffset % 32 == 0 ? "is" : "NOT");
        Buffer shifted;
        std::string outcome;
        if (!createBuffer(ctx, dataBytes, inputUsage, inputMemory, memoryOffset, shifted)) {
            printf("  SKIP  %s: not a multiple of VkMemoryRequirements.alignment (%llu)\n",
                   label, (unsigned long long)input.requirements.alignment);
            ++g_skipped;
            outcome = "skip";
        } else {
            upload(ctx, shifted, data.data(), dataBytes);
            outcome = runCase(ctx, label, shifted, 0, output, data.data());
            destroyBuffer(ctx, shifted);
        }
        if (memoryOffset == 16) g_b16 = outcome;
        if (memoryOffset == 48) g_b48 = outcome;
    }

    printf("\n%d passed, %d failed, %d skipped\n", g_passed, g_failed, g_skipped);

    // one markdown row for the README's results table
    VkPhysicalDeviceDriverProperties driver{};
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    std::string driverText = driverVersionString(ctx.props);
    if (ctx.props.apiVersion >= VK_API_VERSION_1_2) {
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &driver;
        vkGetPhysicalDeviceProperties2(ctx.physical, &props2);
        driverText = std::string(driver.driverName) + " " + driver.driverInfo;
    }
    printf("\nresults row (paste into README.md):\n");
    printf("| %s | %s | %s | %llu | %llu | %s | %s | %s | %s |\n",
           ctx.props.deviceName, osName().c_str(), driverText.c_str(),
           (unsigned long long)storageAlignment, (unsigned long long)input.requirements.alignment,
           g_a16.c_str(), g_a48.c_str(), g_b16.c_str(), g_b48.c_str());

    destroyBuffer(ctx, input);
    destroyBuffer(ctx, output);
    vkDestroyDescriptorPool(ctx.device, ctx.descriptorPool, nullptr);
    vkDestroyPipeline(ctx.device, ctx.pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, ctx.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, ctx.setLayout, nullptr);
    vkDestroyCommandPool(ctx.device, ctx.pool, nullptr);
    vkDestroyDevice(ctx.device, nullptr);
    if (messenger) {
        auto destroy = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(ctx.instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroy) destroy(ctx.instance, messenger, nullptr);
    }
    vkDestroyInstance(ctx.instance, nullptr);
    return g_failed == 0 ? 0 : 1;
}
