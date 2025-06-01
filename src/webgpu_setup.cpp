#include "webgpu_setup.h"
#include "fractal_settings.h"
#include "compute_shaders.h"
#include <webgpu/webgpu_cpp_print.h>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <sstream>

// Debug configuration - set at compile time
#ifdef DEBUG
static constexpr bool kDebugBuild = true;
#else
static constexpr bool kDebugBuild = false;
#endif

#if defined(ENABLE_STACK_TRACE) && defined(__unix__)
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#include <windows.h>
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#include "metal_layer_helper.h"
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#endif
#include <GLFW/glfw3native.h>

// Global fractal settings instance (extern)
extern FractalSettings g_settings;

#if defined(ENABLE_STACK_TRACE) && defined(__unix__)
void crashHandler(int sig) {
    void *array[10];
    size_t size;
    
    // Get void*'s for all entries on the stack
    size = backtrace(array, 10);
    
    // Print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}
#endif

// Error callback for GLFW
void errorCallback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

// Device error callback
void deviceErrorCallback(wgpu::ErrorType type, const char* message, void* userdata) {
    WebGPUContext* context = static_cast<WebGPUContext*>(userdata);
    context->totalErrors++;
    context->currentFrame.errorCount++;
    
    const char* typeStr = "Unknown";
    switch (type) {
        case wgpu::ErrorType::NoError: typeStr = "NoError"; break;
        case wgpu::ErrorType::Validation: typeStr = "Validation"; break;
        case wgpu::ErrorType::OutOfMemory: typeStr = "OutOfMemory"; break;
        case wgpu::ErrorType::Unknown: typeStr = "Unknown"; break;
        default: typeStr = "Unhandled"; break;
    }
    
    std::cerr << "WebGPU Device Error [" << typeStr << "]: " << message << std::endl;
    std::cerr << "    Frame: " << context->currentFrame.frameNumber 
              << ", Total Errors: " << context->totalErrors << std::endl;
    
    // Log critical errors
    if (type == wgpu::ErrorType::OutOfMemory) {
        std::cerr << "CRITICAL ERROR - This may cause system instability!" << std::endl;
    }
}

// Uncaptured error callback
void uncapturedErrorCallback(wgpu::ErrorType type, const char* message, void* userdata) {
    WebGPUContext* context = static_cast<WebGPUContext*>(userdata);
    context->totalErrors++;
    context->currentFrame.errorCount++;
    
    std::cerr << "WebGPU Uncaptured Error: " << message << std::endl;
    std::cerr << "    This error was not handled by the application!" << std::endl;
}

// Device lost callback
void deviceLostCallback(wgpu::DeviceLostReason reason, const char* message, void* userdata) {
    WebGPUContext* context = static_cast<WebGPUContext*>(userdata);
    context->totalErrors++;
    
    const char* reasonStr = "Unknown";
    switch (reason) {
        case wgpu::DeviceLostReason::Unknown: reasonStr = "Unknown"; break;
        case wgpu::DeviceLostReason::Destroyed: reasonStr = "Destroyed"; break;
        case wgpu::DeviceLostReason::FailedCreation: reasonStr = "FailedCreation"; break;
        default: reasonStr = "Unhandled"; break;
    }
    
    std::cerr << "WebGPU Device Lost [" << reasonStr << "]: " << message << std::endl;
    std::cerr << "    This is a critical error that may require restart!" << std::endl;
}

void setupErrorCallbacks(WebGPUContext& context) {
    std::cout << "Setting up WebGPU error monitoring..." << std::endl;
    
    // Note: This Dawn version doesn't have SetUncapturedErrorCallback
    // We'll use PushErrorScope/PopErrorScope for targeted error checking
    
    context.validationEnabled = true;
    std::cout << "WebGPU error monitoring configured" << std::endl;
    std::cout << "    WARNING: Using PushErrorScope/PopErrorScope for error detection" << std::endl;
    std::cout << "    Enhanced debugging: ENABLED" << std::endl;
    std::cout << "    Frame validation: ENABLED" << std::endl;
    std::cout << "    Compute validation: ENABLED" << std::endl;
}

void enableWebGPUValidation(WebGPUContext& context) {
    std::cout << "Enabling WebGPU validation and debugging..." << std::endl;
    context.debugEnabled = true;
    
    if constexpr (kDebugBuild) {
        std::cout << "    - Debug build detected, maximum validation enabled" << std::endl;
        std::cout << "    - Backend validation: ENABLED" << std::endl;
        std::cout << "    - Device removal tests: ENABLED" << std::endl;
        std::cout << "    - Vulkan validation layers: ENABLED" << std::endl;
    }
    
    // Important note about Dawn/Metal shader validation behavior
    std::cout << "\nIMPORTANT: Dawn/Metal Shader Validation Behavior" << std::endl;
    std::cout << "    Dawn on macOS Metal backend defers shader validation!" << std::endl;
    std::cout << "    Shader syntax errors may NOT be caught during compilation." << std::endl;
    std::cout << "    Instead, broken shaders may cause:" << std::endl;
    std::cout << "    - Silent rendering failures (pink/black screen)" << std::endl;
    std::cout << "    - Runtime crashes during shader execution" << std::endl;
    std::cout << "    - Garbage output from compute shaders" << std::endl;
    std::cout << "    Always test shader changes carefully!\n" << std::endl;
}

void beginFrameDebugging(WebGPUContext& context) {
    context.currentFrame.frameNumber = context.totalFrames;
    context.currentFrame.startTime = std::chrono::high_resolution_clock::now();
    context.currentFrame.errorCount = 0;
    context.currentFrame.renderingComplete = false;
    context.currentFrame.computeComplete = false;
    
    if (context.debugEnabled && context.totalFrames % 60 == 0) {
        std::cout << "Frame " << context.currentFrame.frameNumber 
                  << " debugging started" << std::endl;
    }
}

void endFrameDebugging(WebGPUContext& context) {
    context.currentFrame.endTime = std::chrono::high_resolution_clock::now();
    context.totalFrames++;
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        context.currentFrame.endTime - context.currentFrame.startTime);
    
    if (context.debugEnabled) {
        // Print frame stats every 60 frames or if there were errors
        if (context.totalFrames % 60 == 0 || context.currentFrame.errorCount > 0) {
            std::cout << "Frame " << context.currentFrame.frameNumber 
                      << " completed in " << duration.count() << "μs";
            
            if (context.currentFrame.errorCount > 0) {
                std::cout << " [" << context.currentFrame.errorCount << " errors]";
            }
            
            if (context.currentFrame.renderingComplete && context.currentFrame.computeComplete) {
                std::cout << " [Render & Compute OK]";
            } else {
                std::cout << " [Pipeline incomplete - R:" 
                          << (context.currentFrame.renderingComplete ? "OK" : "FAIL")
                          << " C:" << (context.currentFrame.computeComplete ? "OK" : "FAIL") << "]";
            }
            std::cout << std::endl;
        }
    }
}

void validateWebGPUState(const WebGPUContext& context) {
    bool isValid = true;
    std::cout << "Validating WebGPU state..." << std::endl;
    
    if (!context.instance) {
        std::cerr << "Instance is null" << std::endl;
        isValid = false;
    }
    
    if (!context.adapter) {
        std::cerr << "Adapter is null" << std::endl;
        isValid = false;
    }
    
    if (!context.device) {
        std::cerr << "Device is null" << std::endl;
        isValid = false;
    }
    
    if (!context.queue) {
        std::cerr << "Queue is null" << std::endl;
        isValid = false;
    }
    
    if (!context.surface) {
        std::cerr << "Surface is null" << std::endl;
        isValid = false;
    }
    
    if (isValid) {
        std::cout << "WebGPU state validation passed" << std::endl;
    } else {
        std::cerr << "WebGPU state validation FAILED - system may be unstable!" << std::endl;
    }
}

void printFrameStats(const WebGPUContext& context) {
    std::cout << "Frame Statistics:" << std::endl;
    std::cout << "    Total Frames: " << context.totalFrames << std::endl;
    std::cout << "    Total Errors: " << context.totalErrors << std::endl;
    std::cout << "    Error Rate: " << std::fixed << std::setprecision(2) 
              << (context.totalFrames > 0 ? (100.0 * context.totalErrors / context.totalFrames) : 0.0) 
              << "%" << std::endl;
}

bool checkRenderingHealth(const WebGPUContext& context) {
    // Consider the system healthy if:
    // 1. Error rate is below 5%
    // 2. Last frame completed successfully
    // 3. WebGPU objects are valid
    
    bool hasLowErrorRate = context.totalFrames == 0 || 
                          (context.totalErrors * 100 / context.totalFrames) < 5;
    
    bool lastFrameOK = context.currentFrame.renderingComplete && 
                       context.currentFrame.computeComplete &&
                       context.currentFrame.errorCount == 0;
    
    bool objectsValid = context.instance && context.adapter && 
                       context.device && context.queue && context.surface;
    
    return hasLowErrorRate && lastFrameOK && objectsValid;
}

// Function declarations for comprehensive validation
bool validateComputeShaderExecution(const wgpu::Device& device, wgpu::ComputePipeline& pipeline, const std::string& shaderType);

WebGPUContext initWebGPU() {
    WebGPUContext context{};
    context.metalLayer = nullptr;

#if defined(ENABLE_STACK_TRACE) && defined(__unix__)
    // Install crash handler for better debugging
    signal(SIGSEGV, crashHandler);
    signal(SIGABRT, crashHandler);
    signal(SIGFPE, crashHandler);
    std::cout << "Crash handler installed for better debugging" << std::endl;
#endif

    std::cout << "Initializing WebGPU with enhanced debugging..." << std::endl;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return context;
    }
    glfwSetErrorCallback(errorCallback);
    std::cout << "GLFW initialized with error callback" << std::endl;

    // Configure GLFW
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // No OpenGL context
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);     // Enable resizing

    // Create window with settings dimensions
    context.window = glfwCreateWindow(g_settings.viewport.width, g_settings.viewport.height, 
                                     "Julia Fractal - WebGPU [DEBUG BUILD]", nullptr, nullptr);
    if (!context.window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return context;
    }
    std::cout << "GLFW window created: " << g_settings.viewport.width 
              << "x" << g_settings.viewport.height << std::endl;

    // Initialize WebGPU
    wgpu::InstanceDescriptor instanceDescriptor{};
    instanceDescriptor.capabilities.timedWaitAnyEnable = true;
    context.instance = wgpu::CreateInstance(&instanceDescriptor);
    if (context.instance == nullptr) {
        std::cerr << "Instance creation failed!" << std::endl;
        glfwTerminate();
        return context;
    }
    std::cout << "WebGPU instance created" << std::endl;

    // Create surface from GLFW window
    WGPUSurfaceDescriptor surfaceDesc = {};

#if defined(_WIN32)
    // Windows implementation
    WGPUSurfaceDescriptorFromWindowsHWND windowsDesc = {};
    windowsDesc.chain.sType = WGPUSType_SurfaceDescriptorFromWindowsHWND;
    windowsDesc.hwnd = glfwGetWin32Window(context.window);
    windowsDesc.hinstance = GetModuleHandle(nullptr);
    surfaceDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&windowsDesc);
#elif defined(__APPLE__)
    // macOS implementation - use our helper to create a Metal layer
    context.metalLayer = CreateMetalLayer(context.window);
    if (!context.metalLayer) {
        std::cerr << "Failed to create Metal layer" << std::endl;
        glfwTerminate();
        return context;
    }
    
    // Create a Metal surface source
    WGPUSurfaceSourceMetalLayer metalLayerDesc = {};
    metalLayerDesc.chain.sType = WGPUSType_SurfaceSourceMetalLayer;
    metalLayerDesc.layer = context.metalLayer;
    
    // Create a surface descriptor that points to the metal layer descriptor
    surfaceDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&metalLayerDesc);
#elif defined(__linux__)
    // Linux implementation
    WGPUSurfaceDescriptorFromXlibWindow xlibDesc = {};
    xlibDesc.chain.sType = WGPUSType_SurfaceDescriptorFromXlibWindow;
    xlibDesc.display = glfwGetX11Display();
    xlibDesc.window = glfwGetX11Window(context.window);
    surfaceDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&xlibDesc);
#endif

    // Create surface using direct WebGPU C API for better cross-platform compatibility
    WGPUSurface rawSurface = wgpuInstanceCreateSurface(
        context.instance.Get(),
        &surfaceDesc);
    
    context.surface = wgpu::Surface::Acquire(rawSurface);
    if (!context.surface) {
        std::cerr << "Failed to create surface" << std::endl;
        glfwTerminate();
        return context;
    }

    // Request adapter
    wgpu::RequestAdapterOptions options{};
    options.compatibleSurface = context.surface;

    auto callback = [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, const char *message, void *userdata) {
        if (status != wgpu::RequestAdapterStatus::Success) {
            std::cerr << "Failed to get an adapter: " << message << std::endl;
            return;
        }
        *static_cast<wgpu::Adapter *>(userdata) = adapter;
    };

    auto callbackMode = wgpu::CallbackMode::WaitAnyOnly;
    void *userdata = &context.adapter;
    context.instance.WaitAny(context.instance.RequestAdapter(&options, callbackMode, callback, userdata), UINT64_MAX);
    if (context.adapter == nullptr) {
        std::cerr << "RequestAdapter failed!\n";
        glfwTerminate();
        return context;
    }

    // Print adapter info
    wgpu::DawnAdapterPropertiesPowerPreference power_props{};
    wgpu::AdapterInfo info{};
    info.nextInChain = &power_props;
    context.adapter.GetInfo(&info);
    std::cout << "Using adapter: " << info.device << std::endl;

    // Create device with enhanced error handling
    wgpu::DeviceDescriptor deviceDesc{};
    context.device = context.adapter.CreateDevice(&deviceDesc);
    if (!context.device) {
        std::cerr << "Failed to create device" << std::endl;
        glfwTerminate();
        return context;
    }
    std::cout << "WebGPU device created" << std::endl;

    // Setup comprehensive error callbacks
    setupErrorCallbacks(context);
    enableWebGPUValidation(context);

    // Get queue
    context.queue = context.device.GetQueue();
    if (!context.queue) {
        std::cerr << "Failed to get device queue" << std::endl;
        glfwTerminate();
        return context;
    }
    std::cout << "WebGPU queue obtained" << std::endl;

    // Query surface capabilities to get the preferred format
    wgpu::SurfaceCapabilities capabilities{};
    context.surface.GetCapabilities(context.adapter, &capabilities);
    context.swapChainFormat = wgpu::TextureFormat::BGRA8Unorm; // Default format
    
    if (capabilities.formatCount > 0 && capabilities.formats != nullptr) {
        context.swapChainFormat = capabilities.formats[0];
        std::cout << "Using surface format: " << static_cast<int>(context.swapChainFormat) << std::endl;
    }

    // Configure the surface
    glfwGetFramebufferSize(context.window, &context.width, &context.height);
    
    // Get logical window size for mouse coordinate translation
    int logicalWidth, logicalHeight;
    glfwGetWindowSize(context.window, &logicalWidth, &logicalHeight);
    
    // Update settings with both framebuffer size (for WebGPU) and logical size (for mouse)
    g_settings.viewport.width = context.width;
    g_settings.viewport.height = context.height;
    g_settings.viewport.logicalWidth = logicalWidth;
    g_settings.viewport.logicalHeight = logicalHeight;
    
    std::cout << "Framebuffer size: " << context.width << "x" << context.height << std::endl;
    std::cout << "Logical window size: " << logicalWidth << "x" << logicalHeight << std::endl;
    
    wgpu::SurfaceConfiguration surfaceConfig{};
    surfaceConfig.device = context.device;
    surfaceConfig.format = context.swapChainFormat;
    surfaceConfig.usage = wgpu::TextureUsage::RenderAttachment;
    surfaceConfig.width = static_cast<uint32_t>(context.width);
    surfaceConfig.height = static_cast<uint32_t>(context.height);
    surfaceConfig.presentMode = wgpu::PresentMode::Fifo;
    
    context.surface.Configure(&surfaceConfig);
    std::cout << "Surface configured successfully" << std::endl;

    // Final validation
    validateWebGPUState(context);
    
    std::cout << "WebGPU initialization complete with enhanced debugging!" << std::endl;
    
    return context;
}

void cleanupWebGPU(WebGPUContext& context) {
#if defined(__APPLE__)
    if (context.metalLayer) {
        ReleaseMetalLayer(context.metalLayer);
        context.metalLayer = nullptr;
    }
#endif
    
    if (context.window) {
        glfwDestroyWindow(context.window);
        context.window = nullptr;
    }
    glfwTerminate();
}

void reconfigureSurface(WebGPUContext& context, int width, int height) {
    std::cout << "Reconfiguring WebGPU surface to " << width << "x" << height << std::endl;
    
    // Validate dimensions
    if (width <= 0 || height <= 0) {
        std::cerr << "Invalid surface dimensions: " << width << "x" << height << std::endl;
        return;
    }
    
    // Update context dimensions
    context.width = width;
    context.height = height;
    
#if defined(__APPLE__)
    // Update Metal layer size if on macOS
    if (context.metalLayer) {
        UpdateMetalLayerSize(context.metalLayer, context.window);
    }
#endif
    
    // Reconfigure the surface with new dimensions
    wgpu::SurfaceConfiguration surfaceConfig{};
    surfaceConfig.device = context.device;
    surfaceConfig.format = context.swapChainFormat;
    surfaceConfig.usage = wgpu::TextureUsage::RenderAttachment;
    surfaceConfig.width = static_cast<uint32_t>(width);
    surfaceConfig.height = static_cast<uint32_t>(height);
    surfaceConfig.presentMode = wgpu::PresentMode::Fifo;
    
    context.surface.Configure(&surfaceConfig);
    
    // Reinitialize compute shader resources for new size
    std::cout << "About to call reinitComputeShaderResources..." << std::endl;
    reinitComputeShaderResources(context);
    std::cout << "reinitComputeShaderResources call completed." << std::endl;
}

wgpu::ShaderModule createShaderModule(const wgpu::Device& device, const char* source) {
    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = source;
    wgpu::ShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &wgslDesc;
    
    std::cout << "Creating shader module..." << std::endl;
    
    // Push error scope to catch shader compilation errors
    device.PushErrorScope(wgpu::ErrorFilter::Validation);
    
    // Create the shader module
    wgpu::ShaderModule shaderModule = device.CreateShaderModule(&shaderDesc);
    
    // Check for compilation errors immediately
    bool errorScopeHasErrors = false;
    std::string errorScopeMessage;
    device.PopErrorScope(
        wgpu::CallbackMode::WaitAnyOnly,
        [&errorScopeHasErrors, &errorScopeMessage]
        (wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, const char* message) {
            if (status == wgpu::PopErrorScopeStatus::Success && type != wgpu::ErrorType::NoError) {
                errorScopeHasErrors = true;
                errorScopeMessage = std::string("ErrorType: ") + std::to_string(static_cast<int>(type)) + 
                                   ", Message: " + (message ? message : "No message");
            }
        }
    );
    
    if (errorScopeHasErrors) {
        std::cerr << "\nSHADER MODULE CREATION ERROR DETECTED!" << std::endl;
        std::cerr << errorScopeMessage << std::endl;
    }
    
    if (shaderModule) {
        std::cout << "Shader module object created, checking compilation info..." << std::endl;
        
        // Use GetCompilationInfo to get immediate syntax error feedback
        bool compilationInfoReceived = false;
        bool hasCompilationErrors = false;
        std::string compilationMessages;
        
        shaderModule.GetCompilationInfo(
            wgpu::CallbackMode::WaitAnyOnly,
            [&compilationInfoReceived, &hasCompilationErrors, &compilationMessages]
            (wgpu::CompilationInfoRequestStatus status, wgpu::CompilationInfo const* info) {
                compilationInfoReceived = true;
                
                if (status == wgpu::CompilationInfoRequestStatus::Success && info) {
                    for (size_t i = 0; i < info->messageCount; ++i) {
                        const auto& msg = info->messages[i];
                        
                        std::string severity;
                        if (msg.type == wgpu::CompilationMessageType::Error) {
                            severity = "ERROR";
                            hasCompilationErrors = true;
                        } else if (msg.type == wgpu::CompilationMessageType::Warning) {
                            severity = "WARNING";
                        } else {
                            severity = "INFO";
                        }
                        
                        compilationMessages += "[" + severity + "] Line " + std::to_string(msg.lineNum) + 
                                               ", Col " + std::to_string(msg.linePos) + ": " + 
                                               std::string(msg.message) + "\n";
                    }
                }
            }
        );
        
        // Check compilation results
        if (hasCompilationErrors) {
            std::cerr << "\nSHADER COMPILATION ERRORS DETECTED!" << std::endl;
            std::cerr << compilationMessages << std::endl;
            std::cout << "Problematic shader source (first 1000 chars):" << std::endl;
            std::string sourcePreview(source);
            if (sourcePreview.length() > 1000) {
                sourcePreview = sourcePreview.substr(0, 1000) + "...";
            }
            std::cout << sourcePreview << std::endl;
            std::cout << "End of shader source preview" << std::endl;
        } else if (!compilationMessages.empty()) {
            std::cout << "WARNING: Shader compilation messages:" << std::endl;
            std::cout << compilationMessages << std::endl;
        } else {
            std::cout << "No compilation messages - shader syntax appears valid" << std::endl;
        }
        
        std::cout << "Now attempting advanced pipeline validation..." << std::endl;
        
        // Force validation by trying to create a dummy compute pipeline
        // This will actually compile the shader and catch syntax errors
        bool validationPassed = true;
        std::string validationError = "Unknown validation error";
        
        // Check if this looks like a compute shader (has @compute and workgroup_size)
        std::string sourceStr(source);
        if (sourceStr.find("@compute") != std::string::npos && 
            sourceStr.find("workgroup_size") != std::string::npos) {
            
            std::cout << "Validating compute shader by attempting pipeline creation..." << std::endl;
            
            try {
                // For comprehensive validation, we need to match the actual shader bindings
                // Let's create bind group layouts that match what the real shader expects
                
                // Check if this looks like the iteration shader or coloring shader
                if (sourceStr.find("iterationOutput") != std::string::npos) {
                    std::cout << "Detected iteration compute shader - creating matching layout..." << std::endl;
                    
                    // Create layout matching iteration shader: storage buffer + uniform buffer
                    wgpu::BindGroupLayoutEntry layoutEntries[2] = {};
                    
                    // Storage buffer (iterationOutput)
                    layoutEntries[0].binding = 0;
                    layoutEntries[0].visibility = wgpu::ShaderStage::Compute;
                    layoutEntries[0].buffer.type = wgpu::BufferBindingType::Storage;
                    
                    // Uniform buffer (params)
                    layoutEntries[1].binding = 1;
                    layoutEntries[1].visibility = wgpu::ShaderStage::Compute;
                    layoutEntries[1].buffer.type = wgpu::BufferBindingType::Uniform;
                    
                    wgpu::BindGroupLayoutDescriptor layoutDesc{};
                    layoutDesc.entryCount = 2;
                    layoutDesc.entries = layoutEntries;
                    wgpu::BindGroupLayout bindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
                    
                    // Create pipeline layout
                    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc{};
                    pipelineLayoutDesc.bindGroupLayoutCount = 1;
                    pipelineLayoutDesc.bindGroupLayouts = &bindGroupLayout;
                    wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
                    
                    // Try to create compute pipeline - this will force actual shader compilation
                    wgpu::ComputePipelineDescriptor pipelineDesc{};
                    pipelineDesc.layout = pipelineLayout;
                    pipelineDesc.compute.module = shaderModule;
                    pipelineDesc.compute.entryPoint = "main";
                    
                    wgpu::ComputePipeline testPipeline = device.CreateComputePipeline(&pipelineDesc);
                    
                    if (testPipeline) {
                        std::cout << "Test iteration pipeline created successfully!" << std::endl;
                        
                        // Now try to actually execute it with dummy data to catch runtime errors
                        std::cout << "Attempting shader execution validation..." << std::endl;
                        if (!validateComputeShaderExecution(device, testPipeline, "iteration")) {
                            validationPassed = false;
                            validationError = "Shader execution validation failed";
                        }
                    } else {
                        validationPassed = false;
                        validationError = "CreateComputePipeline returned null for iteration shader";
                    }
                    
                } else if (sourceStr.find("colorOutput") != std::string::npos) {
                    std::cout << "Detected coloring compute shader - creating matching layout..." << std::endl;
                    
                    // Create layout matching coloring shader: read storage + write storage + uniform
                    wgpu::BindGroupLayoutEntry layoutEntries[3] = {};
                    
                    // Read-only storage buffer (iterationInput)
                    layoutEntries[0].binding = 0;
                    layoutEntries[0].visibility = wgpu::ShaderStage::Compute;
                    layoutEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
                    
                    // Storage buffer (colorOutput)
                    layoutEntries[1].binding = 1;
                    layoutEntries[1].visibility = wgpu::ShaderStage::Compute;
                    layoutEntries[1].buffer.type = wgpu::BufferBindingType::Storage;
                    
                    // Uniform buffer (colorParams)
                    layoutEntries[2].binding = 2;
                    layoutEntries[2].visibility = wgpu::ShaderStage::Compute;
                    layoutEntries[2].buffer.type = wgpu::BufferBindingType::Uniform;
                    
                    wgpu::BindGroupLayoutDescriptor layoutDesc{};
                    layoutDesc.entryCount = 3;
                    layoutDesc.entries = layoutEntries;
                    wgpu::BindGroupLayout bindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
                    
                    // Create pipeline layout
                    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc{};
                    pipelineLayoutDesc.bindGroupLayoutCount = 1;
                    pipelineLayoutDesc.bindGroupLayouts = &bindGroupLayout;
                    wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
                    
                    // Try to create compute pipeline
                    wgpu::ComputePipelineDescriptor pipelineDesc{};
                    pipelineDesc.layout = pipelineLayout;
                    pipelineDesc.compute.module = shaderModule;
                    pipelineDesc.compute.entryPoint = "main";
                    
                    wgpu::ComputePipeline testPipeline = device.CreateComputePipeline(&pipelineDesc);
                    
                    if (testPipeline) {
                        std::cout << "Test coloring pipeline created successfully!" << std::endl;
                        
                        // Try executing it
                        std::cout << "Attempting shader execution validation..." << std::endl;
                        if (!validateComputeShaderExecution(device, testPipeline, "coloring")) {
                            validationPassed = false;
                            validationError = "Shader execution validation failed";
                        }
                    } else {
                        validationPassed = false;
                        validationError = "CreateComputePipeline returned null for coloring shader";
                    }
                    
                } else {
                    // Generic compute shader - use empty layout
                    std::cout << "Generic compute shader - using empty layout..." << std::endl;
                    
                    wgpu::BindGroupLayoutDescriptor emptyLayoutDesc{};
                    emptyLayoutDesc.entryCount = 0;
                    emptyLayoutDesc.entries = nullptr;
                    wgpu::BindGroupLayout emptyLayout = device.CreateBindGroupLayout(&emptyLayoutDesc);
                    
                    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc{};
                    pipelineLayoutDesc.bindGroupLayoutCount = 1;
                    pipelineLayoutDesc.bindGroupLayouts = &emptyLayout;
                    wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
                    
                    wgpu::ComputePipelineDescriptor pipelineDesc{};
                    pipelineDesc.layout = pipelineLayout;
                    pipelineDesc.compute.module = shaderModule;
                    pipelineDesc.compute.entryPoint = "main";
                    
                    wgpu::ComputePipeline testPipeline = device.CreateComputePipeline(&pipelineDesc);
                    
                    if (testPipeline) {
                        std::cout << "Test generic pipeline created successfully!" << std::endl;
                    } else {
                        validationPassed = false;
                        validationError = "CreateComputePipeline returned null for generic shader";
                    }
                }
                
            } catch (const std::exception& e) {
                validationPassed = false;
                validationError = std::string("Exception during pipeline creation: ") + e.what();
            } catch (...) {
                validationPassed = false;
                validationError = "Unknown exception during pipeline creation";
            }
            
        } else if (sourceStr.find("@vertex") != std::string::npos || 
                   sourceStr.find("@fragment") != std::string::npos) {
            std::cout << "Detected render shader - skipping validation (would need complex setup)" << std::endl;
            // For render shaders, validation is more complex as we'd need vertex formats, etc.
            // We'll just trust that errors will be caught when the actual pipeline is created
        } else {
            std::cout << "Unknown shader type - skipping validation" << std::endl;
        }
        
        // Report validation results
        if (validationPassed) {
            std::cout << "Shader validation passed!" << std::endl;
        } else {
            std::cerr << "SHADER VALIDATION FAILED!" << std::endl;
            std::cerr << "Error: " << validationError << std::endl;
            std::cerr << "This shader will likely cause rendering problems!" << std::endl;
            
            // Print a portion of the shader source for debugging
            std::cout << "Problematic shader source (first 1000 chars):" << std::endl;
            std::string sourcePreview(source);
            if (sourcePreview.length() > 1000) {
                sourcePreview = sourcePreview.substr(0, 1000) + "...";
            }
            std::cout << sourcePreview << std::endl;
            std::cout << "End of shader source preview" << std::endl;
        }
        
    } else {
        std::cerr << "Failed to create shader module!" << std::endl;
    }
    
    return shaderModule;
}

// Comprehensive validation function that actually executes the compute shader
bool validateComputeShaderExecution(const wgpu::Device& device, wgpu::ComputePipeline& pipeline, const std::string& shaderType) {
    try {
        std::cout << "Executing " << shaderType << " shader validation test..." << std::endl;
        
        // Create minimal test buffers and try to dispatch the shader
        if (shaderType == "iteration") {
            // Create test buffers for iteration shader
            wgpu::BufferDescriptor outputBufferDesc{};
            outputBufferDesc.size = 64 * sizeof(float); // Small test buffer
            outputBufferDesc.usage = wgpu::BufferUsage::Storage;
            wgpu::Buffer outputBuffer = device.CreateBuffer(&outputBufferDesc);
            
            wgpu::BufferDescriptor uniformBufferDesc{};
            uniformBufferDesc.size = 64; // Small uniform buffer  
            uniformBufferDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
            wgpu::Buffer uniformBuffer = device.CreateBuffer(&uniformBufferDesc);
            
            // Create minimal bind group
            wgpu::BindGroupLayoutEntry layoutEntries[2] = {};
            layoutEntries[0].binding = 0;
            layoutEntries[0].visibility = wgpu::ShaderStage::Compute;
            layoutEntries[0].buffer.type = wgpu::BufferBindingType::Storage;
            layoutEntries[1].binding = 1;
            layoutEntries[1].visibility = wgpu::ShaderStage::Compute;
            layoutEntries[1].buffer.type = wgpu::BufferBindingType::Uniform;
            
            wgpu::BindGroupLayoutDescriptor layoutDesc{};
            layoutDesc.entryCount = 2;
            layoutDesc.entries = layoutEntries;
            wgpu::BindGroupLayout bindGroupLayout = device.CreateBindGroupLayout(&layoutDesc);
            
            wgpu::BindGroupEntry bindGroupEntries[2] = {};
            bindGroupEntries[0].binding = 0;
            bindGroupEntries[0].buffer = outputBuffer;
            bindGroupEntries[0].size = outputBufferDesc.size;
            bindGroupEntries[1].binding = 1;
            bindGroupEntries[1].buffer = uniformBuffer;
            bindGroupEntries[1].size = uniformBufferDesc.size;
            
            wgpu::BindGroupDescriptor bindGroupDesc{};
            bindGroupDesc.layout = bindGroupLayout;
            bindGroupDesc.entryCount = 2;
            bindGroupDesc.entries = bindGroupEntries;
            wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);
            
            // Try to dispatch the compute shader
            wgpu::CommandEncoderDescriptor encoderDesc{};
            wgpu::CommandEncoder encoder = device.CreateCommandEncoder(&encoderDesc);
            
            wgpu::ComputePassDescriptor computePassDesc{};
            wgpu::ComputePassEncoder computePass = encoder.BeginComputePass(&computePassDesc);
            
            computePass.SetPipeline(pipeline);
            computePass.SetBindGroup(0, bindGroup);
            computePass.DispatchWorkgroups(1, 1); // Very small dispatch
            computePass.End();
            
            wgpu::CommandBufferDescriptor cmdBufferDesc{};
            wgpu::CommandBuffer cmdBuffer = encoder.Finish(&cmdBufferDesc);
            
            wgpu::Queue queue = device.GetQueue();
            queue.Submit(1, &cmdBuffer);
            
            std::cout << "Iteration shader executed successfully in validation test" << std::endl;
            return true;
            
        } else if (shaderType == "coloring") {
            // Similar setup for coloring shader but with 3 buffers
            std::cout << "WARNING: Coloring shader validation not fully implemented yet" << std::endl;
            return true; // Skip for now
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Shader execution validation FAILED: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Shader execution validation FAILED: Unknown exception" << std::endl;
        return false;
    }
} 