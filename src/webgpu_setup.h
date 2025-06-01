#pragma once
#include <webgpu/webgpu_cpp.h>
#include <GLFW/glfw3.h>
#include <functional>
#include <chrono>

// Debug and validation callbacks
using DeviceErrorCallback = std::function<void(wgpu::ErrorType, const char*)>;
using UncapturedErrorCallback = std::function<void(wgpu::ErrorType, const char*)>;
using DeviceLostCallback = std::function<void(wgpu::DeviceLostReason, const char*)>;

struct FrameStats {
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
    uint32_t frameNumber = 0;
    bool renderingComplete = false;
    bool computeComplete = false;
    uint32_t errorCount = 0;
};

struct WebGPUContext {
    GLFWwindow* window;
    wgpu::Instance instance;
    wgpu::Surface surface;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
    wgpu::TextureFormat swapChainFormat;
    int width;
    int height;
    void* metalLayer; // For macOS Metal layer cleanup
    
    // Debug and validation state
    bool validationEnabled = false;
    bool debugEnabled = false;
    DeviceErrorCallback deviceErrorCallback;
    UncapturedErrorCallback uncapturedErrorCallback;
    DeviceLostCallback deviceLostCallback;
    
    // Frame timing and stats
    FrameStats currentFrame;
    uint32_t totalFrames = 0;
    uint32_t totalErrors = 0;
};

// Initialize WebGPU and GLFW with enhanced error handling
WebGPUContext initWebGPU();

// Clean up resources
void cleanupWebGPU(WebGPUContext& context);

// Reconfigure surface for new size
void reconfigureSurface(WebGPUContext& context, int width, int height);

// Create a shader module from WGSL source with validation
wgpu::ShaderModule createShaderModule(const wgpu::Device& device, const char* source);

// Enhanced debugging functions
void enableWebGPUValidation(WebGPUContext& context);
void setupErrorCallbacks(WebGPUContext& context);
void beginFrameDebugging(WebGPUContext& context);
void endFrameDebugging(WebGPUContext& context);
void validateWebGPUState(const WebGPUContext& context);

// Performance monitoring
void printFrameStats(const WebGPUContext& context);
bool checkRenderingHealth(const WebGPUContext& context); 