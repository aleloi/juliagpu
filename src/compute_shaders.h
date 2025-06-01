#pragma once
#include <webgpu/webgpu_cpp.h>
#include "webgpu_setup.h"
#include "fractal_settings.h"

// Initialize compute shader resources
void initComputeShaders(const WebGPUContext& context);

// Clean up compute shader resources
void cleanupComputeShaders();

// Reinitialize compute shader resources for new dimensions
void reinitComputeShaderResources(const WebGPUContext& context);

// Generate fractal data using compute shader
void generateFractalData(const WebGPUContext& context, wgpu::CommandEncoder& encoder, 
                        wgpu::Buffer& outputBuffer,
                        double viewX, double viewY, double viewWidth, double viewHeight);

// Create compute pipelines for different fractal types and visualization modes
void createFractalComputePipelines(const WebGPUContext& context);

// Get vertex and fragment shader sources for rendering
const char* getVertexShaderSource();
const char* getFragmentShaderSource();

// Get render bind group layout and bind group for fractal texture rendering
wgpu::BindGroupLayout getRenderBindGroupLayout();
wgpu::BindGroup getRenderBindGroup();

// Get internal output buffer for debugging
wgpu::Buffer& getOutputBuffer();

// === DEBUG FUNCTIONS ===
// Debug function to dump buffer data to a raw RGBA file
void debugDumpBufferToFile(const WebGPUContext& context, wgpu::Buffer& buffer, 
                          uint32_t width, uint32_t height, const char* filename);

// Debug function to dump texture to a raw RGBA file  
void debugDumpTextureToFile(const WebGPUContext& context, wgpu::Texture& texture,
                           uint32_t width, uint32_t height, const char* filename);

// Debug function to check if compute shaders are working by reading buffer contents
void debugVerifyComputeShaderExecution(const WebGPUContext& context);

// Debug function to save an image in PPM format (simple text-based format)
void debugSaveBufferAsPPM(const WebGPUContext& context, wgpu::Buffer& buffer,
                          uint32_t width, uint32_t height, const char* filename);

// Enhanced debugging functions for compute shaders
void validateComputeShaderSetup(const WebGPUContext& context);
void beginComputeDebugging(WebGPUContext& context);
void endComputeDebugging(WebGPUContext& context);
void debugComputeShaderExecution(const WebGPUContext& context); 