#include "compute_shaders.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <vector>

// Global fractal settings instance (extern)
extern FractalSettings g_settings;

// Step 1: Julia iteration compute shader (outputs normalized iteration counts)
const char* juliaIterationComputeShaderCode = R"(
  @group(0) @binding(0) var<storage, read_write> iterationOutput: array<f32>;
  @group(0) @binding(1) var<uniform> params: FractalParams;

  struct FractalParams {
    viewX: f32,
    viewY: f32,
    viewWidth: f32,
    viewHeight: f32,
    width: u32,
    height: u32,
    maxIterations: u32,
    juliaCX: f32,
    juliaCY: f32,
    coloringMode: u32, // 0 = COLOR_BANDS, 1 = COLOR_CONTINUOUS, 2 = GRAYSCALE, 3 = THRESHOLD
  }

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
      return;
    }
    
    let pixel_index = y * params.width + x;
    
    // Map screen coordinates to complex plane
    let real = params.viewX + (f32(x) / f32(params.width)) * params.viewWidth;
    let imag = params.viewY + (f32(y) / f32(params.height)) * params.viewHeight;
    
    // Julia set computation
    var zr: f32 = real;
    var zi: f32 = imag;
    let cr: f32 = params.juliaCX;
    let ci: f32 = params.juliaCY;
    
    // Iterate to determine if point is in set
    var iter: u32 = 0u;
    var zr2: f32 = zr * zr;
    var zi2: f32 = zi * zi;
    
    while (iter < params.maxIterations && zr2 + zi2 < 4.0) {
      zi = 2.0 * zr * zi + ci;
      zr = zr2 - zi2 + cr;
      zr2 = zr * zr;
      zi2 = zi * zi;
      iter = iter + 1u;
    }
    
    // Calculate smooth iteration count for better color gradients
    var smooth_iter: f32;
    if (iter < params.maxIterations && params.coloringMode == 1u) {
      // Point escaped and COLOR_CONTINUOUS mode - use smooth iteration formula
      // Standard formula: smooth_iter = iter + 1 - log2(log2(|z|))
      // where |z| = sqrt(zr2 + zi2)
      let log_zn = log(zr2 + zi2) / 2.0;  // This is log(|z|)
      let nu = log(log_zn / log(2.0)) / log(2.0);  // This is log2(log2(|z|))
      smooth_iter = f32(iter) + 1.0 - nu;
    } else {
      // Point is in the set or using discrete coloring - use integer iteration count
      smooth_iter = f32(iter);
    }
    
    // Normalize smooth iteration count to [0, 1] range
    let normalized_iter = smooth_iter / f32(params.maxIterations);
    
    // Store normalized iteration count
    iterationOutput[pixel_index] = normalized_iter;
  }
)";

// Step 2: Coloring compute shader (converts iteration counts to colors)
const char* coloringComputeShaderCode = R"(
  @group(0) @binding(0) var<storage, read> iterationInput: array<f32>;
  @group(0) @binding(1) var<storage, read_write> colorOutput: array<u32>;
  @group(0) @binding(2) var<uniform> colorParams: ColoringParams;

  struct ColoringParams {
    width: u32,
    height: u32,
    coloringMode: u32, // 0 = COLOR_BANDS, 1 = COLOR_CONTINUOUS, 2 = GRAYSCALE, 3 = THRESHOLD
    maxIterations: u32,
  }

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= colorParams.width || y >= colorParams.height) {
      return;
    }
    
    let pixel_index = y * colorParams.width + x;
    let normalized_iter = iterationInput[pixel_index];
    
    var r: u32;
    var g: u32;
    var b: u32;
    
    // Color gradient mapping based on normalized iteration count
    if (normalized_iter >= 1.0) {
      // Point is in the set - black for COLOR_BANDS/COLOR_CONTINUOUS/GRAYSCALE, white for THRESHOLD
      if (colorParams.coloringMode == 3u) {
        // THRESHOLD mode: non-escaped points (in set) = WHITE
        r = 255u;
        g = 255u;
        b = 255u;
      } else {
        // COLOR_BANDS/COLOR_CONTINUOUS/GRAYSCALE mode: non-escaped points = BLACK
        r = 0u;
        g = 0u;
        b = 0u;
      }
    } else {
      // Point escaped - apply coloring based on mode
      if (colorParams.coloringMode == 3u) {
        // THRESHOLD mode: escaped points = BLACK
        r = 0u;
        g = 0u;
        b = 0u;
      } else if (colorParams.coloringMode == 2u) {
        // GRAYSCALE mode: RGB = 255 * normalized_iter for all channels
        let gray_value = u32(255.0 * normalized_iter);
        r = gray_value;
        g = gray_value;
        b = gray_value;
      } else {
        // COLOR_BANDS (0) or COLOR_CONTINUOUS (1) mode: apply color gradient
        let t = normalized_iter;
        
        if (t < 0.16) {
          // Black to Blue transition
          let s = t / 0.16;
          r = 0u;
          g = 0u;
          b = u32(255.0 * s);
        } else if (t < 0.42) {
          // Blue to Orange transition  
          let s = (t - 0.16) / (0.42 - 0.16);
          r = u32(255.0 * s);
          g = u32(165.0 * s);
          b = u32(255.0 * (1.0 - s));
        } else if (t < 0.6425) {
          // Orange to Yellow transition
          let s = (t - 0.42) / (0.6425 - 0.42);
          r = 255u;
          g = u32(165.0 * (1.0 - s));
          b = 0u;
        } else {
          // Yellow to White transition
          let s = (t - 0.6425) / (1.0 - 0.6425);
          r = 255u;
          g = 255u;
          b = u32(255.0 * s);
        }
      }
    }
    
    let a: u32 = 255u;
    
    // Store color as RGBA
    colorOutput[pixel_index] = (a << 24u) | (b << 16u) | (g << 8u) | r;
  }
)";

// Vertex shader for full-screen quad
const char* vertexShaderCode = R"(
  @vertex
  fn main(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
      vec2<f32>(-1.0, -1.0),
      vec2<f32>(1.0, -1.0),
      vec2<f32>(-1.0, 1.0),
      vec2<f32>(-1.0, 1.0),
      vec2<f32>(1.0, -1.0),
      vec2<f32>(1.0, 1.0)
    );
    return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
  }
)";

// Fragment shader for displaying fractal texture
const char* fragmentShaderCode = R"(
  @group(0) @binding(0) var fractalTexture: texture_2d<f32>;
  @group(0) @binding(1) var fractalSampler: sampler;

  @fragment
  fn main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let texSize = textureDimensions(fractalTexture);
    // Convert screen coordinates to normalized texture coordinates [0,1]
    // Note: fragCoord is in screen space, need to normalize to [0,1]
    let texCoord = vec2<f32>(fragCoord.x / f32(texSize.x), 1.0 - fragCoord.y / f32(texSize.y));
    
    // Sample the texture
    let sampledColor = textureSample(fractalTexture, fractalSampler, texCoord);
    
    // Debug: Return a test pattern if texture is empty
    if (sampledColor.r == 0.0 && sampledColor.g == 0.0 && sampledColor.b == 0.0 && sampledColor.a == 0.0) {
      // Create a simple test pattern
      let x = u32(fragCoord.x);
      let y = u32(fragCoord.y);
      if (x < 100u && y < 100u) {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0); // Red square
      } else if (x >= texSize.x - 100u && y < 100u) {
        return vec4<f32>(0.0, 1.0, 0.0, 1.0); // Green square
      } else {
        return vec4<f32>(1.0, 0.0, 1.0, 1.0); // Magenta for debugging
      }
    }
    
    return sampledColor;
  }
)";

// Struct to match WGSL FractalParams
struct FractalParams {
    float viewX;
    float viewY;
    float viewWidth;
    float viewHeight;
    uint32_t width;
    uint32_t height;
    uint32_t maxIterations;
    float juliaCX;
    float juliaCY;
    uint32_t coloringMode; // 0 = COLOR_BANDS, 1 = COLOR_CONTINUOUS, 2 = GRAYSCALE, 3 = THRESHOLD
};

// Struct to match WGSL ColoringParams
struct ColoringParams {
    uint32_t width;
    uint32_t height;
    uint32_t coloringMode; // 0 = COLOR_BANDS, 1 = COLOR_CONTINUOUS, 2 = GRAYSCALE, 3 = THRESHOLD
    uint32_t maxIterations;
};

// Static compute shader resources
static wgpu::ComputePipeline iterationPipeline;
static wgpu::ComputePipeline coloringPipeline;
static wgpu::BindGroupLayout iterationBindGroupLayout;
static wgpu::BindGroupLayout coloringBindGroupLayout;
static wgpu::BindGroup iterationBindGroup;
static wgpu::BindGroup coloringBindGroup;
static wgpu::Buffer uniformBuffer;
static wgpu::Buffer coloringUniformBuffer;
static wgpu::Buffer iterationBuffer;
static wgpu::Buffer outputBuffer;
static wgpu::Texture fractalTexture;
static wgpu::TextureView fractalTextureView;
static wgpu::Sampler fractalSampler;
static wgpu::BindGroupLayout renderBindGroupLayout;
static wgpu::BindGroup renderBindGroup;
static bool initialized = false;
static WebGPUContext g_context;

// Resource recreation state - deferred to avoid race conditions
static bool needsResourceRecreation = false;
static uint32_t pendingWidth = 0;
static uint32_t pendingHeight = 0;

void initComputeShaders(const WebGPUContext& context) {
    if (initialized) return;
    
    std::cout << "Initializing two-step compute shaders with enhanced debugging..." << std::endl;
    g_context = context;
    
    // Create compute shader modules
    wgpu::ShaderModule iterationShaderModule = createShaderModule(context.device, juliaIterationComputeShaderCode);
    wgpu::ShaderModule coloringShaderModule = createShaderModule(context.device, coloringComputeShaderCode);
    
    // === STEP 1: ITERATION COMPUTATION ===
    
    // Create bind group layout for iteration shader
    wgpu::BindGroupLayoutEntry iterationLayoutEntries[2] = {};
    
    // Iteration output buffer binding
    iterationLayoutEntries[0].binding = 0;
    iterationLayoutEntries[0].visibility = wgpu::ShaderStage::Compute;
    iterationLayoutEntries[0].buffer.type = wgpu::BufferBindingType::Storage;
    
    // Uniform buffer binding (params)
    iterationLayoutEntries[1].binding = 1;
    iterationLayoutEntries[1].visibility = wgpu::ShaderStage::Compute;
    iterationLayoutEntries[1].buffer.type = wgpu::BufferBindingType::Uniform;
    
    wgpu::BindGroupLayoutDescriptor iterationLayoutDesc{};
    iterationLayoutDesc.entryCount = 2;
    iterationLayoutDesc.entries = iterationLayoutEntries;
    iterationBindGroupLayout = context.device.CreateBindGroupLayout(&iterationLayoutDesc);
    
    // Create iteration pipeline layout
    wgpu::PipelineLayoutDescriptor iterationPipelineLayoutDesc{};
    iterationPipelineLayoutDesc.bindGroupLayoutCount = 1;
    iterationPipelineLayoutDesc.bindGroupLayouts = &iterationBindGroupLayout;
    wgpu::PipelineLayout iterationPipelineLayout = context.device.CreatePipelineLayout(&iterationPipelineLayoutDesc);
    
    // Create iteration compute pipeline
    wgpu::ComputePipelineDescriptor iterationPipelineDesc{};
    iterationPipelineDesc.layout = iterationPipelineLayout;
    iterationPipelineDesc.compute.module = iterationShaderModule;
    iterationPipelineDesc.compute.entryPoint = "main";
    iterationPipeline = context.device.CreateComputePipeline(&iterationPipelineDesc);
    
    // === STEP 2: COLORING COMPUTATION ===
    
    // Create bind group layout for coloring shader
    wgpu::BindGroupLayoutEntry coloringLayoutEntries[3] = {};
    
    // Iteration input buffer binding
    coloringLayoutEntries[0].binding = 0;
    coloringLayoutEntries[0].visibility = wgpu::ShaderStage::Compute;
    coloringLayoutEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    
    // Color output buffer binding
    coloringLayoutEntries[1].binding = 1;
    coloringLayoutEntries[1].visibility = wgpu::ShaderStage::Compute;
    coloringLayoutEntries[1].buffer.type = wgpu::BufferBindingType::Storage;
    
    // Coloring uniform buffer binding
    coloringLayoutEntries[2].binding = 2;
    coloringLayoutEntries[2].visibility = wgpu::ShaderStage::Compute;
    coloringLayoutEntries[2].buffer.type = wgpu::BufferBindingType::Uniform;
    
    wgpu::BindGroupLayoutDescriptor coloringLayoutDesc{};
    coloringLayoutDesc.entryCount = 3;
    coloringLayoutDesc.entries = coloringLayoutEntries;
    coloringBindGroupLayout = context.device.CreateBindGroupLayout(&coloringLayoutDesc);
    
    // Create coloring pipeline layout
    wgpu::PipelineLayoutDescriptor coloringPipelineLayoutDesc{};
    coloringPipelineLayoutDesc.bindGroupLayoutCount = 1;
    coloringPipelineLayoutDesc.bindGroupLayouts = &coloringBindGroupLayout;
    wgpu::PipelineLayout coloringPipelineLayout = context.device.CreatePipelineLayout(&coloringPipelineLayoutDesc);
    
    // Create coloring compute pipeline
    wgpu::ComputePipelineDescriptor coloringPipelineDesc{};
    coloringPipelineDesc.layout = coloringPipelineLayout;
    coloringPipelineDesc.compute.module = coloringShaderModule;
    coloringPipelineDesc.compute.entryPoint = "main";
    coloringPipeline = context.device.CreateComputePipeline(&coloringPipelineDesc);
    
    // === BUFFER CREATION ===
    
    // Create uniform buffer
    wgpu::BufferDescriptor uniformBufferDesc{};
    uniformBufferDesc.size = sizeof(FractalParams);
    uniformBufferDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniformBuffer = context.device.CreateBuffer(&uniformBufferDesc);
    
    // Create coloring uniform buffer
    wgpu::BufferDescriptor coloringUniformBufferDesc{};
    coloringUniformBufferDesc.size = sizeof(ColoringParams);
    coloringUniformBufferDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    coloringUniformBuffer = context.device.CreateBuffer(&coloringUniformBufferDesc);
    
    // Create iteration buffer (stores normalized iteration counts as floats)
    wgpu::BufferDescriptor iterationBufferDesc{};
    iterationBufferDesc.size = context.width * context.height * sizeof(float);
    iterationBufferDesc.usage = wgpu::BufferUsage::Storage;
    iterationBuffer = context.device.CreateBuffer(&iterationBufferDesc);
    
    // Create output buffer (stores final RGBA colors)
    wgpu::BufferDescriptor outputBufferDesc{};
    outputBufferDesc.size = context.width * context.height * 4; // RGBA bytes
    outputBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    outputBuffer = context.device.CreateBuffer(&outputBufferDesc);
    
    // === BIND GROUPS ===
    
    // Create bind group for iteration shader
    wgpu::BindGroupEntry iterationBindGroupEntries[2] = {};
    iterationBindGroupEntries[0].binding = 0;
    iterationBindGroupEntries[0].buffer = iterationBuffer;
    iterationBindGroupEntries[0].size = iterationBufferDesc.size;
    
    iterationBindGroupEntries[1].binding = 1;
    iterationBindGroupEntries[1].buffer = uniformBuffer;
    iterationBindGroupEntries[1].size = sizeof(FractalParams);
    
    wgpu::BindGroupDescriptor iterationBindGroupDesc{};
    iterationBindGroupDesc.layout = iterationBindGroupLayout;
    iterationBindGroupDesc.entryCount = 2;
    iterationBindGroupDesc.entries = iterationBindGroupEntries;
    iterationBindGroup = context.device.CreateBindGroup(&iterationBindGroupDesc);
    
    // Create bind group for coloring shader
    wgpu::BindGroupEntry coloringBindGroupEntries[3] = {};
    coloringBindGroupEntries[0].binding = 0;
    coloringBindGroupEntries[0].buffer = iterationBuffer;
    coloringBindGroupEntries[0].size = iterationBufferDesc.size;
    
    coloringBindGroupEntries[1].binding = 1;
    coloringBindGroupEntries[1].buffer = outputBuffer;
    coloringBindGroupEntries[1].size = outputBufferDesc.size;
    
    coloringBindGroupEntries[2].binding = 2;
    coloringBindGroupEntries[2].buffer = coloringUniformBuffer;
    coloringBindGroupEntries[2].size = sizeof(ColoringParams);
    
    wgpu::BindGroupDescriptor coloringBindGroupDesc{};
    coloringBindGroupDesc.layout = coloringBindGroupLayout;
    coloringBindGroupDesc.entryCount = 3;
    coloringBindGroupDesc.entries = coloringBindGroupEntries;
    coloringBindGroup = context.device.CreateBindGroup(&coloringBindGroupDesc);
    
    // Create fractal texture
    wgpu::TextureDescriptor textureDesc{};
    textureDesc.size = {static_cast<uint32_t>(context.width), static_cast<uint32_t>(context.height), 1};
    textureDesc.format = wgpu::TextureFormat::RGBA8Unorm;
    textureDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    fractalTexture = context.device.CreateTexture(&textureDesc);
    fractalTextureView = fractalTexture.CreateView();
    
    // Create sampler
    wgpu::SamplerDescriptor samplerDesc{};
    samplerDesc.magFilter = wgpu::FilterMode::Nearest;
    samplerDesc.minFilter = wgpu::FilterMode::Nearest;
    fractalSampler = context.device.CreateSampler(&samplerDesc);
    
    // Create bind group layout for render pipeline
    wgpu::BindGroupLayoutEntry renderLayoutEntries[2] = {};
    
    // Texture binding
    renderLayoutEntries[0].binding = 0;
    renderLayoutEntries[0].visibility = wgpu::ShaderStage::Fragment;
    renderLayoutEntries[0].texture.sampleType = wgpu::TextureSampleType::Float;
    renderLayoutEntries[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    
    // Sampler binding
    renderLayoutEntries[1].binding = 1;
    renderLayoutEntries[1].visibility = wgpu::ShaderStage::Fragment;
    renderLayoutEntries[1].sampler.type = wgpu::SamplerBindingType::Filtering;
    
    wgpu::BindGroupLayoutDescriptor renderLayoutDesc{};
    renderLayoutDesc.entryCount = 2;
    renderLayoutDesc.entries = renderLayoutEntries;
    renderBindGroupLayout = context.device.CreateBindGroupLayout(&renderLayoutDesc);
    
    // Create bind group for render pipeline
    wgpu::BindGroupEntry renderBindGroupEntries[2] = {};
    renderBindGroupEntries[0].binding = 0;
    renderBindGroupEntries[0].textureView = fractalTextureView;
    
    renderBindGroupEntries[1].binding = 1;
    renderBindGroupEntries[1].sampler = fractalSampler;
    
    wgpu::BindGroupDescriptor renderBindGroupDesc{};
    renderBindGroupDesc.layout = renderBindGroupLayout;
    renderBindGroupDesc.entryCount = 2;
    renderBindGroupDesc.entries = renderBindGroupEntries;
    renderBindGroup = context.device.CreateBindGroup(&renderBindGroupDesc);
    
    initialized = true;
    std::cout << "Two-step compute shaders initialized successfully!" << std::endl;
    
    // Validate the setup
    validateComputeShaderSetup(context);
}

void cleanupComputeShaders() {
    initialized = false;
    // WebGPU resources are automatically cleaned up when they go out of scope
}

void reinitComputeShaderResources(const WebGPUContext& context) {
    if (!initialized) {
        std::cerr << "ERROR: Cannot reinitialize compute resources: shaders not initialized!" << std::endl;
        return;
    }
    
    // Instead of immediate recreation, flag for deferred recreation to avoid race conditions
    std::cout << "Flagging compute resources for recreation: " << context.width << "x" << context.height << std::endl;
    needsResourceRecreation = true;
    pendingWidth = context.width;
    pendingHeight = context.height;
    g_context = context; // Update stored context
}

// Internal function to actually recreate resources (called from generateFractalData)
static void doActualResourceRecreation() {
    std::cout << "\n=== ACTUALLY RECREATING COMPUTE SHADER RESOURCES ===" << std::endl;
    std::cout << "Recreating for dimensions: " << pendingWidth << "x" << pendingHeight << std::endl;
    
    // Update context dimensions
    g_context.width = pendingWidth;
    g_context.height = pendingHeight;
    
    // === RECREATE SIZE-DEPENDENT BUFFERS ===
    std::cout << "Recreating iteration buffer..." << std::endl;
    
    // Recreate iteration buffer (stores normalized iteration counts as floats)
    wgpu::BufferDescriptor iterationBufferDesc{};
    iterationBufferDesc.size = pendingWidth * pendingHeight * sizeof(float);
    iterationBufferDesc.usage = wgpu::BufferUsage::Storage;
    iterationBuffer = g_context.device.CreateBuffer(&iterationBufferDesc);
    std::cout << "Iteration buffer created, size: " << iterationBufferDesc.size << " bytes" << std::endl;
    
    // Recreate output buffer (stores final RGBA colors)
    std::cout << "Recreating output buffer..." << std::endl;
    wgpu::BufferDescriptor outputBufferDesc{};
    outputBufferDesc.size = pendingWidth * pendingHeight * 4; // RGBA bytes
    outputBufferDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    outputBuffer = g_context.device.CreateBuffer(&outputBufferDesc);
    std::cout << "Output buffer created, size: " << outputBufferDesc.size << " bytes" << std::endl;
    
    // === RECREATE BIND GROUPS (they reference the buffers) ===
    
    // Recreate bind group for iteration shader
    wgpu::BindGroupEntry iterationBindGroupEntries[2] = {};
    iterationBindGroupEntries[0].binding = 0;
    iterationBindGroupEntries[0].buffer = iterationBuffer;
    iterationBindGroupEntries[0].size = iterationBufferDesc.size;
    
    iterationBindGroupEntries[1].binding = 1;
    iterationBindGroupEntries[1].buffer = uniformBuffer;
    iterationBindGroupEntries[1].size = sizeof(FractalParams);
    
    wgpu::BindGroupDescriptor iterationBindGroupDesc{};
    iterationBindGroupDesc.layout = iterationBindGroupLayout;
    iterationBindGroupDesc.entryCount = 2;
    iterationBindGroupDesc.entries = iterationBindGroupEntries;
    iterationBindGroup = g_context.device.CreateBindGroup(&iterationBindGroupDesc);
    
    // Recreate bind group for coloring shader
    wgpu::BindGroupEntry coloringBindGroupEntries[3] = {};
    coloringBindGroupEntries[0].binding = 0;
    coloringBindGroupEntries[0].buffer = iterationBuffer;
    coloringBindGroupEntries[0].size = iterationBufferDesc.size;
    
    coloringBindGroupEntries[1].binding = 1;
    coloringBindGroupEntries[1].buffer = outputBuffer;
    coloringBindGroupEntries[1].size = outputBufferDesc.size;
    
    coloringBindGroupEntries[2].binding = 2;
    coloringBindGroupEntries[2].buffer = coloringUniformBuffer;
    coloringBindGroupEntries[2].size = sizeof(ColoringParams);
    
    wgpu::BindGroupDescriptor coloringBindGroupDesc{};
    coloringBindGroupDesc.layout = coloringBindGroupLayout;
    coloringBindGroupDesc.entryCount = 3;
    coloringBindGroupDesc.entries = coloringBindGroupEntries;
    coloringBindGroup = g_context.device.CreateBindGroup(&coloringBindGroupDesc);
    
    // === RECREATE FRACTAL TEXTURE ===
    
    // Recreate fractal texture
    wgpu::TextureDescriptor textureDesc{};
    textureDesc.size = {pendingWidth, pendingHeight, 1};
    textureDesc.format = wgpu::TextureFormat::RGBA8Unorm;
    textureDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    fractalTexture = g_context.device.CreateTexture(&textureDesc);
    fractalTextureView = fractalTexture.CreateView();
    
    // === RECREATE RENDER BIND GROUP (references the texture) ===
    
    // Recreate bind group for render pipeline
    wgpu::BindGroupEntry renderBindGroupEntries[2] = {};
    renderBindGroupEntries[0].binding = 0;
    renderBindGroupEntries[0].textureView = fractalTextureView;
    
    renderBindGroupEntries[1].binding = 1;
    renderBindGroupEntries[1].sampler = fractalSampler;
    
    wgpu::BindGroupDescriptor renderBindGroupDesc{};
    renderBindGroupDesc.layout = renderBindGroupLayout;
    renderBindGroupDesc.entryCount = 2;
    renderBindGroupDesc.entries = renderBindGroupEntries;
    renderBindGroup = g_context.device.CreateBindGroup(&renderBindGroupDesc);
    
    // Clear the flag
    needsResourceRecreation = false;
    
    std::cout << "=== COMPUTE SHADER RESOURCES RECREATED SUCCESSFULLY! ===" << std::endl;
}

void generateFractalData(const WebGPUContext& context, wgpu::CommandEncoder& encoder, 
                        wgpu::Buffer& outputBuffer,
                        double viewX, double viewY, double viewWidth, double viewHeight) {
    if (!initialized) return;
    
    // Begin compute debugging
    beginComputeDebugging(const_cast<WebGPUContext&>(context));
    
    // Check if we need to recreate resources (deferred from resize callback)
    if (needsResourceRecreation) {
        doActualResourceRecreation();
    }
    
    // Calculate Julia constants from settings
    double julia_cx = g_settings.julia.radius * cos(g_settings.julia.angle);
    double julia_cy = g_settings.julia.radius * sin(g_settings.julia.angle);
    
    // Prepare fractal uniform data
    FractalParams fractalParams{};
    fractalParams.viewX = static_cast<float>(viewX);
    fractalParams.viewY = static_cast<float>(viewY);
    fractalParams.viewWidth = static_cast<float>(viewWidth);
    fractalParams.viewHeight = static_cast<float>(viewHeight);
    fractalParams.width = static_cast<uint32_t>(context.width);
    fractalParams.height = static_cast<uint32_t>(context.height);
    fractalParams.maxIterations = static_cast<uint32_t>(g_settings.visualization.maxIterations);
    fractalParams.juliaCX = static_cast<float>(julia_cx);
    fractalParams.juliaCY = static_cast<float>(julia_cy);
    fractalParams.coloringMode = static_cast<uint32_t>(g_settings.visualization.coloringMode);
    
    // Prepare coloring uniform data
    ColoringParams coloringParams{};
    coloringParams.width = static_cast<uint32_t>(context.width);
    coloringParams.height = static_cast<uint32_t>(context.height);
    coloringParams.coloringMode = static_cast<uint32_t>(g_settings.visualization.coloringMode);
    coloringParams.maxIterations = static_cast<uint32_t>(g_settings.visualization.maxIterations);
    
    // Static variables to track frames and timing for observability
    static uint32_t frameCount = 0;
    static auto lastLogTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto timeSinceLastLog = std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastLogTime);
    
    // Print debug info on first frame or every 10 seconds
    if (frameCount == 0 || timeSinceLastLog.count() >= 10) {
        std::cout << "\n===== FRACTAL PARAMS DEBUG (Frame " << frameCount << ") =====" << std::endl;
        std::cout << "Viewport: viewX=" << fractalParams.viewX << ", viewY=" << fractalParams.viewY << std::endl;
        std::cout << "          viewWidth=" << fractalParams.viewWidth << ", viewHeight=" << fractalParams.viewHeight << std::endl;
        std::cout << "Dimensions: width=" << fractalParams.width << ", height=" << fractalParams.height << std::endl;
        std::cout << "Iterations: maxIterations=" << fractalParams.maxIterations << std::endl;
        std::cout << "Julia constants: juliaCX=" << fractalParams.juliaCX << ", juliaCY=" << fractalParams.juliaCY << std::endl;
        std::cout << "Julia settings: radius=" << g_settings.julia.radius << ", angle=" << g_settings.julia.angle << std::endl;
        std::cout << "Julia settings: angleVelocity=" << g_settings.julia.angleVelocity << ", motionEnabled=" << g_settings.julia.motionEnabled << std::endl;
        std::cout << "Visualization: mode=" << getVisualizationModeName(g_settings.visualization.mode) << std::endl;
        std::cout << "Coloring: mode=" << getColoringModeName(g_settings.visualization.coloringMode) << std::endl;
        std::cout << "===============================================\n" << std::endl;
        
        if (timeSinceLastLog.count() >= 10) {
            lastLogTime = currentTime;
        }
    }
    frameCount++;
    
    // Update uniform buffers
    context.queue.WriteBuffer(uniformBuffer, 0, &fractalParams, sizeof(FractalParams));
    context.queue.WriteBuffer(coloringUniformBuffer, 0, &coloringParams, sizeof(ColoringParams));
    
    // Create compute pass
    wgpu::ComputePassDescriptor computePassDesc{};
    wgpu::ComputePassEncoder computePass = encoder.BeginComputePass(&computePassDesc);
    
    // === STEP 1: COMPUTE ITERATION COUNTS ===
    computePass.SetPipeline(iterationPipeline);
    computePass.SetBindGroup(0, iterationBindGroup);
    
    // Dispatch iteration compute shader
    uint32_t workgroupsX = (context.width + 15) / 16;
    uint32_t workgroupsY = (context.height + 15) / 16;
    
    // DEBUG: Log dispatch dimensions after resize
    static uint32_t lastLoggedWidth = 0;
    static uint32_t lastLoggedHeight = 0;
    if (context.width != lastLoggedWidth || context.height != lastLoggedHeight) {
        std::cout << "DISPATCH DEBUG: context dimensions: " << context.width << "x" << context.height << std::endl;
        std::cout << "DISPATCH DEBUG: workgroups: " << workgroupsX << "x" << workgroupsY << std::endl;
        std::cout << "DISPATCH DEBUG: total threads: " << (workgroupsX * 16) << "x" << (workgroupsY * 16) << std::endl;
        lastLoggedWidth = context.width;
        lastLoggedHeight = context.height;
    }
    
    computePass.DispatchWorkgroups(workgroupsX, workgroupsY);
    
    // === STEP 2: COMPUTE COLORS FROM ITERATION COUNTS ===
    computePass.SetPipeline(coloringPipeline);
    computePass.SetBindGroup(0, coloringBindGroup);
    
    // Dispatch coloring compute shader
    computePass.DispatchWorkgroups(workgroupsX, workgroupsY);
    
    computePass.End();
    
    // End compute debugging
    endComputeDebugging(const_cast<WebGPUContext&>(context));
    
    // Copy from storage buffer to texture
    wgpu::TexelCopyBufferInfo texelCopyBufferInfo{};
    texelCopyBufferInfo.buffer = ::outputBuffer;
    texelCopyBufferInfo.layout.offset = 0;
    texelCopyBufferInfo.layout.bytesPerRow = context.width * 4;
    texelCopyBufferInfo.layout.rowsPerImage = context.height;
    
    wgpu::TexelCopyTextureInfo texelCopyTextureInfo{};
    texelCopyTextureInfo.texture = fractalTexture;
    texelCopyTextureInfo.mipLevel = 0;
    texelCopyTextureInfo.origin = {0, 0, 0};
    texelCopyTextureInfo.aspect = wgpu::TextureAspect::All;
    
    wgpu::Extent3D copySize{};
    copySize.width = context.width;
    copySize.height = context.height;
    copySize.depthOrArrayLayers = 1;
    
    // DEBUG: Log copy dimensions after resize
    static uint32_t lastCopyWidth = 0;
    static uint32_t lastCopyHeight = 0;
    if (context.width != lastCopyWidth || context.height != lastCopyHeight) {
        std::cout << "COPY DEBUG: buffer-to-texture copy size: " << copySize.width << "x" << copySize.height << std::endl;
        std::cout << "COPY DEBUG: bytes per row: " << texelCopyBufferInfo.layout.bytesPerRow << std::endl;
        std::cout << "COPY DEBUG: rows per image: " << texelCopyBufferInfo.layout.rowsPerImage << std::endl;
        lastCopyWidth = context.width;
        lastCopyHeight = context.height;
    }
    
    encoder.CopyBufferToTexture(&texelCopyBufferInfo, &texelCopyTextureInfo, &copySize);
    
    // // === DEBUG: Add debugging for pink screen issue ===
    // static bool debugExecuted = false;
    // if (!debugExecuted && frameCount < 5) {  // Debug first few frames
    //     std::cout << "\n=== DEBUGGING PINK SCREEN ISSUE (Frame " << frameCount << ") ===" << std::endl;
        
    //     // Simple debug: just log that we're executing compute shaders
    //     std::cout << "DEBUG: Compute shaders dispatched successfully" << std::endl;
    //     std::cout << "DEBUG: Workgroups dispatched: " << workgroupsX << "x" << workgroupsY << std::endl;
    //     std::cout << "DEBUG: Total threads: " << (workgroupsX * 16) << "x" << (workgroupsY * 16) << std::endl;
    //     std::cout << "DEBUG: Buffer size: " << (context.width * context.height * 4) << " bytes" << std::endl;
    //     std::cout << "DEBUG: Buffer-to-texture copy completed" << std::endl;
    //     std::cout << "DEBUG: Texture format: RGBA8Unorm" << std::endl;
    //     std::cout << "DEBUG: Copy size: " << copySize.width << "x" << copySize.height << std::endl;
    //     std::cout << "DEBUG: Bytes per row: " << texelCopyBufferInfo.layout.bytesPerRow << std::endl;
        
    //     if (frameCount == 1) {
    //         std::cout << "DEBUG: Skipping buffer mapping for now to avoid hanging" << std::endl;
    //         debugExecuted = true;
    //     }
    // }
    // === END DEBUG ===
}

void createFractalComputePipelines(const WebGPUContext& context) {
    // This function is called from initComputeShaders, so we don't need additional logic here
    std::cout << "Fractal compute pipelines created" << std::endl;
}

const char* getVertexShaderSource() {
    return vertexShaderCode;
}

const char* getFragmentShaderSource() {
    return fragmentShaderCode;
}

wgpu::BindGroupLayout getRenderBindGroupLayout() {
    return renderBindGroupLayout;
}

wgpu::BindGroup getRenderBindGroup() {
    return renderBindGroup;
}

wgpu::Buffer& getOutputBuffer() {
    return outputBuffer;
}

// === DEBUG FUNCTIONS IMPLEMENTATION ===

void debugDumpBufferToFile(const WebGPUContext& context, wgpu::Buffer& buffer, 
                          uint32_t width, uint32_t height, const char* filename) {
    std::cout << "DEBUG: Dumping buffer to file: " << filename << std::endl;
    
    // Create a staging buffer to read the data
    wgpu::BufferDescriptor stagingBufferDesc{};
    stagingBufferDesc.size = width * height * 4; // RGBA
    stagingBufferDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    wgpu::Buffer stagingBuffer = context.device.CreateBuffer(&stagingBufferDesc);
    
    // Create command encoder and copy data
    wgpu::CommandEncoderDescriptor encoderDesc{};
    wgpu::CommandEncoder encoder = context.device.CreateCommandEncoder(&encoderDesc);
    encoder.CopyBufferToBuffer(buffer, 0, stagingBuffer, 0, stagingBufferDesc.size);
    
    // Submit and wait
    wgpu::CommandBufferDescriptor cmdBufferDesc{};
    wgpu::CommandBuffer cmdBuffer = encoder.Finish(&cmdBufferDesc);
    context.queue.Submit(1, &cmdBuffer);
    
    // Map the staging buffer and read data
    bool mapCompleted = false;
    const void* mappedData = nullptr;
    
    stagingBuffer.MapAsync(wgpu::MapMode::Read, 0, stagingBufferDesc.size,
                          wgpu::CallbackMode::WaitAnyOnly,
                          [&mapCompleted](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                              mapCompleted = true;
                              if (status != wgpu::MapAsyncStatus::Success) {
                                  std::cerr << "Failed to map buffer for reading: " << static_cast<int>(status) << std::endl;
                              }
                          });
    
    // Wait for mapping to complete
    while (!mapCompleted) {
        context.instance.ProcessEvents();
    }
    
    if (mapCompleted) {
        mappedData = stagingBuffer.GetConstMappedRange(0, stagingBufferDesc.size);
        
        if (mappedData) {
            // Write data to file
            std::ofstream file(filename, std::ios::binary);
            file.write(static_cast<const char*>(mappedData), stagingBufferDesc.size);
            file.close();
            
            std::cout << "DEBUG: Buffer dumped successfully. Size: " << stagingBufferDesc.size << " bytes" << std::endl;
            
            // Check first few pixels for debugging
            const uint32_t* pixels = static_cast<const uint32_t*>(mappedData);
            std::cout << "DEBUG: First 10 pixels (RGBA as uint32):" << std::endl;
            for (int i = 0; i < std::min(10u, width * height); i++) {
                uint32_t pixel = pixels[i];
                uint8_t r = pixel & 0xFF;
                uint8_t g = (pixel >> 8) & 0xFF;
                uint8_t b = (pixel >> 16) & 0xFF;
                uint8_t a = (pixel >> 24) & 0xFF;
                std::cout << "  Pixel " << i << ": R=" << (int)r << " G=" << (int)g 
                         << " B=" << (int)b << " A=" << (int)a << " (0x" << std::hex << pixel << std::dec << ")" << std::endl;
            }
        }
        
        stagingBuffer.Unmap();
    }
}

void debugDumpTextureToFile(const WebGPUContext& context, wgpu::Texture& texture,
                           uint32_t width, uint32_t height, const char* filename) {
    std::cout << "DEBUG: Dumping texture to file: " << filename << std::endl;
    
    // Create a buffer to copy texture data to
    wgpu::BufferDescriptor bufferDesc{};
    bufferDesc.size = width * height * 4; // RGBA
    bufferDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    wgpu::Buffer readBuffer = context.device.CreateBuffer(&bufferDesc);
    
    // Create command encoder and copy texture to buffer
    wgpu::CommandEncoderDescriptor encoderDesc{};
    wgpu::CommandEncoder encoder = context.device.CreateCommandEncoder(&encoderDesc);
    
    wgpu::TexelCopyTextureInfo srcTexture{};
    srcTexture.texture = texture;
    srcTexture.mipLevel = 0;
    srcTexture.origin = {0, 0, 0};
    srcTexture.aspect = wgpu::TextureAspect::All;
    
    wgpu::TexelCopyBufferInfo dstBuffer{};
    dstBuffer.buffer = readBuffer;
    dstBuffer.layout.offset = 0;
    dstBuffer.layout.bytesPerRow = width * 4;
    dstBuffer.layout.rowsPerImage = height;
    
    wgpu::Extent3D copySize{};
    copySize.width = width;
    copySize.height = height;
    copySize.depthOrArrayLayers = 1;
    
    encoder.CopyTextureToBuffer(&srcTexture, &dstBuffer, &copySize);
    
    // Submit and wait
    wgpu::CommandBufferDescriptor cmdBufferDesc{};
    wgpu::CommandBuffer cmdBuffer = encoder.Finish(&cmdBufferDesc);
    context.queue.Submit(1, &cmdBuffer);
    
    // Now use debugDumpBufferToFile to save the data
    debugDumpBufferToFile(context, readBuffer, width, height, filename);
}

void debugVerifyComputeShaderExecution(const WebGPUContext& context) {
    std::cout << "\n=== DEBUG: Verifying Compute Shader Execution ===" << std::endl;
    
    if (!initialized) {
        std::cout << "DEBUG: Compute shaders not initialized!" << std::endl;
        return;
    }
    
    // Check if the color squares debug pattern is working
    // Create a staging buffer to read output buffer data  
    uint32_t bufferSize = context.width * context.height * 4;
    wgpu::BufferDescriptor stagingBufferDesc{};
    stagingBufferDesc.size = bufferSize;
    stagingBufferDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    wgpu::Buffer stagingBuffer = context.device.CreateBuffer(&stagingBufferDesc);
    
    // Copy output buffer to staging buffer
    wgpu::CommandEncoderDescriptor encoderDesc{};
    wgpu::CommandEncoder encoder = context.device.CreateCommandEncoder(&encoderDesc);
    encoder.CopyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, bufferSize);
    
    wgpu::CommandBufferDescriptor cmdBufferDesc{};
    wgpu::CommandBuffer cmdBuffer = encoder.Finish(&cmdBufferDesc);
    context.queue.Submit(1, &cmdBuffer);
    
    // Map and check data
    bool mapCompleted = false;
    const void* mappedData = nullptr;
    
    stagingBuffer.MapAsync(wgpu::MapMode::Read, 0, stagingBufferDesc.size,
                          wgpu::CallbackMode::WaitAnyOnly,
                          [&mapCompleted](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                              mapCompleted = true;
                              if (status != wgpu::MapAsyncStatus::Success) {
                                  std::cerr << "Failed to map staging buffer: " << static_cast<int>(status) << std::endl;
                              }
                          });
    
    while (!mapCompleted) {
        context.instance.ProcessEvents();
    }
    
    if (mapCompleted) {
        mappedData = stagingBuffer.GetConstMappedRange(0, bufferSize);
        
        if (mappedData) {
            // Check the debug squares:
            // Top-left (0,0) should be red (0xFF0000FF in RGBA)
            const uint32_t* pixels = static_cast<const uint32_t*>(mappedData);
            uint32_t topLeft = pixels[0];
            
            // Top-right (width-50, 0) should be green if debug pattern is working
            if (context.width >= 100) {
                uint32_t topRight = pixels[context.width - 50];
                std::cout << "DEBUG: Top-left pixel: 0x" << std::hex << topLeft << std::dec << std::endl;
                std::cout << "DEBUG: Top-right pixel: 0x" << std::hex << topRight << std::dec << std::endl;
                
                // Extract RGBA values (assuming ABGR packing as used in shader)
                uint8_t tl_r = topLeft & 0xFF;
                uint8_t tl_g = (topLeft >> 8) & 0xFF;
                uint8_t tl_b = (topLeft >> 16) & 0xFF;
                uint8_t tl_a = (topLeft >> 24) & 0xFF;
                
                uint8_t tr_r = topRight & 0xFF;
                uint8_t tr_g = (topRight >> 8) & 0xFF;
                uint8_t tr_b = (topRight >> 16) & 0xFF;
                uint8_t tr_a = (topRight >> 24) & 0xFF;
                
                std::cout << "DEBUG: Top-left RGBA: " << (int)tl_r << "," << (int)tl_g 
                         << "," << (int)tl_b << "," << (int)tl_a << std::endl;
                std::cout << "DEBUG: Top-right RGBA: " << (int)tr_r << "," << (int)tr_g 
                         << "," << (int)tr_b << "," << (int)tr_a << std::endl;
                
                if (tl_r == 255 && tl_g == 0 && tl_b == 0) {
                    std::cout << "DEBUG: ✓ Compute shader debug pattern working - red square detected!" << std::endl;
                } else {
                    std::cout << "DEBUG: ✗ Compute shader debug pattern NOT working - expected red square" << std::endl;
                }
                
                if (tr_g == 255 && tr_r == 0 && tr_b == 0) {
                    std::cout << "DEBUG: ✓ Green square detected in top-right!" << std::endl;
                } else {
                    std::cout << "DEBUG: ✗ Green square NOT detected in top-right" << std::endl;
                }
            }
        }
        
        stagingBuffer.Unmap();
    }
    
    std::cout << "=== DEBUG: Compute shader verification complete ===\n" << std::endl;
}

void debugSaveBufferAsPPM(const WebGPUContext& context, wgpu::Buffer& buffer,
                          uint32_t width, uint32_t height, const char* filename) {
    std::cout << "DEBUG: Saving buffer as PPM: " << filename << std::endl;
    
    // Create staging buffer
    wgpu::BufferDescriptor stagingBufferDesc{};
    stagingBufferDesc.size = width * height * 4;
    stagingBufferDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    wgpu::Buffer stagingBuffer = context.device.CreateBuffer(&stagingBufferDesc);
    
    // Copy data
    wgpu::CommandEncoderDescriptor encoderDesc{};
    wgpu::CommandEncoder encoder = context.device.CreateCommandEncoder(&encoderDesc);
    encoder.CopyBufferToBuffer(buffer, 0, stagingBuffer, 0, stagingBufferDesc.size);
    
    wgpu::CommandBufferDescriptor cmdBufferDesc{};
    wgpu::CommandBuffer cmdBuffer = encoder.Finish(&cmdBufferDesc);
    context.queue.Submit(1, &cmdBuffer);
    
    // Map and save as PPM
    bool mapCompleted = false;
    const void* mappedData = nullptr;
    
    stagingBuffer.MapAsync(wgpu::MapMode::Read, 0, stagingBufferDesc.size,
                          wgpu::CallbackMode::WaitAnyOnly,
                          [&mapCompleted](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                              mapCompleted = true;
                          });
    
    while (!mapCompleted) {
        context.instance.ProcessEvents();
    }
    
    if (mapCompleted) {
        mappedData = stagingBuffer.GetConstMappedRange(0, stagingBufferDesc.size);
        
        if (mappedData) {
            std::ofstream file(filename);
            file << "P3\n" << width << " " << height << "\n255\n";
            
            const uint32_t* pixels = static_cast<const uint32_t*>(mappedData);
            for (uint32_t y = 0; y < height; y++) {
                for (uint32_t x = 0; x < width; x++) {
                    uint32_t pixel = pixels[y * width + x];
                    uint8_t r = pixel & 0xFF;
                    uint8_t g = (pixel >> 8) & 0xFF; 
                    uint8_t b = (pixel >> 16) & 0xFF;
                    file << (int)r << " " << (int)g << " " << (int)b << " ";
                }
                file << "\n";
            }
            file.close();
            std::cout << "DEBUG: PPM file saved successfully" << std::endl;
        }
        
        stagingBuffer.Unmap();
    }
}

void validateComputeShaderSetup(const WebGPUContext& context) {
    std::cout << "Validating compute shader setup..." << std::endl;
    
    bool isValid = true;
    
    if (!initialized) {
        std::cerr << "ERROR: Compute shaders not initialized" << std::endl;
        isValid = false;
    }
    
    if (!iterationPipeline) {
        std::cerr << "ERROR: Iteration pipeline is null" << std::endl;
        isValid = false;
    }
    
    if (!coloringPipeline) {
        std::cerr << "ERROR: Coloring pipeline is null" << std::endl;
        isValid = false;
    }
    
    if (!iterationBuffer) {
        std::cerr << "ERROR: Iteration buffer is null" << std::endl;
        isValid = false;
    }
    
    if (!outputBuffer) {
        std::cerr << "ERROR: Output buffer is null" << std::endl;
        isValid = false;
    }
    
    if (!fractalTexture) {
        std::cerr << "ERROR: Fractal texture is null" << std::endl;
        isValid = false;
    }
    
    if (isValid) {
        std::cout << "Compute shader setup validation passed" << std::endl;
        std::cout << "    - Framebuffer size: " << context.width << "x" << context.height << std::endl;
        std::cout << "    - Iteration buffer size: " << (context.width * context.height * sizeof(float)) << " bytes" << std::endl;
        std::cout << "    - Output buffer size: " << (context.width * context.height * 4) << " bytes" << std::endl;
    } else {
        std::cerr << "CRITICAL: Compute shader setup validation FAILED!" << std::endl;
    }
}

void beginComputeDebugging(WebGPUContext& context) {
    if (context.debugEnabled && context.totalFrames % 60 == 0) {
        std::cout << "Beginning compute operations for frame " << context.currentFrame.frameNumber << std::endl;
    }
}

void endComputeDebugging(WebGPUContext& context) {
    if (context.debugEnabled && context.totalFrames % 60 == 0) {
        std::cout << "Compute operations completed for frame " << context.currentFrame.frameNumber << std::endl;
    }
}

void debugComputeShaderExecution(const WebGPUContext& context) {
    std::cout << "Executing compute shader debug analysis..." << std::endl;
    
    // Validate pipeline objects
    if (!iterationPipeline || !coloringPipeline) {
        std::cerr << "ERROR: Compute pipelines not properly initialized!" << std::endl;
        return;
    }
    
    // Validate bind groups
    if (!iterationBindGroup || !coloringBindGroup) {
        std::cerr << "ERROR: Compute bind groups not properly initialized!" << std::endl;
        return;
    }
    
    // Validate buffers
    if (!iterationBuffer || !outputBuffer) {
        std::cerr << "ERROR: Compute buffers not properly initialized!" << std::endl;
        return;
    }
    
    std::cout << "Compute shader execution validation passed" << std::endl;
    std::cout << "    - Iteration pipeline ready" << std::endl;
    std::cout << "    - Coloring pipeline ready" << std::endl;
    std::cout << "    - All bind groups ready" << std::endl;
    std::cout << "    - All buffers ready" << std::endl;
} 