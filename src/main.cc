#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>

#include "fractal_settings.h"
#include "input_handler.h"
#include "webgpu_setup.h"
#include "compute_shaders.h"

// Global fractal settings instance
FractalSettings g_settings;

int main(int argc, char *argv[]) {
  // Initialize fractal settings (already done via default values, but showing explicit initialization)
  std::cout << "Initializing fractal settings..." << std::endl;
  std::cout << "Julia radius: " << g_settings.julia.radius << std::endl;
  std::cout << "Julia angle velocity: " << g_settings.julia.angleVelocity << std::endl;
  std::cout << "Visualization mode: " << getVisualizationModeName(g_settings.visualization.mode) << std::endl;
  std::cout << "Target FPS: " << g_settings.targetFPS << std::endl;

  // Initialize WebGPU and create window
  WebGPUContext context = initWebGPU();
  if (!context.window) {
    std::cerr << "Failed to initialize WebGPU" << std::endl;
    return EXIT_FAILURE;
  }

  // Setup input handlers
  initInputHandlers(context.window, &context);

  // Initialize compute shaders
  initComputeShaders(context);

  // Create a shader module for the vertex shader
  wgpu::ShaderModule vertexShaderModule = createShaderModule(context.device, getVertexShaderSource());

  // Create a shader module for the fragment shader
  wgpu::ShaderModule fragmentShaderModule = createShaderModule(context.device, getFragmentShaderSource());

  // Create render pipeline layout with bind group layout from compute shaders
  wgpu::PipelineLayoutDescriptor layoutDesc{};
  wgpu::BindGroupLayout bindGroupLayout = getRenderBindGroupLayout();
  layoutDesc.bindGroupLayoutCount = 1;
  layoutDesc.bindGroupLayouts = &bindGroupLayout;
  wgpu::PipelineLayout pipelineLayout = context.device.CreatePipelineLayout(&layoutDesc);

  // Create pipeline
  wgpu::RenderPipelineDescriptor pipelineDesc{};
  pipelineDesc.layout = pipelineLayout;

  // Vertex state
  pipelineDesc.vertex.module = vertexShaderModule;
  pipelineDesc.vertex.entryPoint = "main";

  // Fragment state
  wgpu::FragmentState fragmentState{};
  fragmentState.module = fragmentShaderModule;
  fragmentState.entryPoint = "main";
  wgpu::ColorTargetState colorTargetState{};
  colorTargetState.format = context.swapChainFormat;
  fragmentState.targetCount = 1;
  fragmentState.targets = &colorTargetState;
  pipelineDesc.fragment = &fragmentState;

  // Other pipeline state
  pipelineDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
  pipelineDesc.primitive.stripIndexFormat = wgpu::IndexFormat::Undefined;
  wgpu::MultisampleState msState{};
  msState.count = 1;
  msState.mask = 0xFFFFFFFF;
  pipelineDesc.multisample = msState;

  // Create the pipeline
  wgpu::RenderPipeline pipeline = context.device.CreateRenderPipeline(&pipelineDesc);

  // Create a dummy output buffer for the generateFractalData function (it uses the internal buffer)
  wgpu::BufferDescriptor dummyBufferDesc{};
  dummyBufferDesc.size = 4;
  dummyBufferDesc.usage = wgpu::BufferUsage::Storage;
  wgpu::Buffer dummyBuffer = context.device.CreateBuffer(&dummyBufferDesc);

  // Print controls info
  std::cout << "Controls:" << std::endl;
  //std::cout << "  F: Toggle exponential filter (currently " << (g_settings.visualization.useExponentialFilter ? "ON" : "OFF") << ")" << std::endl;
  std::cout << "  D: Toggle distance field visualization" << std::endl;
  std::cout << "  M: Toggle iteration mask visualization" << std::endl;
  std::cout << "  T: Toggle fractal type" << std::endl;
  std::cout << "  C: Cycle coloring mode (currently " << getColoringModeName(g_settings.visualization.coloringMode) << ")" << std::endl;
  std::cout << "  SPACE: Toggle Julia circular motion (currently " << (g_settings.julia.motionEnabled ? "ON" : "OFF") << ")" << std::endl;
  std::cout << "  === VIEWPORT MOVEMENT ===" << std::endl;
  std::cout << "  Arrow Keys: Move viewport left/right/up/down" << std::endl;
  std::cout << "  J: Zoom in" << std::endl;
  std::cout << "  K: Zoom out" << std::endl;
  std::cout << "  R: Reset viewport to default values" << std::endl;
  std::cout << "  === ITERATION CONTROL ===" << std::endl;
  std::cout << "  I: Increase max iterations (currently " << g_settings.visualization.maxIterations << ")" << std::endl;
  std::cout << "  O: Decrease max iterations" << std::endl;
  std::cout << "  === ANGLE VELOCITY CONTROL ===" << std::endl;
  std::cout << "  Q: Decrease Julia angle velocity (currently " << g_settings.julia.angleVelocity << ")" << std::endl;
  std::cout << "  E: Increase Julia angle velocity" << std::endl;
  std::cout << "  ESC: Quit" << std::endl;

  // Main loop
  unsigned int frameCount = 0;
  auto lastFrameTime = std::chrono::high_resolution_clock::now();
  
  std::cout << "Starting main loop with frame-by-frame debugging..." << std::endl;
  
  while (!glfwWindowShouldClose(context.window)) {
    glfwPollEvents();
    
    // CRITICAL: Process WebGPU events to pump error callbacks!
    // Without this, SetUncapturedErrorCallback never fires!
    context.instance.ProcessEvents();
    
    // Begin frame debugging
    beginFrameDebugging(context);
    
    // Check system health periodically
    if (frameCount % 60 == 0 && frameCount > 0) {
      if (!checkRenderingHealth(context)) {
        std::cerr << "WARNING: Rendering health check failed at frame " << frameCount << std::endl;
        printFrameStats(context);
      }
    }
    
    // Update Julia angle for circular motion if enabled
    if (g_settings.julia.motionEnabled && !isLeftButtonHeld()) {
        // Scale angle velocity by diagonal length to make movement proportional to zoom level
        double diagonal = sqrt(g_settings.viewport.viewWidth * g_settings.viewport.viewWidth + 
                              g_settings.viewport.viewHeight * g_settings.viewport.viewHeight);
        double scaledAngleVelocity = g_settings.julia.angleVelocity * diagonal;
        
        g_settings.julia.angle += scaledAngleVelocity;
        if (g_settings.julia.angle >= 2.0 * M_PI) {
            g_settings.julia.angle -= 2.0 * M_PI;  // Keep angle in [0, 2π) range
        }
    }
    
    // Get the current texture from the surface
    wgpu::SurfaceTexture nextTexture{};
    context.surface.GetCurrentTexture(&nextTexture);
    
    if (nextTexture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
        nextTexture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
      std::cerr << "CRITICAL: Failed to get current surface texture: " << static_cast<int>(nextTexture.status) << std::endl;
      std::cerr << "    This may indicate a serious graphics driver or system issue!" << std::endl;
      break;
    }
    
    wgpu::TextureView currentTextureView = nextTexture.texture.CreateView();
    
    // Create command encoder
    wgpu::CommandEncoderDescriptor encoderDesc{};
    encoderDesc.label = "Main Frame Encoder";
    wgpu::CommandEncoder encoder = context.device.CreateCommandEncoder(&encoderDesc);
    
    // Generate fractal data using compute shader
    generateFractalData(context, encoder, dummyBuffer,
                       g_settings.viewport.viewX, g_settings.viewport.viewY, 
                       g_settings.viewport.viewWidth, g_settings.viewport.viewHeight);
    
    // Mark compute phase complete
    context.currentFrame.computeComplete = true;
    
    // Begin render pass
    wgpu::RenderPassColorAttachment colorAttachment{};
    colorAttachment.view = currentTextureView;
    colorAttachment.loadOp = wgpu::LoadOp::Clear;
    colorAttachment.storeOp = wgpu::StoreOp::Store;
    colorAttachment.clearValue = {0.0f, 0.0f, 0.0f, 1.0f}; // Black
    
    wgpu::RenderPassDescriptor renderPassDesc{};
    renderPassDesc.label = "Main Render Pass";
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &colorAttachment;
    
    wgpu::RenderPassEncoder renderPass = encoder.BeginRenderPass(&renderPassDesc);
    
    // Render the fractal texture as a full-screen quad
    renderPass.SetPipeline(pipeline);
    renderPass.SetBindGroup(0, getRenderBindGroup());
    renderPass.Draw(6, 1, 0, 0); // 6 vertices for 2 triangles (full-screen quad)
    
    renderPass.End();
    
    // Mark rendering phase complete
    context.currentFrame.renderingComplete = true;
    
    // Submit commands
    wgpu::CommandBufferDescriptor cmdBufferDesc{};
    cmdBufferDesc.label = "Main Command Buffer";
    wgpu::CommandBuffer cmdBuffer = encoder.Finish(&cmdBufferDesc);
    context.queue.Submit(1, &cmdBuffer);
    
    // Present the frame
    context.surface.Present();
    
    frameCount++;
    
    // End frame debugging
    endFrameDebugging(context);
    
    // Frame rate limiting
    auto targetFrameTime = std::chrono::milliseconds(1000 / g_settings.targetFPS);
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto frameTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastFrameTime);
    
    if (frameTime < targetFrameTime) {
        std::this_thread::sleep_for(targetFrameTime - frameTime);
    }
    lastFrameTime = std::chrono::high_resolution_clock::now();
  }
  
  std::cout << "Main loop ended. Final statistics:" << std::endl;
  printFrameStats(context);
  
  // Cleanup
  cleanupComputeShaders();
  cleanupWebGPU(context);
  
  return EXIT_SUCCESS;
} 