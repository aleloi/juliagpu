#include <webgpu/webgpu_cpp.h>
#include <webgpu/webgpu_cpp_print.h>
#include <GLFW/glfw3.h>
#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#include "metal_layer_helper.h" // Include our Metal helper header
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#endif
#include <GLFW/glfw3native.h>

#include <cstdlib>
#include <iostream>
#include <array>

// Error callback for GLFW
void errorCallback(int error, const char* description) {
  std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

// Vertex shader
const char* vertexShaderCode = R"(
  @vertex
  fn main(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
      vec2<f32>(0.0, 0.5),
      vec2<f32>(-0.5, -0.5),
      vec2<f32>(0.5, -0.5)
    );
    return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
  }
)";

// Fragment shader
const char* fragmentShaderCode = R"(
  @fragment
  fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0); // Red color
  }
)";

int main(int argc, char *argv[]) {
  // Initialize GLFW
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return EXIT_FAILURE;
  }
  glfwSetErrorCallback(errorCallback);

  // Configure GLFW
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // No OpenGL context
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);    // Non-resizable for simplicity

  // Create window
  GLFWwindow* window = glfwCreateWindow(800, 600, "WebGPU Window", nullptr, nullptr);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return EXIT_FAILURE;
  }

  // Initialize WebGPU
  wgpu::InstanceDescriptor instanceDescriptor{};
  instanceDescriptor.capabilities.timedWaitAnyEnable = true;
  wgpu::Instance instance = wgpu::CreateInstance(&instanceDescriptor);
  if (instance == nullptr) {
    std::cerr << "Instance creation failed!\n";
    glfwTerminate();
    return EXIT_FAILURE;
  }

  // Create surface from GLFW window
  WGPUSurfaceDescriptor surfaceDesc = {};

#if defined(_WIN32)
  // Windows implementation
  WGPUSurfaceDescriptorFromWindowsHWND windowsDesc = {};
  windowsDesc.chain.sType = WGPUSType_SurfaceDescriptorFromWindowsHWND;
  windowsDesc.hwnd = glfwGetWin32Window(window);
  windowsDesc.hinstance = GetModuleHandle(nullptr);
  surfaceDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&windowsDesc);
#elif defined(__APPLE__)
  // macOS implementation - use our helper to create a Metal layer
  void* metalLayer = CreateMetalLayer(window);
  if (!metalLayer) {
    std::cerr << "Failed to create Metal layer" << std::endl;
    glfwTerminate();
    return EXIT_FAILURE;
  }
  
  // Create a Metal surface source
  WGPUSurfaceSourceMetalLayer metalLayerDesc = {};
  metalLayerDesc.chain.sType = WGPUSType_SurfaceSourceMetalLayer;
  metalLayerDesc.layer = metalLayer;
  
  // Create a surface descriptor that points to the metal layer descriptor
  surfaceDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&metalLayerDesc);
#elif defined(__linux__)
  // Linux implementation
  WGPUSurfaceDescriptorFromXlibWindow xlibDesc = {};
  xlibDesc.chain.sType = WGPUSType_SurfaceDescriptorFromXlibWindow;
  xlibDesc.display = glfwGetX11Display();
  xlibDesc.window = glfwGetX11Window(window);
  surfaceDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&xlibDesc);
#endif

  // Create surface using direct WebGPU C API for better cross-platform compatibility
  WGPUSurface rawSurface = wgpuInstanceCreateSurface(
      instance.Get(),
      &surfaceDesc);
  
  wgpu::Surface surface = wgpu::Surface::Acquire(rawSurface);
  if (!surface) {
    std::cerr << "Failed to create surface" << std::endl;
    glfwTerminate();
    return EXIT_FAILURE;
  }

  // Request adapter
  wgpu::RequestAdapterOptions options{};
  options.compatibleSurface = surface;
  wgpu::Adapter adapter;

  auto callback = [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, const char *message, void *userdata) {
    if (status != wgpu::RequestAdapterStatus::Success) {
      std::cerr << "Failed to get an adapter: " << message << std::endl;
      return;
    }
    *static_cast<wgpu::Adapter *>(userdata) = adapter;
  };

  auto callbackMode = wgpu::CallbackMode::WaitAnyOnly;
  void *userdata = &adapter;
  instance.WaitAny(instance.RequestAdapter(&options, callbackMode, callback, userdata), UINT64_MAX);
  if (adapter == nullptr) {
    std::cerr << "RequestAdapter failed!\n";
    glfwTerminate();
    return EXIT_FAILURE;
  }

  // Print adapter info
  wgpu::DawnAdapterPropertiesPowerPreference power_props{};
  wgpu::AdapterInfo info{};
  info.nextInChain = &power_props;
  adapter.GetInfo(&info);
  std::cout << "Using adapter: " << info.device << std::endl;

  // Create device
  wgpu::DeviceDescriptor deviceDesc{};
  wgpu::Device device = adapter.CreateDevice(&deviceDesc);
  if (!device) {
    std::cerr << "Failed to create device" << std::endl;
    glfwTerminate();
    return EXIT_FAILURE;
  }

  // Query surface capabilities to get the preferred format
  wgpu::SurfaceCapabilities capabilities{};
  surface.GetCapabilities(adapter, &capabilities);
  wgpu::TextureFormat swapChainFormat = wgpu::TextureFormat::BGRA8Unorm; // Default format
  
  if (capabilities.formatCount > 0 && capabilities.formats != nullptr) {
    swapChainFormat = capabilities.formats[0];
    std::cout << "Using surface format: " << static_cast<int>(swapChainFormat) << std::endl;
  }

  // Configure the surface
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  
  wgpu::SurfaceConfiguration surfaceConfig{};
  surfaceConfig.device = device;
  surfaceConfig.format = swapChainFormat;
  surfaceConfig.usage = wgpu::TextureUsage::RenderAttachment;
  surfaceConfig.width = static_cast<uint32_t>(width);
  surfaceConfig.height = static_cast<uint32_t>(height);
  surfaceConfig.presentMode = wgpu::PresentMode::Fifo;
  
  surface.Configure(&surfaceConfig);

  // Create a shader module for the vertex shader
  wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
  wgslDesc.code = vertexShaderCode;
  wgpu::ShaderModuleDescriptor vertexShaderDesc{};
  vertexShaderDesc.nextInChain = &wgslDesc;
  wgpu::ShaderModule vertexShaderModule = device.CreateShaderModule(&vertexShaderDesc);

  // Create a shader module for the fragment shader
  wgslDesc.code = fragmentShaderCode;
  wgpu::ShaderModuleDescriptor fragmentShaderDesc{};
  fragmentShaderDesc.nextInChain = &wgslDesc;
  wgpu::ShaderModule fragmentShaderModule = device.CreateShaderModule(&fragmentShaderDesc);

  // Create render pipeline layout
  wgpu::PipelineLayoutDescriptor layoutDesc{};
  wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&layoutDesc);

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
  colorTargetState.format = swapChainFormat;  // Use the swap chain format
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
  wgpu::RenderPipeline pipeline = device.CreateRenderPipeline(&pipelineDesc);

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    
    // Get the current texture from the surface
    wgpu::SurfaceTexture nextTexture{};
    surface.GetCurrentTexture(&nextTexture);
    
    if (nextTexture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
        nextTexture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
      std::cerr << "Failed to get current surface texture: " << static_cast<int>(nextTexture.status) << std::endl;
      break;
    }
    
    wgpu::TextureView currentTextureView = nextTexture.texture.CreateView();
    
    // Create command encoder
    wgpu::CommandEncoderDescriptor encoderDesc{};
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder(&encoderDesc);
    
    // Begin render pass
    wgpu::RenderPassColorAttachment colorAttachment{};
    colorAttachment.view = currentTextureView;
    colorAttachment.loadOp = wgpu::LoadOp::Clear;
    colorAttachment.storeOp = wgpu::StoreOp::Store;
    colorAttachment.clearValue = {0.0f, 0.0f, 0.0f, 1.0f}; // Black
    
    wgpu::RenderPassDescriptor renderPassDesc{};
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &colorAttachment;
    
    wgpu::RenderPassEncoder renderPass = encoder.BeginRenderPass(&renderPassDesc);
    
    // Draw the triangle
    renderPass.SetPipeline(pipeline);
    renderPass.Draw(3, 1, 0, 0);
    
    renderPass.End();
    
    // Submit commands
    wgpu::CommandBufferDescriptor cmdBufferDesc{};
    wgpu::CommandBuffer cmdBuffer = encoder.Finish(&cmdBufferDesc);
    wgpu::Queue queue = device.GetQueue();
    queue.Submit(1, &cmdBuffer);
    
    // Present the frame
    surface.Present();
  }
  
  // Cleanup
#if defined(__APPLE__)
  if (metalLayer) {
    ReleaseMetalLayer(metalLayer);
  }
#endif
  
  glfwDestroyWindow(window);
  glfwTerminate();
  
  return EXIT_SUCCESS;
}
