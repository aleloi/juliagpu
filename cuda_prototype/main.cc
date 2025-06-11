// main.cc
// Regular C++ file compiled with gcc/g++.
// It uses GLFW and OpenGL for windowing and rendering.
// It calls the CUDA function defined in cuda_kernels.cu.

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <algorithm>
#include <deque>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>
#include "cuda_kernels.h" // Declaration for launchGradientKernel()
#include "bitmap.h" // For bitmap font rendering and debug overlay
#include "viewport.h" // For viewport management

// Target frame rate
#define FPS 30

// Julia set parameters
double juliaAngle = 0.0;  // Angle for circular motion
const double JULIA_RADIUS = 0.7885;  // Radius for circular motion
double julia_angle_velocity = 0.001;  // Angle increment per frame
bool juliaMotionEnabled = true;  // Flag to enable/disable Julia motion

// Flag to control the exponential filter
bool useExponentialFilter = false;

// Visualization modes enum
enum VisualizationMode {
    MANDELBROT_NORMAL,
    DISTANCE_FIELD,
    ITERATION_MASK
};

// Current visualization mode
VisualizationMode visualizationMode = MANDELBROT_NORMAL;

// Helper function to get visualization mode name
std::string getVisualizationModeName() {
    switch (visualizationMode) {
        case MANDELBROT_NORMAL: return "Normal";
        case DISTANCE_FIELD: return "Distance Field";
        case ITERATION_MASK: return "Iteration Mask";
        default: return "Unknown";
    }
}

// Helper function to get fractal type name
std::string getFractalTypeName() {
    return getUseJuliaFractal() ? "Julia" : "Mandelbrot";
}

// Window dimensions - will be set to monitor resolution
int WIDTH;
int HEIGHT;

// Keyboard callback function
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // Call the viewport key callback first to handle viewport-related keys
    viewport_key_callback(window, key, scancode, action, mods);
    
    // Handle non-viewport related keys
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
        return;
        
    switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        case GLFW_KEY_F:
            // Toggle exponential filter
            useExponentialFilter = !useExponentialFilter;
            std::cout << "Exponential Filter: " << (useExponentialFilter ? "ON" : "OFF") << std::endl;
            break;
        case GLFW_KEY_D:
            // Toggle distance visualization
            if (visualizationMode == DISTANCE_FIELD) {
                visualizationMode = MANDELBROT_NORMAL;
            } else {
                visualizationMode = DISTANCE_FIELD;
            }
            std::cout << "Visualization Mode: " << getVisualizationModeName() << std::endl;
            break;
        case GLFW_KEY_M:
            // Toggle mask visualization
            if (visualizationMode == ITERATION_MASK) {
                visualizationMode = MANDELBROT_NORMAL;
            } else {
                visualizationMode = ITERATION_MASK;
            }
            std::cout << "Visualization Mode: " << getVisualizationModeName() << std::endl;
            break;
        case GLFW_KEY_T:
            // Toggle between Julia and Mandelbrot fractals
            setUseJuliaFractal(!getUseJuliaFractal());
            std::cout << "Fractal type: " << getFractalTypeName() << std::endl;
            break;
        case GLFW_KEY_PAGE_UP:
            // Increase high iteration threshold
            increaseHighIterationThreshold(10);
            std::cout << "High Iteration Threshold: " << getHighIterationThreshold() << std::endl;
            break;
        case GLFW_KEY_PAGE_DOWN:
            // Decrease high iteration threshold
            decreaseHighIterationThreshold(10);
            std::cout << "High Iteration Threshold: " << getHighIterationThreshold() << std::endl;
            break;
        case GLFW_KEY_1:
            // Decrease Julia angle
            juliaAngle -= 100*julia_angle_velocity;
            std::cout << "Julia angle: " << juliaAngle << std::endl;
            break;
        case GLFW_KEY_2:
            // Increase Julia radius
            juliaAngle += 100*julia_angle_velocity;
            std::cout << "Julia angle: " << juliaAngle << std::endl;
            break;
        case GLFW_KEY_3:
            // Decrease Julia angle increment (slower motion)
            if (julia_angle_velocity > 0.00001) {
                julia_angle_velocity = julia_angle_velocity / 2.0;
                std::cout << "Julia motion speed: " << julia_angle_velocity << std::endl;
            }
            break;
        case GLFW_KEY_4:
            // Increase Julia angle increment (faster motion)
            julia_angle_velocity = julia_angle_velocity * 2.0;
            std::cout << "Julia motion speed: " << julia_angle_velocity << std::endl;
            break;
        case GLFW_KEY_SPACE:
            // Toggle Julia circular motion
            juliaMotionEnabled = !juliaMotionEnabled;
            std::cout << "Julia motion: " << (juliaMotionEnabled ? "ON" : "OFF") << std::endl;
            break;
    }
}

int new_new_gl_stuff() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "GLFW init failed.\n";
        return -1;
    }

    // Explicitly use legacy-compatible OpenGL (2.1)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    // Get primary monitor and its resolution
    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(primaryMonitor);
    
    // Set resolution to monitor's full resolution
    WIDTH = mode->width;
    HEIGHT = mode->height;
    
    // Initialize viewport with correct aspect ratio
    initViewport(WIDTH, HEIGHT);

    // Create fullscreen window
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "CUDA-OpenGL Interop", primaryMonitor, NULL);
    if (!window) {
        std::cerr << "Window creation failed.\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    
    // Set callbacks
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, viewport_mouse_button_callback);
    glfwSetCursorPosCallback(window, viewport_cursor_position_callback);
    glfwSetScrollCallback(window, viewport_scroll_callback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW init failed.\n";
        glfwTerminate();
        return -1;
    }

    // Explicitly set viewport and clear color (black background)
    glViewport(0, 0, WIDTH, HEIGHT);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Create Pixel Buffer Object (PBO)
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // CUDA Graphics Resource
    cudaGraphicsResource* cuda_pbo;
    cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard);
    
    // Initialize the exponential filter buffer
    if (!initExponentialFilter(WIDTH, HEIGHT)) {
        std::cerr << "Failed to initialize exponential filter buffer.\n";
        return -1;
    }

    // Texture setup
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    unsigned int frameCount = 0;  // Frame counter for circular motion
    
    // Simple performance tracking
    double totalCudaTime = 0.0;
    double maxCudaTime = 0.0;
    double totalRenderTime = 0.0;
    double maxRenderTime = 0.0;
    auto lastLogTime = std::chrono::high_resolution_clock::now();

    // Print controls info
    std::cout << "Running in fullscreen mode (" << WIDTH << "x" << HEIGHT << ")" << std::endl;
    std::cout << "Controls:\n";
    std::cout << "  Arrow keys: Pan view\n";
    std::cout << "  J: Zoom in\n";
    std::cout << "  K: Zoom out\n";
    std::cout << "  Mouse drag: Pan view\n";
    std::cout << "  Mouse wheel: Zoom in/out\n";
    std::cout << "  Left mouse button hold: Pause circular motion\n";
    std::cout << "  R: Reset view\n";
    std::cout << "  F: Toggle exponential filter (currently " << (useExponentialFilter ? "ON" : "OFF") << ")\n";
    std::cout << "  D: Toggle distance field visualization\n";
    std::cout << "  M: Toggle iteration mask visualization\n";
    std::cout << "  T: Toggle between Mandelbrot and Julia fractals (currently " << getFractalTypeName() << ")\n";
    std::cout << "  1/2: Decrease/Increase Julia radius (currently " << JULIA_RADIUS << ")\n";
    std::cout << "  3/4: Decrease/Increase Julia motion speed\n";
    std::cout << "  Current visualization mode: " << getVisualizationModeName() << std::endl;
    std::cout << "  PAGE UP: Increase high iteration threshold (currently " << getHighIterationThreshold() << ")\n";
    std::cout << "  PAGE DOWN: Decrease high iteration threshold (currently " << getHighIterationThreshold() << ")\n";
    std::cout << "  SPACE: Toggle Julia circular motion (currently " << (juliaMotionEnabled ? "ON" : "OFF") << ")\n";
    std::cout << "  ESC: Quit\n";

    // Main rendering loop
    while (!glfwWindowShouldClose(window)) {
        // CUDA timing
        auto cudaStart = std::chrono::high_resolution_clock::now();
        
        // Update viewport offset for circular motion
        updateViewportOffset(frameCount, leftButtonHeld);
        
        // Get actual viewport coordinates with offsets applied
        double actualViewX, actualViewY;
        getActualViewport(&actualViewX, &actualViewY);
        
        // Update Julia angle for circular motion
        if (juliaMotionEnabled && !leftButtonHeld) {
            juliaAngle += julia_angle_velocity;
            if (juliaAngle >= 2.0 * M_PI) {
                juliaAngle -= 2.0 * M_PI;  // Keep angle in [0, 2π) range
            }
        }
        
        // CUDA: compute data
        unsigned char* devPtr;
        size_t numBytes;
        cudaGraphicsMapResources(1, &cuda_pbo, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &numBytes, cuda_pbo);

        // Choose visualization mode and fractal type
        if (visualizationMode == DISTANCE_FIELD) {
            if (getUseJuliaFractal()) {
                // Calculate julia constants for visualization
                double julia_cx = JULIA_RADIUS * cos(juliaAngle);
                double julia_cy = JULIA_RADIUS * sin(juliaAngle);
                
                launchDistanceVisualizationKernel(devPtr, WIDTH, HEIGHT, 
                                                actualViewX, actualViewY, viewWidth, viewHeight,
                                                julia_cx, julia_cy);
            } else {
                launchDistanceVisualizationKernel(devPtr, WIDTH, HEIGHT, 
                                                actualViewX, actualViewY, viewWidth, viewHeight);
            }
        } else if (visualizationMode == ITERATION_MASK) {
            if (getUseJuliaFractal()) {
                // Calculate julia constants for visualization
                double julia_cx = JULIA_RADIUS * cos(juliaAngle);
                double julia_cy = JULIA_RADIUS * sin(juliaAngle);
                
                launchMaskVisualizationKernel(devPtr, WIDTH, HEIGHT,
                                            actualViewX, actualViewY, viewWidth, viewHeight,
                                            julia_cx, julia_cy);
            } else {
                launchMaskVisualizationKernel(devPtr, WIDTH, HEIGHT,
                                            actualViewX, actualViewY, viewWidth, viewHeight);
            }
        } else {
            // Normal mode - choose between Mandelbrot and Julia
            if (getUseJuliaFractal()) {
                // Calculate julia constants here directly from radius and angle
                double julia_cx = JULIA_RADIUS * cos(juliaAngle);
                double julia_cy = JULIA_RADIUS * sin(juliaAngle);
                
                launchJuliaKernel(devPtr, WIDTH, HEIGHT,
                                actualViewX, actualViewY, viewWidth, viewHeight,
                                julia_cx, julia_cy,
                                useExponentialFilter);
            } else {
                launchMandelbrotKernel(devPtr, WIDTH, HEIGHT, 
                                    actualViewX, actualViewY, viewWidth, viewHeight, 
                                    useExponentialFilter);
            }
        }

        cudaGraphicsUnmapResources(1, &cuda_pbo, 0);
        
        auto cudaEnd = std::chrono::high_resolution_clock::now();
        double cudaTime = std::chrono::duration<double, std::milli>(cudaEnd - cudaStart).count();
        totalCudaTime += cudaTime;
        maxCudaTime = std::max(maxCudaTime, cudaTime);
        
        // OpenGL timing
        auto renderStart = std::chrono::high_resolution_clock::now();

        // OpenGL: update texture from CUDA buffer (PBO)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // OpenGL: clear and draw quad
        glClear(GL_COLOR_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, WIDTH, HEIGHT, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glEnable(GL_TEXTURE_2D);
        glColor3f(1.0f, 1.0f, 1.0f);  // Ensure color is white
        glBindTexture(GL_TEXTURE_2D, texture);

        glBegin(GL_QUADS);
            glTexCoord2f(0, 0); glVertex2f(0, 0);
            glTexCoord2f(1, 0); glVertex2f(WIDTH, 0);
            glTexCoord2f(1, 1); glVertex2f(WIDTH, HEIGHT);
            glTexCoord2f(0, 1); glVertex2f(0, HEIGHT);
        glEnd();

        glDisable(GL_TEXTURE_2D);

        // Get actual viewport for debug info
        double actualViewX_debug, actualViewY_debug;
        getActualViewport(&actualViewX_debug, &actualViewY_debug);
        
        // Render debug info overlay using the bitmap font renderer
        renderDebugInfo(window, actualViewX_debug, actualViewY_debug, viewWidth, viewHeight, 
                       mouseDragging, leftButtonHeld, offsetX, offsetY,
                       getVisualizationModeName(), getFractalTypeName(),
                       getUseJuliaFractal(), JULIA_RADIUS, juliaAngle);

        glfwSwapBuffers(window);
        glfwPollEvents();
        
        auto renderEnd = std::chrono::high_resolution_clock::now();
        double renderTime = std::chrono::duration<double, std::milli>(renderEnd - renderStart).count();
        totalRenderTime += renderTime;
        maxRenderTime = std::max(maxRenderTime, renderTime);
        
        // Increment frame counter for circular motion
        frameCount++;
        
        // Check if 10 seconds have passed
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastLogTime).count() >= 10) {
            // Print stats
            int statsFrameCount = frameCount % 10000; // To avoid division by zero if frameCount was reset
            std::cout << "CUDA avg: " << (totalCudaTime / statsFrameCount) << "ms, max: " << maxCudaTime << "ms" << std::endl;
            std::cout << "Render avg: " << (totalRenderTime / statsFrameCount) << "ms, max: " << maxRenderTime << "ms" << std::endl;
            
            // Get actual viewport for logging
            double actualViewX_log, actualViewY_log;
            getActualViewport(&actualViewX_log, &actualViewY_log);
            
            std::cout << "Current view: X=" << viewX << ", Y=" << viewY << ", W=" << viewWidth << ", H=" << viewHeight << std::endl;
            std::cout << "Fullscreen resolution: " << WIDTH << "x" << HEIGHT << std::endl;
            std::cout << "Exponential Filter: " << (useExponentialFilter ? "ON" : "OFF") << std::endl;
            std::cout << "Fractal type: " << getFractalTypeName() << std::endl;
            if (getUseJuliaFractal()) {
                // Calculate julia constants directly from radius and angle for display
                double julia_cx = JULIA_RADIUS * cos(juliaAngle);
                double julia_cy = JULIA_RADIUS * sin(juliaAngle);
                std::cout << "Julia constants: c = " << julia_cx << " + " << julia_cy << "i" << std::endl;
                std::cout << "Julia motion: " << (juliaMotionEnabled ? "ON" : "OFF") << ", Radius: " << JULIA_RADIUS << std::endl;
            }
            std::cout << "Visualization Mode: " << getVisualizationModeName() << std::endl;
            std::cout << "High Iteration Threshold: " << getHighIterationThreshold() << std::endl;
            
            // Reset tracking
            totalCudaTime = 0.0;
            maxCudaTime = 0.0;
            totalRenderTime = 0.0;
            maxRenderTime = 0.0;
            lastLogTime = now;
        }


        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / FPS));
    }

    // Cleanup
    cleanupExponentialFilter();  // Clean up the exponential filter buffer
    cudaGraphicsUnregisterResource(cuda_pbo);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}


int main() {
  return new_new_gl_stuff();
}
