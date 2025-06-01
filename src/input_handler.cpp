#include "input_handler.h"
#include "fractal_settings.h"
#include "webgpu_setup.h"
#include <iostream>

// Global fractal settings instance (extern)
extern FractalSettings g_settings;

// Global WebGPU context for resize callbacks
static WebGPUContext* g_webgpu_context = nullptr;

// Mouse state variables - now using fractal settings struct

// Debug logging for mouse dragging
static int dragDebugCounter = 0;

// Old debug function - no longer used
// void logMouseDragDebug(...) { ... }

void initInputHandlers(GLFWwindow* window, WebGPUContext* context) {
    g_webgpu_context = context;
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetWindowSizeCallback(window, window_size_callback);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
        return;
        
    switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        //case GLFW_KEY_F:
            // Toggle exponential filter
            //g_settings.visualization.useExponentialFilter = !g_settings.visualization.useExponentialFilter;
            //std::cout << "Exponential Filter: " << (g_settings.visualization.useExponentialFilter ? "ON" : "OFF") << std::endl;
            //break;
        case GLFW_KEY_D:
            // Toggle distance visualization
            if (g_settings.visualization.mode == DISTANCE_FIELD) {
                g_settings.visualization.mode = JULIA_NORMAL;
            } else {
                g_settings.visualization.mode = DISTANCE_FIELD;
            }
            std::cout << "Visualization Mode: " << getVisualizationModeName(g_settings.visualization.mode) << std::endl;
            break;
        case GLFW_KEY_M:
            // Toggle mask visualization
            if (g_settings.visualization.mode == ITERATION_MASK) {
                g_settings.visualization.mode = JULIA_NORMAL;
            } else {
                g_settings.visualization.mode = ITERATION_MASK;
            }
            std::cout << "Visualization Mode: " << getVisualizationModeName(g_settings.visualization.mode) << std::endl;
            break;
        case GLFW_KEY_SPACE:
            // Toggle Julia circular motion
            g_settings.julia.motionEnabled = !g_settings.julia.motionEnabled;
            std::cout << "Julia motion: " << (g_settings.julia.motionEnabled ? "ON" : "OFF") << std::endl;
            break;
        case GLFW_KEY_T:
            // For now, just toggle between normal modes - will add Mandelbrot later
            std::cout << "Fractal type toggle - functionality to be implemented" << std::endl;
            break;
        case GLFW_KEY_C:
            // Cycle through coloring modes: COLOR_BANDS -> COLOR_CONTINUOUS -> GRAYSCALE -> THRESHOLD -> COLOR_BANDS
            if (g_settings.visualization.coloringMode == COLOR_BANDS) {
                g_settings.visualization.coloringMode = COLOR_CONTINUOUS;
            } else if (g_settings.visualization.coloringMode == COLOR_CONTINUOUS) {
                g_settings.visualization.coloringMode = GRAYSCALE;
            } else if (g_settings.visualization.coloringMode == GRAYSCALE) {
                g_settings.visualization.coloringMode = THRESHOLD;
            } else {
                g_settings.visualization.coloringMode = COLOR_BANDS;
            }
            std::cout << "Coloring Mode: " << getColoringModeName(g_settings.visualization.coloringMode) << std::endl;
            break;
            
        // === NEW VIEWPORT MOVEMENT CONTROLS ===
        case GLFW_KEY_LEFT:
            // Move viewport left
            {
                double panAmount = g_settings.viewport.viewWidth * 0.1; // Pan by 10% of current view width
                g_settings.viewport.viewX -= panAmount;
                std::cout << "Viewport moved left to X=" << g_settings.viewport.viewX << std::endl;
            }
            break;
        case GLFW_KEY_RIGHT:
            // Move viewport right
            {
                double panAmount = g_settings.viewport.viewWidth * 0.1;
                g_settings.viewport.viewX += panAmount;
                std::cout << "Viewport moved right to X=" << g_settings.viewport.viewX << std::endl;
            }
            break;
        case GLFW_KEY_UP:
            // Move viewport up
            {
                double panAmount = g_settings.viewport.viewHeight * 0.1;
                g_settings.viewport.viewY += panAmount;  // flipped y axis
                std::cout << "Viewport moved up to Y=" << g_settings.viewport.viewY << std::endl;
            }
            break;
        case GLFW_KEY_DOWN:
            // Move viewport down
            {
                double panAmount = g_settings.viewport.viewHeight * 0.1;
                g_settings.viewport.viewY -= panAmount;  // flipped y axis
                std::cout << "Viewport moved down to Y=" << g_settings.viewport.viewY << std::endl;
            }
            break;
            
        // === NEW ZOOM CONTROLS ===
        case GLFW_KEY_J:
            // Zoom in
            {
                double zoomFactor = 0.9; // Zoom in by 10%
                double centerX = g_settings.viewport.viewX + g_settings.viewport.viewWidth / 2.0;
                double centerY = g_settings.viewport.viewY + g_settings.viewport.viewHeight / 2.0;
                
                g_settings.viewport.viewWidth *= zoomFactor;
                g_settings.viewport.viewHeight *= zoomFactor;
                
                // Adjust position to keep center point fixed
                g_settings.viewport.viewX = centerX - g_settings.viewport.viewWidth / 2.0;
                g_settings.viewport.viewY = centerY - g_settings.viewport.viewHeight / 2.0;
                
                std::cout << "Zoomed in - Width=" << g_settings.viewport.viewWidth 
                          << ", Height=" << g_settings.viewport.viewHeight << std::endl;
            }
            break;
        case GLFW_KEY_K:
            // Zoom out
            {
                double zoomFactor = 1.1; // Zoom out by 10%
                double centerX = g_settings.viewport.viewX + g_settings.viewport.viewWidth / 2.0;
                double centerY = g_settings.viewport.viewY + g_settings.viewport.viewHeight / 2.0;
                
                g_settings.viewport.viewWidth *= zoomFactor;
                g_settings.viewport.viewHeight *= zoomFactor;
                
                // Adjust position to keep center point fixed
                g_settings.viewport.viewX = centerX - g_settings.viewport.viewWidth / 2.0;
                g_settings.viewport.viewY = centerY - g_settings.viewport.viewHeight / 2.0;
                
                std::cout << "Zoomed out - Width=" << g_settings.viewport.viewWidth 
                          << ", Height=" << g_settings.viewport.viewHeight << std::endl;
            }
            break;
            
        // === ITERATION CONTROL ===
        case GLFW_KEY_I:
            // Increase max iterations
            {
                g_settings.visualization.maxIterations += 1;
                // Set a reasonable upper limit
                if (g_settings.visualization.maxIterations > 5000) {
                    g_settings.visualization.maxIterations = 5000;
                }
                std::cout << "Max iterations increased to " << g_settings.visualization.maxIterations << std::endl;
            }
            break;
        case GLFW_KEY_O:
            // Decrease max iterations
            {
                g_settings.visualization.maxIterations -= 1;
                // Set a reasonable lower limit
                if (g_settings.visualization.maxIterations < 1) {
                    g_settings.visualization.maxIterations = 1;
                }
                std::cout << "Max iterations decreased to " << g_settings.visualization.maxIterations << std::endl;
            }
            break;
        case GLFW_KEY_R:
            // Reset viewport to default values
            g_settings.viewport.viewX = -2.0;
            g_settings.viewport.viewY = -2.0;
            g_settings.viewport.viewWidth = 4.0;
            g_settings.viewport.viewHeight = 4.0;
            g_settings.viewport.offsetX = 0.0;
            g_settings.viewport.offsetY = 0.0;
            std::cout << "Viewport reset to default values" << std::endl;
            break;
        
        // === ANGLE VELOCITY CONTROL ===
        case GLFW_KEY_Q:
            // Decrease angle velocity
            g_settings.julia.angleVelocity *= 0.9;
            std::cout << "Angle velocity decreased to " << g_settings.julia.angleVelocity << std::endl;
            break;
        case GLFW_KEY_E:
            // Increase angle velocity
            g_settings.julia.angleVelocity *= 1.1;
            std::cout << "Angle velocity increased to " << g_settings.julia.angleVelocity << std::endl;
            break;
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            g_settings.interaction.leftButtonHeld = true;
            g_settings.interaction.mouseDragging = false;
            
            // Store drag start point in screen coordinates
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            g_settings.interaction.dragStartScreenX = xpos;
            g_settings.interaction.dragStartScreenY = ypos;
            
            // Store initial viewport position when drag starts
            g_settings.interaction.dragStartViewX = g_settings.viewport.viewX;
            g_settings.interaction.dragStartViewY = g_settings.viewport.viewY;
        } else if (action == GLFW_RELEASE) {
            g_settings.interaction.leftButtonHeld = false;
            g_settings.interaction.mouseDragging = false;
        }
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (g_settings.interaction.leftButtonHeld) {
        if (!g_settings.interaction.mouseDragging) {
            // // Reset debug counter when we begin dragging
            // dragDebugCounter = 0;
            
            // std::cout << "Starting drag at screen: (" << xpos << ", " << ypos << ")" << std::endl;
            // std::cout << "Initial viewport: viewX=" << g_settings.viewport.viewX << ", viewY=" << g_settings.viewport.viewY << std::endl;
            // std::cout << "Viewport dimensions: " << g_settings.viewport.viewWidth << "x" << g_settings.viewport.viewHeight << std::endl;
            // std::cout << "Screen dimensions: logical=" << g_settings.viewport.logicalWidth << "x" << g_settings.viewport.logicalHeight << std::endl << std::endl;

            g_settings.interaction.mouseDragging = true;
        }
        
        // Calculate screen coordinate offset from drag start
        double screenDx = xpos - g_settings.interaction.dragStartScreenX;
        double screenDy = ypos - g_settings.interaction.dragStartScreenY;
        
        // Convert screen offset to fractal coordinate offset
        double fractalDx = (screenDx / g_settings.viewport.logicalWidth) * g_settings.viewport.viewWidth;
        double fractalDy = -(screenDy / g_settings.viewport.logicalHeight) * g_settings.viewport.viewHeight; // Note: negative because screen Y increases down, fractal Y increases up
        
        // // Debug logging for the first 10 times
        // if (dragDebugCounter < 10) {
        //     std::cout << "=== DRAG DEBUG #" << (dragDebugCounter + 1) << " ===" << std::endl;
        //     std::cout << "Screen offset: dx=" << screenDx << ", dy=" << screenDy << std::endl;
        //     std::cout << "Fractal offset: dx=" << fractalDx << ", dy=" << fractalDy << std::endl;
        //     std::cout << "Current screen pos: (" << xpos << ", " << ypos << ")" << std::endl;
        //     std::cout << "Drag start screen: (" << g_settings.interaction.dragStartScreenX << ", " << g_settings.interaction.dragStartScreenY << ")" << std::endl;
        //     std::cout << "=========================" << std::endl;
        //     dragDebugCounter++;
        // }

        // Apply total offset from drag start to initial viewport position
        g_settings.viewport.viewX = g_settings.interaction.dragStartViewX - fractalDx;
        g_settings.viewport.viewY = g_settings.interaction.dragStartViewY - fractalDy;
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    // Get current cursor position
    double mouseX, mouseY;
    glfwGetCursorPos(window, &mouseX, &mouseY);
    
    // Convert mouse position to fractal coordinates (point to zoom around)
    double fractalMouseX = g_settings.viewport.viewX + (mouseX / g_settings.viewport.logicalWidth) * g_settings.viewport.viewWidth;
    double fractalMouseY = g_settings.viewport.viewY + ((g_settings.viewport.logicalHeight - mouseY) / g_settings.viewport.logicalHeight) * g_settings.viewport.viewHeight;
    
    // Calculate zoom factor (scroll up zooms in, scroll down zooms out)
    double zoomFactor = 1.0 - (yoffset * 0.1);
    
    // Apply zoom to viewport dimensions
    g_settings.viewport.viewWidth *= zoomFactor;
    g_settings.viewport.viewHeight *= zoomFactor;
    
    // Adjust viewport position to keep the mouse cursor point fixed
    g_settings.viewport.viewX = fractalMouseX - (mouseX / g_settings.viewport.logicalWidth) * g_settings.viewport.viewWidth;
    g_settings.viewport.viewY = fractalMouseY - ((g_settings.viewport.logicalHeight - mouseY) / g_settings.viewport.logicalHeight) * g_settings.viewport.viewHeight;
    
    std::cout << "Zoom factor: " << zoomFactor << " (centered at fractal coords: " << fractalMouseX << ", " << fractalMouseY << ")" << std::endl;
}

// Mouse state getters
bool isLeftButtonHeld() {
    return g_settings.interaction.leftButtonHeld;
}

bool isMouseDragging() {
    return g_settings.interaction.mouseDragging;
}

double getMouseDragOffsetX() {
    return g_settings.interaction.mouseDragOffsetX;
}

double getMouseDragOffsetY() {
    return g_settings.interaction.mouseDragOffsetY;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    std::cout << "Framebuffer resize: " << width << "x" << height << std::endl;
    
    // Avoid division by zero and ensure minimum size
    if (width <= 0 || height <= 0) {
        std::cout << "Invalid framebuffer size, ignoring resize" << std::endl;
        return;
    }
    
    // Update framebuffer dimensions in settings
    g_settings.viewport.width = width;
    g_settings.viewport.height = height;
    
    // Maintain aspect ratio for fractal viewport
    double aspectRatio = (double)width / height;
    g_settings.viewport.viewHeight = g_settings.viewport.viewWidth / aspectRatio;
    
    // Reconfigure WebGPU surface
    if (g_webgpu_context) {
        reconfigureSurface(*g_webgpu_context, width, height);
    }
}

void window_size_callback(GLFWwindow* window, int width, int height) {
    std::cout << "Window resize: " << width << "x" << height << std::endl;
    
    // Avoid invalid window sizes
    if (width <= 0 || height <= 0) {
        std::cout << "Invalid window size, ignoring resize" << std::endl;
        return;
    }
    
    // Update logical window dimensions in settings
    g_settings.viewport.logicalWidth = width;
    g_settings.viewport.logicalHeight = height;
} 