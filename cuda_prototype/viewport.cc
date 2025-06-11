#include "viewport.h"
#include <iostream>
#include <cmath>

// Constants for viewport manipulation
const double RADIUS_FACTOR = 0.1f;       // Radius as a fraction of viewport size
const double SMALL_ANGULAR_VELOCITY = 0.00001f;
const double PAN_FACTOR = 0.01f;
const double ZOOM_FACTOR = 0.95f;

// Viewport state
double viewX = -2.0f;
double viewY = -1.12f;
double viewWidth = 2.47f;
double viewHeight = 2.24f;
double offsetX = 0.0;
double offsetY = 0.0;

// Mouse control variables
bool mouseDragging = false;
double lastMouseX = 0.0, lastMouseY = 0.0;
bool leftButtonHeld = false;

// Current mouse position
double currentMouseX = 0.0, currentMouseY = 0.0;

// Initialize the viewport with default values and adjust for aspect ratio
void initViewport(double width, double height) {
    viewX = -2.0f;
    viewY = -1.12f;
    viewWidth = 2.47f;
    
    // Adjust the viewport height to maintain aspect ratio
    double aspectRatio = width / height;
    viewHeight = viewWidth / aspectRatio;
    
    offsetX = 0.0;
    offsetY = 0.0;
    
    mouseDragging = false;
    leftButtonHeld = false;
}

// Reset the viewport to default position and size
void resetViewport() {
    viewX = -2.0f;
    viewY = -1.12f;
    viewWidth = 2.47f;
    viewHeight = 2.24f;
    offsetX = 0.0;
    offsetY = 0.0;
}

// Update the viewport for circular motion
void updateViewportOffset(unsigned int frameCount, bool leftButtonHeld) {
    // Calculate circular motion offset
    double angle = 2.0f * M_PI * SMALL_ANGULAR_VELOCITY * frameCount;
    
    if (!leftButtonHeld) {
        offsetX = RADIUS_FACTOR * viewWidth * cos(angle);
        offsetY = RADIUS_FACTOR * viewHeight * sin(angle);
    }
}

// Get the actual viewport coordinates after applying offsets
void getActualViewport(double* outViewX, double* outViewY) {
    *outViewX = viewX + offsetX;
    *outViewY = viewY + offsetY;
}

// Keyboard callback for viewport manipulation
void viewport_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
        return;
        
    // Calculate the current center point
    double centerX = viewX + viewWidth / 2.0f;
    double centerY = viewY + viewHeight / 2.0f;
    
    switch (key) {
        case GLFW_KEY_LEFT:
            // Pan left
            viewX -= viewWidth * PAN_FACTOR;
            break;
        case GLFW_KEY_RIGHT:
            // Pan right
            viewX += viewWidth * PAN_FACTOR;
            break;
        case GLFW_KEY_UP:
            // Pan up
            viewY -= viewHeight * PAN_FACTOR;
            break;
        case GLFW_KEY_DOWN:
            // Pan down
            viewY += viewHeight * PAN_FACTOR;
            break;
        case GLFW_KEY_J:
            // Zoom in
            viewWidth *= ZOOM_FACTOR;
            viewHeight *= ZOOM_FACTOR;
            // Recenter the view around the same point
            viewX = centerX - viewWidth / 2.0f;
            viewY = centerY - viewHeight / 2.0f;
            break;
        case GLFW_KEY_K:
            // Zoom out
            viewWidth /= ZOOM_FACTOR;
            viewHeight /= ZOOM_FACTOR;
            // Recenter the view around the same point
            viewX = centerX - viewWidth / 2.0f;
            viewY = centerY - viewHeight / 2.0f;
            break;
        case GLFW_KEY_R:
            // Reset view
            resetViewport();
            break;
        default:
            // Other keys are handled elsewhere
            break;
    }
}

// Mouse button callback for viewport manipulation
void viewport_mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
            leftButtonHeld = true;
            mouseDragging = true;
        } else if (action == GLFW_RELEASE) {
            leftButtonHeld = false;
            mouseDragging = false;
        }
    }
}

// Cursor position callback for viewport manipulation
void viewport_cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    // Update current mouse position
    currentMouseX = xpos;
    currentMouseY = ypos;
    
    if (mouseDragging) {
        // Calculate movement in screen coordinates
        double deltaX = xpos - lastMouseX;
        double deltaY = ypos - lastMouseY;
        
        // Get window dimensions
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        
        // Convert to world coordinates based on viewport
        double worldDeltaX = -deltaX * viewWidth / width;
        double worldDeltaY = -deltaY * viewHeight / height;
        
        // Move the view
        viewX += worldDeltaX;
        viewY += worldDeltaY;
        
        // Update last position
        lastMouseX = xpos;
        lastMouseY = ypos;
    }
}

// Scroll callback for viewport manipulation
void viewport_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    // Calculate the center point
    double centerX = viewX + viewWidth / 2.0f;
    double centerY = viewY + viewHeight / 2.0f;
    
    // Zoom factor based on scroll direction
    double zoomFactor = (yoffset > 0) ? ZOOM_FACTOR : 1.0/ZOOM_FACTOR;
    
    // Apply zoom
    viewWidth *= zoomFactor;
    viewHeight *= zoomFactor;
    
    // Recenter the view around the same point
    viewX = centerX - viewWidth / 2.0f;
    viewY = centerY - viewHeight / 2.0f;
} 