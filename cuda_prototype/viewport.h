#ifndef VIEWPORT_H
#define VIEWPORT_H

#include <GLFW/glfw3.h>

#ifdef __cplusplus
extern "C" {
#endif

// Constants for viewport manipulation
extern const double RADIUS_FACTOR;     // Radius as a fraction of viewport size for circular motion
extern const double SMALL_ANGULAR_VELOCITY;  // Angular velocity for circular motion
extern const double PAN_FACTOR;        // Factor for panning the viewport
extern const double ZOOM_FACTOR;       // Factor for zooming the viewport

// Viewport state
extern double viewX;          // Viewport X position
extern double viewY;          // Viewport Y position
extern double viewWidth;      // Viewport width
extern double viewHeight;     // Viewport height
extern double offsetX;        // Circular motion X offset
extern double offsetY;        // Circular motion Y offset

// Mouse control variables
extern bool mouseDragging;
extern double lastMouseX, lastMouseY;
extern bool leftButtonHeld;

// Current mouse position
extern double currentMouseX, currentMouseY;

// Initialize the viewport with default values
void initViewport(double width, double height);

// Reset the viewport to default position and size
void resetViewport();

// Update the viewport for circular motion
void updateViewportOffset(unsigned int frameCount, bool leftButtonHeld);

// Get the actual viewport coordinates after applying offsets
void getActualViewport(double* outViewX, double* outViewY);

// Callbacks for keyboard and mouse input
void viewport_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void viewport_mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void viewport_cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void viewport_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

#ifdef __cplusplus
}
#endif

#endif // VIEWPORT_H 