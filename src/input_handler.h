#pragma once
#include <GLFW/glfw3.h>

struct WebGPUContext; // Forward declaration

// Initialize input handlers
void initInputHandlers(GLFWwindow* window, WebGPUContext* context);

// Callback functions
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void window_size_callback(GLFWwindow* window, int width, int height);

// Mouse state getters
bool isLeftButtonHeld();
bool isMouseDragging();
double getMouseDragOffsetX();
double getMouseDragOffsetY(); 