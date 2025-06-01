#pragma once

struct GLFWwindow;

#ifdef __cplusplus
extern "C" {
#endif

// Creates a CAMetalLayer on the GLFW window and returns it
void* CreateMetalLayer(GLFWwindow* window);

// Releases the CAMetalLayer when done
void ReleaseMetalLayer(void* layer);

// Updates the Metal layer size to match the window
void UpdateMetalLayerSize(void* layer, GLFWwindow* window);

#ifdef __cplusplus
}
#endif 