#pragma once

struct GLFWwindow;

#ifdef __cplusplus
extern "C" {
#endif

// Creates a CAMetalLayer on the GLFW window and returns it
void* CreateMetalLayer(GLFWwindow* window);

// Releases the CAMetalLayer when done
void ReleaseMetalLayer(void* layer);

#ifdef __cplusplus
}
#endif 