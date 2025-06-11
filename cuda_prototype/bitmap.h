#ifndef BITMAP_H
#define BITMAP_H

#include <string>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Function to render a string using the bitmap font
void renderBitmapText(const std::string& text, float x, float y, float scale = 1.0f);

// Function to render debug information overlay
void renderDebugInfo(GLFWwindow* window, double viewX, double viewY, double viewWidth, double viewHeight, 
                    bool mouseDragging, bool leftButtonHeld, double offsetX, double offsetY,
                    const std::string& visualizationModeName, const std::string& fractalTypeName,
                    bool useJuliaFractal, double juliaRadius, double juliaAngle);

#endif // BITMAP_H 