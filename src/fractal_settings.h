#pragma once

#include <string>

// Visualization mode enumeration
enum VisualizationMode {
    JULIA_NORMAL,
    DISTANCE_FIELD,
    ITERATION_MASK
};

// Coloring mode enumeration
enum ColoringMode {
    COLOR_BANDS,        // 0 - Discrete color bands (old COLORED)
    COLOR_CONTINUOUS,   // 1 - Smooth continuous colors
    GRAYSCALE,          // 2 - Grayscale 
    THRESHOLD           // 3 - Binary threshold
};

// Viewport settings sub-struct
struct ViewportSettings {
    double viewX = -2.0;
    double viewY = -2.0;
    double viewWidth = 4.0;
    double viewHeight = 4.0;
    double offsetX = 0.0;
    double offsetY = 0.0;
    int width = 864;        // Framebuffer width (physical pixels for WebGPU)
    int height = 664;       // Framebuffer height (physical pixels for WebGPU)
    int logicalWidth = 864; // Logical window width (for mouse coordinates)
    int logicalHeight = 664; // Logical window height (for mouse coordinates)
};

// Julia fractal settings sub-struct
struct JuliaSettings {
    double angle = 2.8;  // Angle for circular motion
    double radius = 0.7885;  // Radius for circular motion
    double angleVelocity = 0.0001;  // Angle increment per frame
    bool motionEnabled = true;  // Flag to enable/disable Julia motion
};

// Visualization and coloring settings sub-struct
struct VisualizationSettings {
    VisualizationMode mode = JULIA_NORMAL;
    ColoringMode coloringMode = COLOR_CONTINUOUS;
    //bool useExponentialFilter = false;
    bool useBanding = true;  // For future color banding feature
    int highIterationThreshold = 200;
    int maxIterations = 200;
};

// Interaction settings sub-struct
struct InteractionSettings {
    bool mouseDragging = false;
    bool leftButtonHeld = false;
    double mouseDragStartX = 0.0;     // Fractal coordinates where drag started (deprecated)
    double mouseDragStartY = 0.0;     // Fractal coordinates where drag started (deprecated)
    double mouseDragOffsetX = 0.0;    // Fractal coordinate offset (deprecated - not needed)
    double mouseDragOffsetY = 0.0;    // Fractal coordinate offset (deprecated - not needed)
    double lastMouseX = 0.0;          // Last mouse position in fractal coordinates (deprecated)
    double lastMouseY = 0.0;          // Last mouse position in fractal coordinates (deprecated)
    
    // New drag handling - using screen coordinates for stability
    double dragStartScreenX = 0.0;   // Screen coordinates where drag started
    double dragStartScreenY = 0.0;   // Screen coordinates where drag started
    double dragStartViewX = 0.0;     // Viewport viewX position when drag started
    double dragStartViewY = 0.0;     // Viewport viewY position when drag started
};

// Main fractal settings struct
struct FractalSettings {
    ViewportSettings viewport;
    JuliaSettings julia;
    VisualizationSettings visualization;
    InteractionSettings interaction;
    
    // Performance settings
    int targetFPS = 30;
    
    // Debug/logging settings
    bool showDebugInfo = true;
};

// Helper function to get visualization mode name
inline std::string getVisualizationModeName(VisualizationMode mode) {
    switch (mode) {
        case JULIA_NORMAL: return "Normal";
        case DISTANCE_FIELD: return "Distance Field";
        case ITERATION_MASK: return "Iteration Mask";
        default: return "Unknown";
    }
}

// Helper function to get coloring mode name
inline std::string getColoringModeName(ColoringMode mode) {
    switch (mode) {
        case COLOR_BANDS: return "Color Bands";
        case COLOR_CONTINUOUS: return "Color Continuous"; 
        case GRAYSCALE: return "Grayscale";
        case THRESHOLD: return "Threshold";
        default: return "Unknown";
    }
} 