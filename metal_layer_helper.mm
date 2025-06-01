#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

// C-style interface for calling from C++
extern "C" {
    // Sets up a CAMetalLayer on the GLFW window and returns it
    void* CreateMetalLayer(GLFWwindow* window) {
        @autoreleasepool {
            // Get the NSWindow from GLFW
            NSWindow* nsWindow = glfwGetCocoaWindow(window);
            if (!nsWindow) {
                return nullptr;
            }
            
            // Get the content view
            NSView* view = [nsWindow contentView];
            
            // Set up the Metal layer
            [view setWantsLayer:YES];
            [view setLayer:[CAMetalLayer layer]];
            
            // Use retina if the window was created with retina support
            [[view layer] setContentsScale:[nsWindow backingScaleFactor]];
            
            // Return the layer pointer (we need to retain it so it lives beyond this function)
            CAMetalLayer* layer = (CAMetalLayer*)[view layer];
            [layer retain]; // Retain to prevent deallocation
            return layer;
        }
    }
    
    // Release the CAMetalLayer when done
    void ReleaseMetalLayer(void* layer) {
        if (layer) {
            // Release the retained object
            [(CAMetalLayer*)layer release];
        }
    }
} 