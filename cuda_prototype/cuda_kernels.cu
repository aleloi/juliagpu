// cuda_kernels.cu
// Compile this file with nvcc.

#include "cuda_kernels.h"
#include <cuda_runtime.h>

#include <cuda_gl_interop.h>
#include <cstdio>
#include <cstdlib>

// Exponential filter constant
#define EXPOFILTER_CONSTANT 0.5f
// Maximum number of iterations for fractal calculation
#define MAX_ITERATION 200

// Macro for error checking; note the do/while(0) idiom can be used for macros in production.
#define CUDA_SAFE_CALL(call) {                                           \
    cudaError_t err = call;                                              \
    if(err != cudaSuccess) {                                             \
       fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
               cudaGetErrorString(err));                                 \
       exit(EXIT_FAILURE);                                               \
    }                                                                    \
}

// Global persistent buffer for exponential filtering
unsigned char* g_persistentBuffer = nullptr;
int g_bufferWidth = 0;
int g_bufferHeight = 0;

// Global buffers for Jump Flooding Algorithm (JFA)
// Binary mask of high-iteration points
unsigned char* g_mandelbrotMask = nullptr;
// Two ping-pong buffers for JFA
int2* g_jfaBufferA = nullptr;  // [x, y] coordinates of nearest feature point
int2* g_jfaBufferB = nullptr;
// Distance buffer
float* g_distanceBuffer = nullptr;

// Flag for visualization mode
bool g_useDistanceVisualization = false;

// Global variable for high iteration threshold
int g_highIterationThreshold = 150;

// Flag to track which fractal we're currently displaying
bool g_useJuliaFractal = true;

// Global variables to store min and max distance values
__device__ float g_minDistance = INFINITY;
__device__ float g_maxDistance = -INFINITY;

// Device function to snap coordinates to pixel grid for numerical stability
__device__ void snapCoordinates(double viewX, double viewY, double viewWidth, double viewHeight,
                             int width, int height, int x, int y,
                             double* outX, double* outY) {
    double pixelSizeX = viewWidth / width;
    double pixelSizeY = viewHeight / height;

    // Snap the view anchor to the grid of multiples of pixelSize
    double snappedViewX = round(viewX / pixelSizeX) * pixelSizeX;
    double snappedViewY = round(viewY / pixelSizeY) * pixelSizeY;

    // Calculate the coordinates for this pixel
    *outX = snappedViewX + x * pixelSizeX;
    *outY = snappedViewY + y * pixelSizeY;
}

// Device function to calculate smooth color based on iteration count and escaped values
__device__ void calculateSmoothColor(int iteration, double zx2, double zy2, double* smoothColor) {
    if (iteration == MAX_ITERATION) {
        *smoothColor = iteration;
    } else {
        // Smooth coloring formula
        double log_zn = log(zx2 + zy2) / 2.0;
        double nu = log(log_zn / log(2.0)) / log(2.0);
        *smoothColor = iteration + 1.0 - nu;
    }
}

// Device function to map smooth color to RGB values
__device__ void mapColorGradient(double smoothColor, unsigned char* r, unsigned char* g, unsigned char* b) {
    // Default to black
    *r = 0;
    *g = 0;
    *b = 0;
    
    if (smoothColor < MAX_ITERATION) {
        // Create color using smooth value
        double t = smoothColor / MAX_ITERATION;
        
        // Color gradient mapping
        if (t < 0.16f) {
            float s = t / 0.16f;
            *r = 0;
            *g = 0;
            *b = static_cast<unsigned char>(255 * s);
        } else if (t < 0.42f) {
            float s = (t - 0.16f) / (0.42f - 0.16f);
            *r = static_cast<unsigned char>(255 * s);
            *g = static_cast<unsigned char>(165 * s);
            *b = static_cast<unsigned char>(255 * (1.0f - s));
        } else if (t < 0.6425f) {
            float s = (t - 0.42f) / (0.6425f - 0.42f);
            *r = 255;
            *g = static_cast<unsigned char>(165 * (1.0f - s));
            *b = 0;
        } else {
            float s = (t - 0.6425f) / (1.0f - 0.6425f);
            *r = 255;
            *g = static_cast<unsigned char>(255 * s);
            *b = static_cast<unsigned char>(255 * s);
        }
    }
}

// Device function to apply exponential filter and update buffers
__device__ void applyExponentialFilter(
    unsigned char* ptr, unsigned char* persistentPtr, int idx,
    unsigned char* r, unsigned char* g, unsigned char* b, bool useFilter) {
    
    if (useFilter && persistentPtr != nullptr) {
        // Read old values
        float oldR = persistentPtr[idx + 0];
        float oldG = persistentPtr[idx + 1];
        float oldB = persistentPtr[idx + 2];
        
        // Apply exponential filter
        *r = static_cast<unsigned char>(oldR * EXPOFILTER_CONSTANT + *r * (1.0f - EXPOFILTER_CONSTANT));
        *g = static_cast<unsigned char>(oldG * EXPOFILTER_CONSTANT + *g * (1.0f - EXPOFILTER_CONSTANT));
        *b = static_cast<unsigned char>(oldB * EXPOFILTER_CONSTANT + *b * (1.0f - EXPOFILTER_CONSTANT));
        
        // Update persistent buffer
        persistentPtr[idx + 0] = *r;
        persistentPtr[idx + 1] = *g;
        persistentPtr[idx + 2] = *b;
        persistentPtr[idx + 3] = 255;
    }
    
    // Write to output buffer
    if (ptr != nullptr) {
        ptr[idx + 0] = *r; // R
        ptr[idx + 1] = *g; // G
        ptr[idx + 2] = *b; // B
        ptr[idx + 3] = 255; // Fully opaque
    }
}

// Initializes the persistent buffer for exponential filtering
bool initExponentialFilter(int width, int height) {
    g_bufferWidth = width;
    g_bufferHeight = height;
    
    // Allocate memory for persistent buffer
    size_t bufferSize = width * height * 4 * sizeof(unsigned char);
    CUDA_SAFE_CALL(cudaMalloc(&g_persistentBuffer, bufferSize));
    
    // Initialize buffer to zeros
    CUDA_SAFE_CALL(cudaMemset(g_persistentBuffer, 0, bufferSize));
    
    return true;
}

// Initializes buffers for Jump Flooding Algorithm (JFA)
bool initJFABuffers(int width, int height) {
    size_t maskSize = width * height * sizeof(unsigned char);
    size_t jfaSize = width * height * sizeof(int2);
    size_t distanceSize = width * height * sizeof(float);
    
    // Allocate memory for mask buffer
    CUDA_SAFE_CALL(cudaMalloc(&g_mandelbrotMask, maskSize));
    
    // Allocate memory for JFA ping-pong buffers
    CUDA_SAFE_CALL(cudaMalloc(&g_jfaBufferA, jfaSize));
    CUDA_SAFE_CALL(cudaMalloc(&g_jfaBufferB, jfaSize));
    
    // Allocate memory for distance buffer
    CUDA_SAFE_CALL(cudaMalloc(&g_distanceBuffer, distanceSize));
    
    // Initialize buffers
    CUDA_SAFE_CALL(cudaMemset(g_mandelbrotMask, 0, maskSize));
    CUDA_SAFE_CALL(cudaMemset(g_distanceBuffer, 0, distanceSize));
    
    return true;
}

// Cleans up the persistent buffer for exponential filtering
void cleanupExponentialFilter() {
    if (g_persistentBuffer) {
        CUDA_SAFE_CALL(cudaFree(g_persistentBuffer));
        g_persistentBuffer = nullptr;
    }
    g_bufferWidth = 0;
    g_bufferHeight = 0;
}

// Cleans up the JFA buffers
void cleanupJFABuffers() {
    if (g_mandelbrotMask) {
        CUDA_SAFE_CALL(cudaFree(g_mandelbrotMask));
        g_mandelbrotMask = nullptr;
    }
    if (g_jfaBufferA) {
        CUDA_SAFE_CALL(cudaFree(g_jfaBufferA));
        g_jfaBufferA = nullptr;
    }
    if (g_jfaBufferB) {
        CUDA_SAFE_CALL(cudaFree(g_jfaBufferB));
        g_jfaBufferB = nullptr;
    }
    if (g_distanceBuffer) {
        CUDA_SAFE_CALL(cudaFree(g_distanceBuffer));
        g_distanceBuffer = nullptr;
    }
}

// CUDA kernel: fills the buffer with a simple gradient.
// Each pixel is 4 bytes (RGBA).
__global__ void gradientKernel(unsigned char* ptr, int width, int height, float offset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = (y * width + x) * 4;
    // Simple gradient: add offset so the image changes over time.
    unsigned char r = static_cast<unsigned char>((x + offset));
    unsigned char g = static_cast<unsigned char>((y + offset));
    unsigned char b = static_cast<unsigned char>(((x + y) / 2 + offset));
    ptr[idx + 0] = r;
    ptr[idx + 1] = g;
    ptr[idx + 2] = b;
    ptr[idx + 3] = 255; // Fully opaque.
}

// CUDA kernel: fills the buffer with a Mandelbrot set.
// Also outputs a binary mask of high-iteration points (iteration > g_highIterationThreshold)
__global__ void mandelbrotKernel(unsigned char* ptr, unsigned char* persistentPtr, 
                               unsigned char* mask, 
                               int width, int height, 
                               double viewX, double viewY, double viewWidth, double viewHeight,
                               bool useFilter, int highIterationThreshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Get coordinates using the snap function
    double x0, y0;
    snapCoordinates(viewX, viewY, viewWidth, viewHeight, width, height, x, y, &x0, &y0);
    
    int iteration = 0;
    
    double zx = 0.0;
    double zy = 0.0;
    double zx2 = 0.0;
    double zy2 = 0.0;
    
    // Mandelbrot iteration
    while (zx2 + zy2 <= 4.0f && iteration < MAX_ITERATION) {
        zy = 2.0 * zx * zy + y0;
        zx = zx2 - zy2 + x0;
        zx2 = zx * zx;
        zy2 = zy * zy;
        iteration++;
    }
    
    // Set mask value for high-iteration points if mask buffer is provided
    if (mask != nullptr) {
        int idx = y * width + x;
        mask[idx] = (iteration > highIterationThreshold) ? 1 : 0;
    }
    
    // If no output buffer is provided, return after setting mask
    if (ptr == nullptr) return;
    
    // Calculate smooth color
    double smooth_color;
    calculateSmoothColor(iteration, zx2, zy2, &smooth_color);
    
    // Map smooth_color to RGB
    int idx = (y * width + x) * 4;
    unsigned char r, g, b;
    
    // Get color from gradient
    mapColorGradient(smooth_color, &r, &g, &b);
    
    // Apply exponential filter and update buffers
    applyExponentialFilter(ptr, persistentPtr, idx, &r, &g, &b, useFilter);
}

// CUDA kernel: fills the buffer with a Julia set.
// Also outputs a binary mask of high-iteration points (iteration > highIterationThreshold)
__global__ void juliaKernel(unsigned char* ptr, unsigned char* persistentPtr, 
                          unsigned char* mask,
                          int width, int height, 
                          double viewX, double viewY, double viewWidth, double viewHeight,
                          double julia_cx, double julia_cy,
                          bool useFilter, int highIterationThreshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Get coordinates using the snap function
    double zx, zy;
    snapCoordinates(viewX, viewY, viewWidth, viewHeight, width, height, x, y, &zx, &zy);
    
    int iteration = 0;
    
    double zx2 = zx * zx;
    double zy2 = zy * zy;
    
    // Julia set iteration: z = z² + c where c is a constant
    while (zx2 + zy2 <= 4.0f && iteration < MAX_ITERATION) {
        double zy_new = 2.0 * zx * zy + julia_cy;
        double zx_new = zx2 - zy2 + julia_cx;
        
        zx = zx_new;
        zy = zy_new;
        
        zx2 = zx * zx;
        zy2 = zy * zy;
        iteration++;
    }
    
    // Set mask value for high-iteration points if mask buffer is provided
    if (mask != nullptr) {
        int idx = y * width + x;
        mask[idx] = (iteration > highIterationThreshold) ? 1 : 0;
    }
    
    // If no output buffer is provided, return after setting mask
    if (ptr == nullptr) return;
    
    // Calculate smooth color
    double smooth_color;
    calculateSmoothColor(iteration, zx2, zy2, &smooth_color);
    
    // Map smooth_color to RGB
    int idx = (y * width + x) * 4;
    unsigned char r, g, b;
    
    // Get color from gradient
    mapColorGradient(smooth_color, &r, &g, &b);
    
    // Apply exponential filter and update buffers
    applyExponentialFilter(ptr, persistentPtr, idx, &r, &g, &b, useFilter);
}

// CUDA kernel: initialize JFA buffers
__global__ void initJFAKernel(unsigned char* mask, int2* jfaBuffer, float* distanceBuffer, 
                           int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Initialize JFA buffer
    if (mask[idx] == 1) {
        // This is a seed point (high-iteration)
        jfaBuffer[idx] = make_int2(x, y);  // Store its own coordinates
        distanceBuffer[idx] = 0.0f;        // Distance to itself is 0
    } else {
        // Non-seed point
        jfaBuffer[idx] = make_int2(-1, -1);  // -1 means "no seed found yet"
        distanceBuffer[idx] = INFINITY;     // Initial distance is infinity
    }
}

// CUDA kernel: Jump Flooding Algorithm (JFA) step
__global__ void jfaKernel(int2* inputBuffer, int2* outputBuffer, float* distanceBuffer, 
                       int width, int height, int jumpLength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int2 currentSeed = inputBuffer[idx];
    float currentDistance = distanceBuffer[idx];
    
    // Check 8 neighbors at current jump length
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx * jumpLength;
            int ny = y + dy * jumpLength;
            
            // Skip if neighbor is out of bounds
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            
            int nidx = ny * width + nx;
            int2 neighborSeed = inputBuffer[nidx];
            
            // Skip if neighbor has no seed
            if (neighborSeed.x == -1 && neighborSeed.y == -1) continue;
            
            // Calculate distance to neighbor's seed
            float dx_seed = x - neighborSeed.x;
            float dy_seed = y - neighborSeed.y;
            float distance = dx_seed * dx_seed + dy_seed * dy_seed;
            
            // Update if this seed is closer
            if (distance < currentDistance) {
                currentSeed = neighborSeed;
                currentDistance = distance;
            }
        }
    }
    
    // Update output buffer with new closest seed
    outputBuffer[idx] = currentSeed;
    distanceBuffer[idx] = currentDistance;
}

// CUDA kernel: find min and max distance values using atomic operations
__global__ void findMinMaxDistanceKernel(float* distanceBuffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float distance = sqrtf(distanceBuffer[idx]);
    
    // Skip invalid or infinite distances
    if (isfinite(distance)) {
        atomicMin((int*)&g_minDistance, __float_as_int(distance));
        atomicMax((int*)&g_maxDistance, __float_as_int(distance));
    }
}

// CUDA kernel: visualize distance field
__global__ void visualizeDistanceKernel(unsigned char* ptr, float* distanceBuffer, 
                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float distance = sqrtf(distanceBuffer[idx]);
    
    // Get min and max distance for normalization
    float minDistance = g_minDistance;
    float maxDistance = min(100.0f, g_maxDistance);
    
    // Ensure we have a valid range
    if (minDistance == maxDistance) {
        minDistance = 0.0f;
        maxDistance = 100.0f;
    }
    
    // Normalize distance to [0, 1]
    float normalizedDistance = 0.0f;
    if (isfinite(distance)) {
        normalizedDistance = (distance - minDistance) / (maxDistance - minDistance);
        normalizedDistance = max(0.0f, min(normalizedDistance, 1.0f)); // Clamp to [0,1]
        normalizedDistance = (1-normalizedDistance)*(1-normalizedDistance)*(1-normalizedDistance);
    }
    
    // Grayscale mapping: 0 = white, 1 = black
    unsigned char intensity = static_cast<unsigned char>((1.0f - normalizedDistance) * 255);
    
    // Set pixel color (grayscale)
    idx = (y * width + x) * 4;
    ptr[idx + 0] = intensity;
    ptr[idx + 1] = intensity;
    ptr[idx + 2] = intensity;
    ptr[idx + 3] = 255; // Fully opaque
}

// CUDA kernel: visualize the iteration mask
__global__ void visualizeMaskKernel(unsigned char* ptr, unsigned char* mask, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int pixelIdx = idx * 4;
    
    // Set color based on mask value (white for high-iteration points, black for low-iteration)
    unsigned char value = mask[idx] == 1 ? 255 : 0;
    
    ptr[pixelIdx + 0] = value;     // R
    ptr[pixelIdx + 1] = value;     // G
    ptr[pixelIdx + 2] = value;     // B
    ptr[pixelIdx + 3] = 255;       // Fully opaque
}

// Exported function to launch the gradient kernel.
// This function will be called from the C++ code in main.cc.
void launchGradientKernel(unsigned char* devPtr, int width, int height, float offset) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    gradientKernel<<<grid, block>>>(devPtr, width, height, offset);
    CUDA_SAFE_CALL(cudaGetLastError());
    // Optionally, you can call cudaDeviceSynchronize() here if you need to ensure the kernel finished.
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

// Exported function to launch the Mandelbrot kernel.
void launchMandelbrotKernel(unsigned char* devPtr, int width, int height, 
                          double viewX, double viewY, double viewWidth, double viewHeight,
                          bool useFilter) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Initialize exponential filter buffer if needed
    if (useFilter && (g_persistentBuffer == nullptr || width != g_bufferWidth || height != g_bufferHeight)) {
        if (g_persistentBuffer != nullptr) {
            cleanupExponentialFilter();
        }
        g_bufferWidth = width;
        g_bufferHeight = height;
        initExponentialFilter(width, height);
    }
    
    // Check if we need to initialize the JFA buffers
    if (g_mandelbrotMask == nullptr || width != g_bufferWidth || height != g_bufferHeight) {
        if (g_mandelbrotMask != nullptr) {
            cleanupJFABuffers();
        }
        g_bufferWidth = width;
        g_bufferHeight = height;
        initJFABuffers(width, height);
    }
    
    g_useDistanceVisualization = false;
    
    // Run the Mandelbrot kernel to generate color and mask
    mandelbrotKernel<<<grid, block>>>(devPtr, g_persistentBuffer, g_mandelbrotMask, 
                                     width, height, 
                                     viewX, viewY, viewWidth, viewHeight, 
                                     useFilter, g_highIterationThreshold);
    
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

// Exported function to launch the Julia kernel.
void launchJuliaKernel(unsigned char* devPtr, int width, int height,
                      double viewX, double viewY, double viewWidth, double viewHeight,
                      double julia_cx, double julia_cy,
                      bool useFilter) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Initialize exponential filter buffer if needed
    if (useFilter && (g_persistentBuffer == nullptr || width != g_bufferWidth || height != g_bufferHeight)) {
        if (g_persistentBuffer != nullptr) {
            cleanupExponentialFilter();
        }
        g_bufferWidth = width;
        g_bufferHeight = height;
        initExponentialFilter(width, height);
    }
    
    // Check if we need to initialize the JFA buffers
    if (g_mandelbrotMask == nullptr || width != g_bufferWidth || height != g_bufferHeight) {
        if (g_mandelbrotMask != nullptr) {
            cleanupJFABuffers();
        }
        g_bufferWidth = width;
        g_bufferHeight = height;
        initJFABuffers(width, height);
    }
    
    g_useDistanceVisualization = false;
    
    // Run the Julia kernel to generate color and mask
    juliaKernel<<<grid, block>>>(devPtr, g_persistentBuffer, g_mandelbrotMask,
                               width, height,
                               viewX, viewY, viewWidth, viewHeight,
                               julia_cx, julia_cy,
                               useFilter, g_highIterationThreshold);
    
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

// Exported function to launch the distance visualization kernel.
void launchDistanceVisualizationKernel(unsigned char* devPtr, int width, int height, 
                                     double viewX, double viewY, double viewWidth, double viewHeight,
                                     double julia_cx, double julia_cy) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Check if we need to initialize the JFA buffers
    if (g_mandelbrotMask == nullptr || width != g_bufferWidth || height != g_bufferHeight) {
        if (g_mandelbrotMask != nullptr) {
            cleanupJFABuffers();
        }
        g_bufferWidth = width;
        g_bufferHeight = height;
        initJFABuffers(width, height);
        g_useDistanceVisualization = true;
    }
    
    // Run the appropriate kernel to generate mask (but don't update display buffer yet)
    // Pass nullptr for the display buffer and persistent buffer since we only need the mask
    if (g_useJuliaFractal) {
        // Use the passed in julia parameters
        juliaKernel<<<grid, block>>>(nullptr, nullptr, g_mandelbrotMask,
                                   width, height,
                                   viewX, viewY, viewWidth, viewHeight,
                                   julia_cx, julia_cy,
                                   false, g_highIterationThreshold);
    } else {
        mandelbrotKernel<<<grid, block>>>(nullptr, nullptr, g_mandelbrotMask, 
                                        width, height, 
                                        viewX, viewY, viewWidth, viewHeight, 
                                        false, g_highIterationThreshold);
    }
    CUDA_SAFE_CALL(cudaGetLastError());
    
    // Initialize JFA buffers
    initJFAKernel<<<grid, block>>>(g_mandelbrotMask, g_jfaBufferA, g_distanceBuffer, width, height);
    CUDA_SAFE_CALL(cudaGetLastError());
    
    // Run Jump Flooding Algorithm - log2(N) iterations
    int maxDim = max(width, height);
    int jumpLength = maxDim / 2;
    
    while (jumpLength >= 1) {
        // Ping-pong between buffers A and B
        jfaKernel<<<grid, block>>>(g_jfaBufferA, g_jfaBufferB, g_distanceBuffer, width, height, jumpLength);
        CUDA_SAFE_CALL(cudaGetLastError());
        
        // Swap buffers
        int2* temp = g_jfaBufferA;
        g_jfaBufferA = g_jfaBufferB;
        g_jfaBufferB = temp;
        
        // Halve the jump length
        jumpLength /= 2;
    }
    
    // Reset min/max device variables
    float hostMinInit = INFINITY;
    float hostMaxInit = -INFINITY;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_minDistance, &hostMinInit, sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_maxDistance, &hostMaxInit, sizeof(float)));
    
    // Find min and max distance values
    findMinMaxDistanceKernel<<<grid, block>>>(g_distanceBuffer, width, height);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    // Visualize distance field - ensure devPtr is not null
    if (devPtr != nullptr) {
        visualizeDistanceKernel<<<grid, block>>>(devPtr, g_distanceBuffer, width, height);
        CUDA_SAFE_CALL(cudaGetLastError());
    }
    
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

// Exported function to launch the mask visualization kernel
void launchMaskVisualizationKernel(unsigned char* devPtr, int width, int height,
                                 double viewX, double viewY, double viewWidth, double viewHeight,
                                 double julia_cx, double julia_cy) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Check if we need to initialize the JFA buffers (which includes the mask)
    if (g_mandelbrotMask == nullptr || width != g_bufferWidth || height != g_bufferHeight) {
        if (g_mandelbrotMask != nullptr) {
            cleanupJFABuffers();
        }
        g_bufferWidth = width;
        g_bufferHeight = height;
        initJFABuffers(width, height);
    }
    
    // Generate the mask with the appropriate kernel
    if (g_useJuliaFractal) {
        // Use the passed in julia parameters
        juliaKernel<<<grid, block>>>(nullptr, nullptr, g_mandelbrotMask,
                                   width, height,
                                   viewX, viewY, viewWidth, viewHeight,
                                   julia_cx, julia_cy,
                                   false, g_highIterationThreshold);
    } else {
        mandelbrotKernel<<<grid, block>>>(nullptr, nullptr, g_mandelbrotMask, 
                                     width, height, 
                                     viewX, viewY, viewWidth, viewHeight, 
                                     false, g_highIterationThreshold);
    }
    CUDA_SAFE_CALL(cudaGetLastError());
    
    // Now visualize the mask
    visualizeMaskKernel<<<grid, block>>>(devPtr, g_mandelbrotMask, width, height);
    CUDA_SAFE_CALL(cudaGetLastError());
    
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

// Gets the current high iteration threshold
int getHighIterationThreshold() {
    return g_highIterationThreshold;
}

// Increases the high iteration threshold
void increaseHighIterationThreshold(int amount) {
    g_highIterationThreshold += amount;
    printf("High iteration threshold increased to: %d\n", g_highIterationThreshold);
}

// Decreases the high iteration threshold, ensuring it doesn't go below zero
void decreaseHighIterationThreshold(int amount) {
    g_highIterationThreshold = max(0, g_highIterationThreshold - amount);
    printf("High iteration threshold decreased to: %d\n", g_highIterationThreshold);
}

// Get/set whether to use Julia fractal
bool getUseJuliaFractal() {
    return g_useJuliaFractal;
}

void setUseJuliaFractal(bool useJulia) {
    g_useJuliaFractal = useJulia;
    printf("Fractal type: %s\n", useJulia ? "Julia" : "Mandelbrot");
}