#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// Launches the CUDA kernel that fills the mapped buffer with a gradient.
// Parameters:
//   devPtr - pointer to the mapped CUDA buffer
//   width, height - dimensions of the image buffer
//   offset - a parameter to vary the gradient (changes every frame)
void launchGradientKernel(unsigned char* devPtr, int width, int height, float offset);

// Initializes the persistent buffer for exponential filtering
// Should be called once at the beginning of the program
// Returns: true if initialization successful, false otherwise
bool initExponentialFilter(int width, int height);

// Cleans up the persistent buffer for exponential filtering
// Should be called at program exit
void cleanupExponentialFilter();

// Launches the CUDA kernel that fills the mapped buffer with a Mandelbrot set.
// Parameters:
//   devPtr - pointer to the mapped CUDA buffer
//   width, height - dimensions of the image buffer
//   viewX, viewY - top-left coordinates of the viewport in Mandelbrot space
//   viewWidth, viewHeight - width and height of the viewport in Mandelbrot space
//   useFilter - whether to apply the exponential filter (blend with previous frame)
void launchMandelbrotKernel(unsigned char* devPtr, int width, int height, 
                           double viewX, double viewY, double viewWidth, double viewHeight,
                           bool useFilter = true);

// Launches the CUDA kernel that fills the mapped buffer with a Julia set.
// Parameters:
//   devPtr - pointer to the mapped CUDA buffer
//   width, height - dimensions of the image buffer
//   viewX, viewY - top-left coordinates of the viewport in complex plane
//   viewWidth, viewHeight - width and height of the viewport in complex plane
//   julia_cx, julia_cy - constant c value (cx + cy*i) in f(z) = z^2 + c
//   useFilter - whether to apply the exponential filter (blend with previous frame)
void launchJuliaKernel(unsigned char* devPtr, int width, int height,
                      double viewX, double viewY, double viewWidth, double viewHeight,
                      double julia_cx, double julia_cy,
                      bool useFilter = true);

// Launches the CUDA kernel that visualizes the iteration count mask
// Parameters:
//   devPtr - pointer to the mapped CUDA buffer
//   width, height - dimensions of the image buffer
//   viewX, viewY - top-left coordinates of the viewport in Mandelbrot space
//   viewWidth, viewHeight - width and height of the viewport in Mandelbrot space
//   julia_cx, julia_cy - constant c value (cx + cy*i) for Julia set, if needed
void launchMaskVisualizationKernel(unsigned char* devPtr, int width, int height,
                                 double viewX, double viewY, double viewWidth, double viewHeight,
                                 double julia_cx = 0.0, double julia_cy = 0.0);

// Launches the CUDA kernel that fills the mapped buffer with distance visualization using Jump Flooding Algorithm
// Parameters:
//   devPtr - pointer to the mapped CUDA buffer
//   width, height - dimensions of the image buffer
//   viewX, viewY - top-left coordinates of the viewport in Mandelbrot space
//   viewWidth, viewHeight - width and height of the viewport in Mandelbrot space
//   julia_cx, julia_cy - constant c value (cx + cy*i) for Julia set, if needed
void launchDistanceVisualizationKernel(unsigned char* devPtr, int width, int height,
                                     double viewX, double viewY, double viewWidth, double viewHeight,
                                     double julia_cx = 0.0, double julia_cy = 0.0);

// Gets the current high iteration threshold used in the mandelbrot kernel
int getHighIterationThreshold();

// Increases the high iteration threshold by the specified amount (default 10)
void increaseHighIterationThreshold(int amount = 10);

// Decreases the high iteration threshold by the specified amount (default 10)
// Will never decrease below zero
void decreaseHighIterationThreshold(int amount = 10);

// Gets current flag for whether to use Julia fractal instead of Mandelbrot
bool getUseJuliaFractal();

// Sets the flag for using Julia fractal
void setUseJuliaFractal(bool useJulia);

#ifdef __cplusplus
}
#endif

#endif // CUDA_KERNELS_H
