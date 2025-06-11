#!/bin/bash
# nix --extra-experimental-features nix-command --extra-experimental-features flakes eval --impure --raw --expr '(import <nixpkgs> {}).cudaPackages.cuda_nvcc'

# Set compiler flags
CXXFLAGS="-O3 -Wall -Wextra -fsanitize=address,undefined,leak -fno-omit-frame-pointer"
NVCC_FLAGS="-Xcompiler -Wall,-Wextra"

# Compile with flags
nvcc -O3 -c cuda_kernels.cu -I$CUDA_INCLUDE_DIR -lGL -o cuda_kernels.o $NVCC_FLAGS
g++ $CXXFLAGS -c main.cc -I$GLFW_PATH -I$GLEW_PATH -I$CUDA_INCLUDE_DIR -o main.o
g++ $CXXFLAGS -c bitmap.cc -I$GLFW_PATH -I$GLEW_PATH -I$CUDA_INCLUDE_DIR -o bitmap.o
g++ $CXXFLAGS -c viewport.cc -I$GLFW_PATH -I$GLEW_PATH -I$CUDA_INCLUDE_DIR -o viewport.o
g++ main.o cuda_kernels.o bitmap.o viewport.o -o cuda_gl_app -lglfw -lGL -lGLEW -lcudart -fsanitize=address,undefined,leak