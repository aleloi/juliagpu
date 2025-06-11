{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {

      packages.${system} = {
        default = self.packages.${system}.cuda_gl_app;
        
        cuda_gl_app = pkgs.stdenv.mkDerivation {
          pname = "cuda_gl_app";
          version = "0.1.0";
          
          src = ./.;
          
          nativeBuildInputs = [
            pkgs.cudaPackages.cuda_nvcc
            pkgs.gcc11
            pkgs.pkg-config
          ];
          
          buildInputs = [
            pkgs.cudaPackages.cudatoolkit  # cuda v 12.4 on nixos 24.11. Your driver needs to support at least this high. 
            pkgs.glfw
            pkgs.mesa
            pkgs.glew
          ];
          
          # Set environment variables for the build
          CUDA_INCLUDE_DIR = "${pkgs.cudaPackages.cudatoolkit}/include";
          GLFW_PATH = "${pkgs.glfw}/include";
          GLEW_PATH = "${pkgs.glew}/include";
          
          buildPhase = ''
            # Set compiler flags
            export CXXFLAGS="-O3 -Wall -Wextra"
            export NVCC_FLAGS="-Xcompiler -Wall,-Wextra"
            
            # Compile CUDA kernels
            nvcc -O3 -c cuda_kernels.cu -I$CUDA_INCLUDE_DIR -o cuda_kernels.o $NVCC_FLAGS
            
            # Compile C++ source files
            g++ $CXXFLAGS -c main.cc -I$GLFW_PATH -I$GLEW_PATH -I$CUDA_INCLUDE_DIR -o main.o
            g++ $CXXFLAGS -c bitmap.cc -I$GLFW_PATH -I$GLEW_PATH -I$CUDA_INCLUDE_DIR -o bitmap.o
            g++ $CXXFLAGS -c viewport.cc -I$GLFW_PATH -I$GLEW_PATH -I$CUDA_INCLUDE_DIR -o viewport.o
            
            # Link final executable
            g++ main.o cuda_kernels.o bitmap.o viewport.o -o cuda_gl_app \
              -lglfw -lGL -lGLEW -lcudart \
              -L${pkgs.glfw}/lib -L${pkgs.mesa}/lib -L${pkgs.glew}/lib -L${pkgs.cudaPackages.cudatoolkit}/lib
          '';
          
          installPhase = ''
            mkdir -p $out/bin
            cp cuda_gl_app $out/bin/
          '';
          
          meta = with pkgs.lib; {
            description = "CUDA-OpenGL application for fractal rendering";
            license = licenses.mit;
            platforms = platforms.unix;
          };
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pkgs.cudaPackages.cudatoolkit
          #pkgs.gcc11      # Use gcc11 explicitly
          pkgs.glfw
          pkgs.mesa
          pkgs.glew
        ];
        shellHook = ''
          # Set CC explicitly to the gcc binary from gcc11
          #export GCC11=${pkgs.gcc11}/bin/gcc
          export CUDA_INCLUDE_DIR=${pkgs.cudaPackages.cudatoolkit}/include
          export TERM=xterm
          export EDITOR="emacs -nw"
          export GLFW_PATH=${pkgs.glfw}/include
          export GLEW_PATH=${pkgs.glew}/include
          #echo "Using host compiler: $GCC11, cuda include dir: $CUDA_INCLUDE_DIR"
          echo "Cuda include dir: $CUDA_INCLUDE_DIR"
        '';
      };
    };
}