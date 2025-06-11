# Julia Fractal WebGPU Renderer

There is a CUDA prototype that's "easy" to run if you have nix and a nvidia card: just do `nix run github:aleloi/juliagpu?dir=cuda_prototype --impure` without cloning.

Building the WebGPU version depends on [Dawn](https://dawn.googlesource.com/dawn/+/HEAD/docs/building.md) which has its own non-trivial build system and dependencies.

```bash
git clone --recursive https://github.com/aleloi/juliagpu.git
# Follow build instructions for Dawn https://dawn.googlesource.com/dawn/+/HEAD/docs/building.md
# In theory just this:
git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
cd dawn && cp scripts/standalone.gclient .gclient && gclient sync && cd -
mkdir -p out/Release && cd out/Release && cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DDAWN_ENABLE_METAL=ON -DDAWN_ENABLE_VULKAN=OFF ../.. && cd -
cmake --build out/Release --target install
# I had it ask me for passwords I don't have, wait forever until it fetches all the deps,
# try again from different commits...
# Can some one please make a nix flake?
# Fetch precompiled shared lib?

# Finally for this project:
mkdir build && cd build && cmake ../src -DCMAKE_PREFIX_PATH=../dawn/install/Release && cmake --build . && cd -
``` 

![](presentation/fractal2.png)