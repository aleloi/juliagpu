# CUDA+opengl fractal renderer TODO screenshot

If with nix, just run `nix run github:aleloi/juliagpu?dir=cuda_prototype --impure` without cloning or downloading anything.


`nix build --impure`
and 
`nix run --impure`

should work. It compiles with cuda 12.4, which means your driver must support at least that much. I got into incompatible nvcc / gcc version when trying to change to an older.

Example screenshot:

![](./cuda_fractal.png)