# Mandelbrot on the Raspberry Pi GPU

This repository contains the code to compute the Mandelbrot set on the GPU of the Raspberry Pi. It employs the twelve Quad Processor Units (QPUs) for the computation and optionally writes the final result into a NetCDF file.

The code for the host processor (ARM) is mainly contained in the `qou_mandel.c` file, which basically calls the QPU code for each twelve lines. The GPU code is in the file `gpu_code.qasm`, which can be compiled with the [vc4asm macro assembler](http://maazl.de/project/vc4asm/doc/). For convenience, the compiled code is provided in the `gpu_code.hex` file.

### Compiling
If the NetCDF header files are installed, `make` should be sufficient. Depending on the distribution of the NetCDF package, it might be required to use `CC=mpicc` for the compilation. The code has been tested with the Raspberry Pi 2.

### Performance
Running times for a resolution of 1920 x 1080 in the range of `-2,2;-1.125,1.125` with 100,000 iterations and a maximum distance of 2 (implies that `maxValue` is 4). The Raspberry Pi is not overclocked, i.e. the ARM core is clocked at 900 MHz and the GPU at 250 MHz.

| Method | Running Time |
| ------ | ------------ |
| ARM, single-core, scalar | 1046.365 seconds |
| ARM NEON, single-core, 4-lane SIMD | 409.827 seconds |
| ARM NEON, 4 cores (OpenMP, dynamic scheduling), 4-lane SIMD | 104.036 seconds |
| VideoCore IV, 12 QPUs, 16-lane SIMD | 33.781 seconds |
