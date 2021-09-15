#pragma once

#include "complex.cuh"
#include "macros.cuh"
#include "mandelbrotHelper.cuh"

__global__ void mandelbrot_k(int *dwells, unsigned int w, unsigned int h,
                             complex cmin, complex cmax, unsigned int MAX_DWELL) {
    // complex value to start iteration (c)
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    int dwell = pixel_dwell(w, h, cmin, cmax, x, y, MAX_DWELL);
    dwells[y * (size_t)w + x] = dwell;

} // mandelbrot_k