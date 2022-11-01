#pragma once

#include "complex.cuh"
#include "macros.cuh"
#include "mandelbrotHelper.cuh"

/** evaluates the common border dwell, if it exists */
__device__ int dp_sbr_border_dwell(int *dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d, unsigned int MAX_DWELL) {
    // check whether all boundary pixels have the same dwell
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int bs = blockDim.x * blockDim.y;
    int comm_dwell = NEUT_DWELL;
    // for all boundary pixels, distributed across threads
    for (int r = tid; r < d; r += bs) {
        // for each boundary: b = 0 is east, then counter-clockwise
        for (int b = 0; b < 4; b++) {
            int x = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
            int y = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
            int dwell = pixel_dwell(w, h, cmin, cmax, x, y, MAX_DWELL);
            comm_dwell = same_dwell(comm_dwell, dwell, MAX_DWELL);
            // dwells[y * w + x] = constant;
            // pixel_dwell(w, h, cmin, cmax, x, y);
        }
    } // for all boundary pixels
    // reduce across threads in the block
    __shared__ int ldwells[BSX * BSY];
    int nt = min(d, BSX * BSY);
    if (tid < nt)
        ldwells[tid] = comm_dwell;
    __syncthreads();
    for (; nt > 1; nt /= 2) {
        if (tid < nt / 2){
            ldwells[tid] = same_dwell(ldwells[tid], ldwells[tid + nt / 2], MAX_DWELL);
        }
        __syncthreads();
    }
    return ldwells[0];
} 

/** the kernel to fill the image region with a specific dwell value */
__global__ void dp_sbr_dwell_fill_k(int *dwells, size_t w, unsigned int x0,
                             unsigned int y0, int d, int dwell) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    for (unsigned int ry = y; ry < d; ry += blockDim.y) {
        for (unsigned int rx = x; rx < d; rx += blockDim.x) {
            unsigned int rxx = rx + x0, ryy = ry + y0;
            dwells[ryy * (size_t)w + rxx] = dwell;
        }
    }
} 

/** the kernel to fill in per-pixel values of the portion of the Mandelbrot set */
__global__ void dp_sbr_mandelbrot_pixel_k(int *dwells, unsigned int w, unsigned int h, complex cmin, complex cmax, unsigned int x0, unsigned int y0, int d, unsigned int MAX_DWELL) {
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    for (unsigned int ry = y; ry < d; ry += blockDim.y) {
        for (unsigned int rx = x; rx < d; rx += blockDim.x) {
            unsigned int rxx = rx + x0, ryy = ry + y0;
            dwells[ryy * (size_t)w + rxx] = pixel_dwell(w, h, cmin, cmax, rxx, ryy, MAX_DWELL);
        }
    }

}


/** Equivalent to the dynamic parallelism approach **/
#ifdef RDC_TRUE
__global__ void dp_sbr_mandelbrot_block_k(int *dwells, unsigned int w, unsigned int h,
                                   complex cmin, complex cmax, unsigned int x0,
                                   unsigned int y0, int d, int depth,
                                   unsigned int SUBDIV, unsigned int MAX_DWELL,
                                   unsigned int MIN_SIZE, unsigned int MAX_DEPTH) {
    x0 += d * blockIdx.x, y0 += d * blockIdx.y;
    int comm_dwell = dp_sbr_border_dwell(dwells, w, h, cmin, cmax, x0, y0, d, MAX_DWELL);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (comm_dwell != DIFF_DWELL) {
            // uniform dwell, just fill
            dim3 bs(BSX, BSY), grid(1, 1);
            dp_sbr_dwell_fill_k<<<grid, bs>>>(dwells, w, x0, y0, d, comm_dwell);
        } else if (depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
            // subdivide recursively
            dim3 bs(blockDim.x, blockDim.y), grid(SUBDIV, SUBDIV);
            dp_sbr_mandelbrot_block_k<<<grid, bs>>>(dwells, w, h, cmin, cmax, x0, y0, d / SUBDIV, depth + 1, SUBDIV, MAX_DWELL, MIN_SIZE, MAX_DEPTH);
        } else {
            // leaf, per-pixel kernel
            dim3 bs(BSX, BSY), grid(1, 1);
            dp_sbr_mandelbrot_pixel_k<<<grid, bs>>>(dwells, w, h, cmin, cmax, x0, y0, d, MAX_DWELL);
        }
    }
} // mandelbrot_block_k
#endif
