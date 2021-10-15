#include <algorithm> // std::swap/
#include <assert.h>
#include <iostream>
#include <omp.h>
#include <png.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GRID_CODE 999
#define SAVE_FRACTAL 0
#define SAVE_GRIDLINES 1

#include "gridlines.cuh"
#include "ask.cuh"
#include "askNEW.cuh"
#include "exhaustive.cuh"
#include "complex.cuh"
#include "dynamicParallelism.cuh"
#include "macros.cuh"
#include "mandelbrotHelper.cuh"
#include "tools.cuh"


using namespace std;

/** gets the color, given the dwell */
void dwell_color(int *r, int *g, int *b, int dwell, unsigned int CA_MAXDWELL);

float doExhaustive(int *d_dwells, unsigned int w, unsigned int h,
                   complex bottomLeftCorner, complex upperRightCorner,
                   unsigned int CA_MAXDWELL) {
    float elapsedTime;
    dim3 blockSize(BSX, BSY), gridSize(divup(w, blockSize.x), divup(h, blockSize.y));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < REPEATS; i++) {
        mandelbrot_k<<<gridSize, blockSize>>>(d_dwells, w, h, bottomLeftCorner,
                                              upperRightCorner, CA_MAXDWELL);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}


float doAdaptiveSerialKernels(int *d_dwells, unsigned int w, unsigned int h,
                              complex bottomLeftCorner, complex upperRightCorner,
                              unsigned int G0, unsigned int r,
                              unsigned int CA_MAXDWELL, unsigned int B,
                              unsigned int MAX_DEPTH) {

    int *h_offsets, *d_offsets1, *d_offsets2; // OLT
    unsigned int *h_OLTSize, *d_OLTSize;      // OLT SIZE

    float elapsedTime = 0;

    h_OLTSize = (unsigned int *)malloc(sizeof(int));
    *h_OLTSize = G0 * G0 * r * r * 2;

    cucheck(cudaMalloc(&d_OLTSize, sizeof(int)));

    size_t initialOLTSize = *h_OLTSize * sizeof(int);

    h_offsets = (int *)malloc(*h_OLTSize * sizeof(int));

    for (int i = 0; i < G0 * G0 * 2; i += 2) {
        h_offsets[i] = ((i / 2) % G0) * (w / G0);
        h_offsets[i + 1] = ((i / 2) / G0) * (w / G0);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 1 KERNEL
    for (int i = 0; i < REPEATS; i++) {
        for (int j = 0; j < G0 * G0 * 2; j += 2) {
            h_offsets[j] = ((j / 2) % G0) * (w / G0);
            h_offsets[j + 1] = ((j / 2) / G0) * (w / G0);
            // printf("Offsets Iniciales: (%i) - %i, %i\n", i/2, h_offsets[i],
            // h_offsets[i+1]);
        }
        *h_OLTSize = 1;
        cucheck(cudaMalloc((void **)&d_offsets1, initialOLTSize));
        cucheck(cudaMalloc((void **)&d_offsets2, initialOLTSize));

        cucheck(cudaMemcpy(d_offsets1, h_offsets, initialOLTSize,
                           cudaMemcpyHostToDevice));
        cucheck(cudaMemset(d_OLTSize, 0, sizeof(int)));

        float iterationTime = 0;
        cudaEventRecord(start, 0);

        AdaptiveSerialKernels(d_dwells, h_OLTSize, d_OLTSize, d_offsets1, d_offsets2,
                              w, h, bottomLeftCorner, upperRightCorner,
                              w / G0, 1, G0, r, CA_MAXDWELL,
                              B, MAX_DEPTH);
        cucheck(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&iterationTime, start, stop); // that's our time!
        elapsedTime += iterationTime;

        //cucheck(cudaFree(d_offsets1));
        //cucheck(cudaFree(d_offsets2));
    }

    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}

float doAdaptiveSerialKernelsNEW(int *d_dwells, unsigned int w, unsigned int h,
                              complex bottomLeftCorner, complex upperRightCorner,
                              unsigned int G0, unsigned int r,
                              unsigned int CA_MAXDWELL, unsigned int B,
                              unsigned int MAX_DEPTH) {

    int *h_offsets, *d_offsets1, *d_offsets2; // OLT
    unsigned int *h_OLTSize, *d_OLTSize;      // OLT SIZE

    float elapsedTime = 0;

    h_OLTSize = (unsigned int *)malloc(sizeof(int));
    *h_OLTSize = G0 * G0 * r * r * 2;

    cucheck(cudaMalloc(&d_OLTSize, sizeof(int)));

    size_t initialOLTSize = *h_OLTSize * sizeof(int);

    h_offsets = (int *)malloc(*h_OLTSize * sizeof(int));

    for (int i = 0; i < G0 * G0 * 2; i += 2) {
        h_offsets[i] = ((i / 2) % G0) * (w / G0);
        h_offsets[i + 1] = ((i / 2) / G0) * (w / G0);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 1 KERNEL
    for (int i = 0; i < REPEATS; i++) {
        for (int j = 0; j < G0 * G0 * 2; j += 2) {
            h_offsets[j] = ((j / 2) % G0) * (w / G0);
            h_offsets[j + 1] = ((j / 2) / G0) * (w / G0);
            // printf("Offsets Iniciales: (%i) - %i, %i\n", i/2, h_offsets[i],
            // h_offsets[i+1]);
        }
        *h_OLTSize = 1;
        cucheck(cudaMalloc((void **)&d_offsets1, initialOLTSize));
        cucheck(cudaMalloc((void **)&d_offsets2, initialOLTSize));

        cucheck(cudaMemcpy(d_offsets1, h_offsets, initialOLTSize,
                           cudaMemcpyHostToDevice));
        cucheck(cudaMemset(d_OLTSize, 0, sizeof(int)));

        float iterationTime = 0;
        cudaEventRecord(start, 0);

        AdaptiveSerialKernelsNEW(d_dwells, h_OLTSize, d_OLTSize, d_offsets1, d_offsets2,
                              w, h, bottomLeftCorner, upperRightCorner,
                              w / G0, 1, G0, r, CA_MAXDWELL,
                              B, MAX_DEPTH);
        cucheck(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&iterationTime, start, stop); // that's our time!
        elapsedTime += iterationTime;

        //cucheck(cudaFree(d_offsets1));
        //cucheck(cudaFree(d_offsets2));
    }

    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}

float doDynamicParallelism(int *d_dwells, unsigned int w, unsigned int h,
                           complex bottomLeftCorner, complex upperRightCorner,
                           unsigned int G0, unsigned int r,
                           unsigned int CA_MAXDWELL, unsigned int B,
                           unsigned int MAX_DEPTH) {
    float elapsedTime;
    dim3 blockSize(BSX, BSY), gridSize(G0, G0);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < REPEATS; i++) {
        mandelbrot_block_k<<<gridSize, blockSize>>>(
            d_dwells, w, h, bottomLeftCorner, upperRightCorner, 0, 0,
            w / G0, 1, r, CA_MAXDWELL, B, MAX_DEPTH);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}

float doGridLines(int *d_dwells, unsigned int w, unsigned int h,
                              complex bottomLeftCorner, complex upperRightCorner,
                              unsigned int G0, unsigned int r,
                              unsigned int CA_MAXDWELL, unsigned int B,
                              unsigned int MAX_DEPTH) {

    int *h_offsets, *d_offsets1, *d_offsets2; // OLT
    unsigned int *h_OLTSize, *d_OLTSize;      // OLT SIZE

    float elapsedTime = 0;

    h_OLTSize = (unsigned int *)malloc(sizeof(int));
    *h_OLTSize = G0 * G0 * r * r * 2;

    cucheck(cudaMalloc(&d_OLTSize, sizeof(int)));

    size_t initialOLTSize = *h_OLTSize * sizeof(int);

    h_offsets = (int *)malloc(*h_OLTSize * sizeof(int));

    for (int i = 0; i < G0 * G0 * 2; i += 2) {
        h_offsets[i] = ((i / 2) % G0) * (w / G0);
        h_offsets[i + 1] = ((i / 2) / G0) * (w / G0);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 1 KERNEL
    for (int i = 0; i < REPEATS; i++) {
        for (int j = 0; j < G0 * G0 * 2; j += 2) {
            h_offsets[j] = ((j / 2) % G0) * (w / G0);
            h_offsets[j + 1] = ((j / 2) / G0) * (w / G0);
            // printf("Offsets Iniciales: (%i) - %i, %i\n", i/2, h_offsets[i],
            // h_offsets[i+1]);
        }
        *h_OLTSize = 1;
        cucheck(cudaMalloc((void **)&d_offsets1, initialOLTSize));
        cucheck(cudaMalloc((void **)&d_offsets2, initialOLTSize));

        cucheck(cudaMemcpy(d_offsets1, h_offsets, initialOLTSize,
                           cudaMemcpyHostToDevice));
        cucheck(cudaMemset(d_OLTSize, 0, sizeof(int)));

        float iterationTime = 0;
        cudaEventRecord(start, 0);

        GridLines(d_dwells, h_OLTSize, d_OLTSize, d_offsets1, d_offsets2,
                              w, h, bottomLeftCorner, upperRightCorner,
                              w / G0, 1, G0, r, CA_MAXDWELL,
                              B, MAX_DEPTH);
        cucheck(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&iterationTime, start, stop); // that's our time!
        elapsedTime += iterationTime;

        //cucheck(cudaFree(d_offsets1));
        //cucheck(cudaFree(d_offsets2));
    }
    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}

int main(int argc, char **argv) {

    check_args_info(argc);

    char approach = atoi(argv[1]);
    unsigned int W = stoi(argv[2]);
    unsigned int H = stoi(argv[3]);
    float rmin = atof(argv[4]);
    float rmax = atof(argv[5]);
    float cmin = atof(argv[6]);
    float cmax = atof(argv[7]);

    int CA_MAXDWELL = atoi(argv[8]);
    int B = atoi(argv[9]);
    int G0 = atoi(argv[10]);
    int r = atoi(argv[11]);
    int MAX_DEPTH = atoi(argv[12]);
    string fileName = argv[13];

    float elapsedTime;
    int *h_dwells;
    int *d_dwells;

    #ifdef VERBOSE
    printf("Grid %i x %i --> %.2f GiB\n", W, H, (float)(sizeof(unsigned int) * W * H)/(1024*1024*1024));
    #endif

    complex bottomLeftCorner = complex(rmin, cmin);
    complex upperRightCorner = complex(rmax, cmax);
    size_t dwell_sz = (size_t)W * H * sizeof(int);
    cucheck(cudaMalloc((void **)&d_dwells, dwell_sz));
    cudaDeviceSynchronize();
    h_dwells = (int *)malloc(dwell_sz);

    switch (approach) {
    case 0:
        elapsedTime = doExhaustive(d_dwells, W, H, bottomLeftCorner, upperRightCorner, CA_MAXDWELL);
        break;
    case 1:
        elapsedTime = doDynamicParallelism(d_dwells, W, H, bottomLeftCorner, upperRightCorner, G0, r, CA_MAXDWELL, B, MAX_DEPTH);
        break;
    case 2:
        elapsedTime = doAdaptiveSerialKernels( d_dwells, W, H, bottomLeftCorner, upperRightCorner, G0, r, CA_MAXDWELL, B, MAX_DEPTH);
        break;
    case 3:
        elapsedTime = doAdaptiveSerialKernelsNEW( d_dwells, W, H, bottomLeftCorner, upperRightCorner, G0, r, CA_MAXDWELL, B, MAX_DEPTH);
        break;
    default:
        cout << approach << " is not a valid approach." << endl;
        exit(-2);
    }

    cudaDeviceSynchronize();
    cucheck(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));
    if (fileName != "none"){
        string fractalFileName = fileName + string(".png");
        save_image(fractalFileName.c_str(), h_dwells, W, H, CA_MAXDWELL, SAVE_FRACTAL);
    }
    printf("%i, %i, %i, %i, %i, %i, %i, %i, %i, %f\n", approach, BSX, BSY, W, H, CA_MAXDWELL, MAX_DEPTH, r, B, elapsedTime);

    #ifdef GRIDLINES
        elapsedTime = doGridLines( d_dwells, W, H, bottomLeftCorner, upperRightCorner, G0, r, CA_MAXDWELL, B, MAX_DEPTH);
        cudaDeviceSynchronize();
        cucheck(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));
        if (fileName != "none"){
            string gridFileName = fileName + string("-gridlines.png");
            save_image( gridFileName.c_str(), h_dwells, W, H, CA_MAXDWELL, SAVE_GRIDLINES);
        }
    #endif
    exit(0);

}
