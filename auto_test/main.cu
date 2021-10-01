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
#include "bruteForce.cuh"
#include "complex.cuh"
#include "dynamicParallelism.cuh"
#include "macros.cuh"
#include "mandelbrotHelper.cuh"


using namespace std;

/** gets the color, given the dwell */
void dwell_color(int *r, int *g, int *b, int dwell, unsigned int MAX_DWELL);

float doBruteForce(int *d_dwells, unsigned int w, unsigned int h,
                   complex bottomLeftCorner, complex upperRightCorner,
                   unsigned int MAX_DWELL) {
    float elapsedTime;
    dim3 blockSize(BSX, BSY), gridSize(divup(w, blockSize.x), divup(h, blockSize.y));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < REPEATS; i++) {
        mandelbrot_k<<<gridSize, blockSize>>>(d_dwells, w, h, bottomLeftCorner,
                                              upperRightCorner, MAX_DWELL);
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
                              unsigned int INIT_SUBDIV, unsigned int SUBDIV,
                              unsigned int MAX_DWELL, unsigned int MIN_SIZE,
                              unsigned int MAX_DEPTH) {

    int *h_offsets, *d_offsets1, *d_offsets2; // OLT
    unsigned int *h_OLTSize, *d_OLTSize;      // OLT SIZE

    float elapsedTime = 0;

    h_OLTSize = (unsigned int *)malloc(sizeof(int));
    *h_OLTSize = INIT_SUBDIV * INIT_SUBDIV * SUBDIV * SUBDIV * 2;

    cucheck(cudaMalloc(&d_OLTSize, sizeof(int)));

    size_t initialOLTSize = *h_OLTSize * sizeof(int);

    h_offsets = (int *)malloc(*h_OLTSize * sizeof(int));

    for (int i = 0; i < INIT_SUBDIV * INIT_SUBDIV * 2; i += 2) {
        h_offsets[i] = ((i / 2) % INIT_SUBDIV) * (w / INIT_SUBDIV);
        h_offsets[i + 1] = ((i / 2) / INIT_SUBDIV) * (w / INIT_SUBDIV);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 1 KERNEL
    for (int i = 0; i < REPEATS; i++) {
        for (int j = 0; j < INIT_SUBDIV * INIT_SUBDIV * 2; j += 2) {
            h_offsets[j] = ((j / 2) % INIT_SUBDIV) * (w / INIT_SUBDIV);
            h_offsets[j + 1] = ((j / 2) / INIT_SUBDIV) * (w / INIT_SUBDIV);
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
                              w / INIT_SUBDIV, 1, INIT_SUBDIV, SUBDIV, MAX_DWELL,
                              MIN_SIZE, MAX_DEPTH);
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
                           unsigned int INIT_SUBDIV, unsigned int SUBDIV,
                           unsigned int MAX_DWELL, unsigned int MIN_SIZE,
                           unsigned int MAX_DEPTH) {
    float elapsedTime;
    dim3 blockSize(BSX, BSY), gridSize(INIT_SUBDIV, INIT_SUBDIV);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < REPEATS; i++) {
        mandelbrot_block_k<<<gridSize, blockSize>>>(
            d_dwells, w, h, bottomLeftCorner, upperRightCorner, 0, 0,
            w / INIT_SUBDIV, 1, SUBDIV, MAX_DWELL, MIN_SIZE, MAX_DEPTH);
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
                              unsigned int INIT_SUBDIV, unsigned int SUBDIV,
                              unsigned int MAX_DWELL, unsigned int MIN_SIZE,
                              unsigned int MAX_DEPTH) {

    int *h_offsets, *d_offsets1, *d_offsets2; // OLT
    unsigned int *h_OLTSize, *d_OLTSize;      // OLT SIZE

    float elapsedTime = 0;

    h_OLTSize = (unsigned int *)malloc(sizeof(int));
    *h_OLTSize = INIT_SUBDIV * INIT_SUBDIV * SUBDIV * SUBDIV * 2;

    cucheck(cudaMalloc(&d_OLTSize, sizeof(int)));

    size_t initialOLTSize = *h_OLTSize * sizeof(int);

    h_offsets = (int *)malloc(*h_OLTSize * sizeof(int));

    for (int i = 0; i < INIT_SUBDIV * INIT_SUBDIV * 2; i += 2) {
        h_offsets[i] = ((i / 2) % INIT_SUBDIV) * (w / INIT_SUBDIV);
        h_offsets[i + 1] = ((i / 2) / INIT_SUBDIV) * (w / INIT_SUBDIV);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 1 KERNEL
    for (int i = 0; i < REPEATS; i++) {
        for (int j = 0; j < INIT_SUBDIV * INIT_SUBDIV * 2; j += 2) {
            h_offsets[j] = ((j / 2) % INIT_SUBDIV) * (w / INIT_SUBDIV);
            h_offsets[j + 1] = ((j / 2) / INIT_SUBDIV) * (w / INIT_SUBDIV);
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
                              w / INIT_SUBDIV, 1, INIT_SUBDIV, SUBDIV, MAX_DWELL,
                              MIN_SIZE, MAX_DEPTH);
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

    if (argc != 14) {
        cout << "Error. Execute as ./mandelbrot <Approach> <WIDTH> <HEIGHT> "
                "<xlim_min> <xlim_max> <ylim_min> <ylim_max> <MAX_DWELL> <MIN_SIZE> "
                "<INIT_SUBDIV> <SUBDIV> <MAX_DEPTH> <filename>.png"
             << endl;
        cout << "Apporach:  0 - Brute Force" << endl;
        cout << "           1 - Dynamic Parallelism" << endl;
        cout << "           2 - Adaptive Serial Kernels" << endl;
        cout << "Default values:" << endl;
        cout << "   xlim -------- [-1.5, 0.5] - x range in the plane (real values)."
             << endl;
        cout << "   ylim -------- [-1.0, 1.0] - y range in the plane (imaginary "
                "values)."
             << endl;
        cout << "   MAX_DWELL ----------- 512 - Maximum numbers of iterarions per "
                "element."
             << endl;
        cout << "   MIN_SIZE ------------- 32 - Minimum size of the region to force "
                "BF (powers of 2)."
             << endl;
        cout << "   INIT_SUBDIV ---------- 32 - Initial numbers of regions (powers "
                "of 2)."
             << endl;
        cout << "   SUBDIV ---------------- 4 - Subdivide factor (powers of 2)."
             << endl;
        cout << "   MAX_DEPTH ------------- 5 - Maximum recursion depth." << endl;
        cout << "   filename  -------- 'none' - Use 'none' to skip file output." << endl;
        exit(-1);
    }
    char approach = atoi(argv[1]);
    unsigned int WIDTH = stoi(argv[2]);
    unsigned int HEIGHT = stoi(argv[3]);
    float xlim_min = atof(argv[4]);
    float xlim_max = atof(argv[5]);
    float ylim_min = atof(argv[6]);
    float ylim_max = atof(argv[7]);

    int MAX_DWELL = atoi(argv[8]);
    int MIN_SIZE = atoi(argv[9]);
    int INIT_SUBDIV = atoi(argv[10]);
    int SUBDIV = atoi(argv[11]);
    int MAX_DEPTH = atoi(argv[12]);
    string fileName = argv[13];

    float elapsedTime;
    int *h_dwells;
    int *d_dwells;

    size_t memsize = sizeof(unsigned int) * WIDTH * HEIGHT;
    #ifdef VERBOSE
    printf("Grid %i x %i --> %.2f GiB\n", WIDTH, HEIGHT, (float)memsize/(1024*1024*1024));
    #endif

    complex bottomLeftCorner = complex(xlim_min, ylim_min);
    complex upperRightCorner = complex(xlim_max, ylim_max);

    size_t dwell_sz = (size_t)WIDTH * HEIGHT * sizeof(int);
    cucheck(cudaMalloc((void **)&d_dwells, dwell_sz));
    cudaDeviceSynchronize();

    h_dwells = (int *)malloc(dwell_sz);


    switch (approach) {
    case 0:
        elapsedTime = doBruteForce(d_dwells, WIDTH, HEIGHT, bottomLeftCorner,
                                   upperRightCorner, MAX_DWELL);
        break;
    case 1:
        elapsedTime = doDynamicParallelism(d_dwells, WIDTH, HEIGHT, bottomLeftCorner,
                                           upperRightCorner, INIT_SUBDIV, SUBDIV,
                                           MAX_DWELL, MIN_SIZE, MAX_DEPTH);
        break;
    case 2:
        elapsedTime = doAdaptiveSerialKernels(
            d_dwells, WIDTH, HEIGHT, bottomLeftCorner, upperRightCorner, INIT_SUBDIV,
            SUBDIV, MAX_DWELL, MIN_SIZE, MAX_DEPTH);
        break;
    default:
        cout << approach << " is not a valid approach." << endl;
        exit(-2);
    }

    cudaDeviceSynchronize();
    cucheck(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));
    if (fileName != "none"){
        string fractalFileName = fileName + string(".png");
        save_image(fractalFileName.c_str(), h_dwells, WIDTH, HEIGHT, MAX_DWELL, SAVE_FRACTAL);
    }
    printf("%i, %i, %i, %i, %i, %i, %i, %i, %i, %f\n", approach, BSX, BSY, WIDTH, HEIGHT, MAX_DWELL, MAX_DEPTH, SUBDIV, MIN_SIZE, elapsedTime);

    #ifdef GRIDLINES
        elapsedTime = doGridLines( d_dwells, WIDTH, HEIGHT, bottomLeftCorner, upperRightCorner, INIT_SUBDIV, SUBDIV, MAX_DWELL, MIN_SIZE, MAX_DEPTH);
        cudaDeviceSynchronize();
        cucheck(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));
        if (fileName != "none"){
            string gridFileName = fileName + string("-gridlines.png");
            save_image( gridFileName.c_str(), h_dwells, WIDTH, HEIGHT, MAX_DWELL, SAVE_GRIDLINES);
        }
    #endif
    exit(0);

} // main
