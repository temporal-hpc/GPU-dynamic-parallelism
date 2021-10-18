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
#include "doTests.cuh"


using namespace std;

/** gets the color, given the dwell */
void dwell_color(int *r, int *g, int *b, int dwell, unsigned int CA_MAXDWELL);

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


    complex bottomLeftCorner = complex(rmin, cmin);
    complex upperRightCorner = complex(rmax, cmax);
    size_t dwell_sz = (size_t)W * H * sizeof(int);
    #ifdef VERBOSE
        float domainGBytes = (float)(sizeof(unsigned int) * W * H)/(1024*1024*1024);
        printf("Grid......................................%i x %i (%.2f GiB)\n", W, H, domainGBytes);
    #endif


    // ---------------------
    // 1) memory allocation
    // ---------------------
    cucheck(cudaMalloc((void **)&d_dwells, dwell_sz));
    cudaDeviceSynchronize();
    h_dwells = (int *)malloc(dwell_sz);




    // ---------------------
    // 2) GPU Compute
    // ---------------------
    switch (approach) {
    case 0:
        #ifdef VERBOSE
            printf("Ex (REPEATS=%3i)..........................", REPEATS); fflush(stdout);
        #endif
        elapsedTime = doExhaustive(d_dwells, W, H, bottomLeftCorner, upperRightCorner, CA_MAXDWELL);
        break;
    case 1:
        #ifdef VERBOSE
            printf("DP (REPEATS=%3i)..........................", REPEATS); fflush(stdout);
        #endif
        elapsedTime = doDynamicParallelism(d_dwells, W, H, bottomLeftCorner, upperRightCorner, G0, r, CA_MAXDWELL, B, MAX_DEPTH);
        break;
    case 2:
        #ifdef VERBOSE
            printf("ASK-v1 (REPEATS=%3i)......................", REPEATS); fflush(stdout);
        #endif
        elapsedTime = doAdaptiveSerialKernels( d_dwells, W, H, bottomLeftCorner, upperRightCorner, G0, r, CA_MAXDWELL, B, MAX_DEPTH);
        break;
    case 3:
        #ifdef VERBOSE
            printf("ASK-v2 (REPEATS=%3i)......................", REPEATS); fflush(stdout);
        #endif
        elapsedTime = doAdaptiveSerialKernelsNEW( d_dwells, W, H, bottomLeftCorner, upperRightCorner, G0, r, CA_MAXDWELL, B, MAX_DEPTH);
        break;
    default:
        cout << approach << " is not a valid approach." << endl;
        exit(-2);
    }
    cudaDeviceSynchronize();
    #ifdef VERBOSE
        printf("done: %f secs\n", elapsedTime); fflush(stdout);
    #endif




    // ----------------------------
    // 3) copy domain back to Host
    // ----------------------------
    #ifdef VERBOSE
        printf("cudaMemcpy: Host <- Dev (%5.2f GiB).......", domainGBytes); fflush(stdout);
    #endif
    cucheck(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));
    #ifdef VERBOSE
        printf("done\n"); fflush(stdout);
    #endif



    // -------------------
    // 4) Export Fractal Image
    // -------------------

    // fractal image
    if (fileName != "none"){
        string fractalFileName = fileName + string(".png");
        #ifdef VERBOSE
            printf("Saving %s.................................", fractalFileName.c_str()); fflush(stdout);
        #endif
        save_image(fractalFileName.c_str(), h_dwells, W, H, CA_MAXDWELL, SAVE_FRACTAL);
        #ifdef VERBOSE
            printf("done\n"); fflush(stdout);
        #endif
    }


    // -----------------------
    // Export gridlines image
    // -----------------------
    #ifdef GRIDLINES
        // ------------------
        // compute gridlines
        // ------------------
        #ifdef VERBOSE
            printf("GridLines................................."); fflush(stdout);
        #endif
        float gridTime = doGridLines( d_dwells, W, H, bottomLeftCorner, upperRightCorner, G0, r, CA_MAXDWELL, B, MAX_DEPTH);
        cudaDeviceSynchronize();
        #ifdef VERBOSE
            printf("done: %f secs\n", gridTime); fflush(stdout);
        #endif



        // ----------------------------
        // copy gridlines back to host
        // ----------------------------
        #ifdef VERBOSE
            printf("cudaMemcpy: Host <-- Dev (%5.2f GiB)......", domainGBytes); fflush(stdout);
        #endif
        cucheck(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));
        #ifdef VERBOSE
            printf("done\n"); fflush(stdout);
        #endif



        // ---------------------
        // save gridlines image
        // ---------------------
        if (fileName != "none"){
            string gridFileName = fileName + string("-gridlines.png");
            #ifdef VERBOSE
                printf("Saving %s......", gridFileName.c_str()); fflush(stdout);
            #endif
            save_image( gridFileName.c_str(), h_dwells, W, H, CA_MAXDWELL, SAVE_GRIDLINES);
            #ifdef VERBOSE
                printf("done\n"); fflush(stdout);
            #endif
        }
    #endif
    printf("%i, %i, %i, %i, %i, %i, %i, %i, %i, %f\n", approach, BSX, BSY, W, H, CA_MAXDWELL, MAX_DEPTH, r, B, elapsedTime);
    exit(EXIT_SUCCESS);

}
