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

#include "stats.cuh"
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
const char* approachStr[4] = {"Ex", "DP", "ASK-v1", "ASK-v2"};

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
    int g0 = atoi(argv[10]);
    int r = atoi(argv[11]);
    int MAX_DEPTH = atoi(argv[12]);
    string fileName = argv[13];

    int *h_dwells;
    int *d_dwells;


    complex bottomLeftCorner = complex(rmin, cmin);
    complex upperRightCorner = complex(rmax, cmax);
    size_t dwell_sz = (size_t)W * H * sizeof(int);
    #ifdef VERBOSE
        float domainGBytes = (float)(sizeof(unsigned int) * W * H)/(1024*1024*1024);
        printf("\nGrid..............................................%i x %i (%.2f GiB)\n", W, H, domainGBytes);
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
    #ifdef VERBOSE
        printf("%s (REPEATS=%3i, REALIZATIONS=%3i)............", approachStr[approach], REPEATS, REALIZATIONS); fflush(stdout);
    #endif
    statistics stat = doTest(approach, d_dwells, W, H, bottomLeftCorner, upperRightCorner, g0, r, CA_MAXDWELL, B, MAX_DEPTH);
    cudaDeviceSynchronize();
    #ifdef VERBOSE
        printf("done: %f secs (stErr %f%%)\n", stat.mean, 100.0*stat.sterr/stat.mean); fflush(stdout);
    #endif




    // ----------------------------
    // 3) copy domain back to Host
    // ----------------------------
    #ifdef VERBOSE
        printf("cudaMemcpy: Host <-- Dev (%5.2f GiB)..............", domainGBytes); fflush(stdout);
    #endif
    cucheck(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));
    #ifdef VERBOSE
        printf("done\n"); fflush(stdout);
    #endif



    // -------------------
    // 4) Export Fractal Image
    // -------------------
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
    // Export gridlines image (does computation)
    // -----------------------
    #ifdef GRIDLINES
        // ------------------
        // compute gridlines
        // ------------------
        #ifdef VERBOSE
            printf("GridLines........................................."); fflush(stdout);
        #endif
        float gridTime = doGridLines( d_dwells, W, H, bottomLeftCorner, upperRightCorner, g0, r, CA_MAXDWELL, B, MAX_DEPTH);
        cudaDeviceSynchronize();
        #ifdef VERBOSE
            printf("done: %f secs\n", gridTime); fflush(stdout);
        #endif



        // ----------------------------
        // copy gridlines back to host
        // ----------------------------
        #ifdef VERBOSE
            printf("cudaMemcpy: Host <-- Dev (%5.2f GiB)..............", domainGBytes); fflush(stdout);
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
    #ifdef VERBOSE
        printf("\n");
    #endif

    //printf("%i,%s,   %i, %i,   %i, %i,   %i, %i,   %i, %i, %i,   %f, %f, %f, %f\n", 
    //        approach, approachStr[approach], BSX, BSY, W, H, CA_MAXDWELL, MAX_DEPTH, g0, r, B, 
    //        stat.mean, stat.stdev, stat.sterr, 100.0*stat.sterr/stat.mean);
    printf("%i,%f,%f,%f,%f", approach, stat.mean, stat.stdev, stat.sterr, 100.0*stat.sterr/stat.mean);
    exit(EXIT_SUCCESS);
}
