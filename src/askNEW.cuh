#pragma once

#include "complex.cuh"
#include "macros.cuh"
#include "mandelbrotHelper.cuh"

__global__ void ASKNEW(unsigned int *d_ns, unsigned int *d_nbf, int *d_offs1, int *d_offs2, int *dwells,
                    int w, int h, complex cmin, complex cmax, int d, int depth,
                    unsigned int SUBDIV, unsigned int MAX_DWELL,
                    unsigned int MIN_SIZE, unsigned int MAX_DEPTH,
                    unsigned int SUBDIV_ELEMS, unsigned int SUBDIV_ELEMS2,
                    unsigned int SUBDIV_ELEMSP, unsigned int SUBDIV_ELEMSX, int OLTSize) {
    // check whether all boundary pixels have the same dwell
    unsigned int use = blockIdx.x * SUBDIV_ELEMS2 + (blockIdx.z * gridDim.y + blockIdx.y) * 2;

    const unsigned int x0 = d_offs1[use];
    const unsigned int y0 = d_offs1[use + 1];

    __shared__ unsigned int off_index;
    __shared__ unsigned int off_index_work;

    // --------------
    // -- EXPLORACION
    // --------------
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int bs = blockDim.x * blockDim.y;
    int comm_dwell = NEUT_DWELL;
    // for all boundary pixels, distributed across threads
    for (int r = tid; r < d; r += bs) {
        // for each boundary: b = 0 is east, then counter-clockwise
        for (int b = 0; b < 4; b++) {
            unsigned int x = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
            unsigned int y = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
            int dwell = pixel_dwell(w, h, cmin, cmax, x, y, MAX_DWELL);
            comm_dwell = same_dwell(comm_dwell, dwell, MAX_DWELL);
        }
    } // for all boundary pixels
    // reduce across threads in the block
    __shared__ int ldwells[BSX * BSY];
    int nt = min(d, BSX * BSY);
    if (tid < nt)
        ldwells[tid] = comm_dwell;
    __syncthreads();
    for (; nt > 1; nt /= 2) {
        if (tid < nt / 2)
            ldwells[tid] =
                same_dwell(ldwells[tid], ldwells[tid + nt / 2], MAX_DWELL);
        __syncthreads();
    }
    comm_dwell = ldwells[0];

    __syncthreads();
    // ----------------------
    // SUBDIVISION
    // ----------------------
    //printf("%i - %i - %i - %i, %i\n", comm_dwell, depth, MAX_DEPTH, d, MIN_SIZE);
    if (comm_dwell != DIFF_DWELL) {
        // LLENAR DE ATRAS PA ADELANTE
        if (tid==0){
            dwells[y0*(size_t)w + x0] = comm_dwell;
            off_index_work = atomicAdd(d_nbf, 1);
        }
        __syncthreads();
        if (tid < 2) {
            // cada region (bloque) escribe su par (x0,y0) en la OLT de atras hacia adelante - no es necesaria
            // sincronizacion por bloque pues cada bloque tiene resevado su espacio (por blockIdx)
            d_offs2[OLTSize - ((off_index_work * 2) + tid) - 1] = x0 * ((tid + 1) & 1) + y0 * (tid & 1);
        }
    } else if (depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
        // SUBDIVIDIR
        //printf("asd\n");
        if (tid == 0) {
        //printf("Block (%i, %i, %i) doing SUBDIV\n", blockIdx.x, blockIdx.y, blockIdx.z);
            off_index = atomicAdd(d_ns, 1);
        }
        __syncthreads();
        if (tid < SUBDIV_ELEMS2) {
            d_offs2[(off_index * SUBDIV_ELEMS2) + tid] =
                (x0 + ((tid >> 1) & SUBDIV_ELEMSX) * (d / SUBDIV)) *
                    ((tid + 1) & 1) +
                (y0 + (tid >> SUBDIV_ELEMSP) * (d / SUBDIV)) * (tid & 1);
        }
    } else {
        //BF
        // LLENAR DE ATRAS PA ADELANTE
        if (tid==0){
            off_index_work = atomicAdd(d_nbf, 1);
        }
        __syncthreads();
        if (tid < 2) {
            // cada region escribe su par (x,y)
            d_offs2[OLTSize - ((off_index_work * 2) + tid) - 1] = x0 * ((tid + 1) & 1) + y0 * (tid & 1);
        }
    }
}

/** the kernel to fill the image region with a specific dwell value */
__global__ void doT(int *dwells, size_t w, int d, int *OLT, int OLTSize) {

    // offset = OLTSize - numRegionesDerecha
    // acceso -> offset + blockIdx.*2
    // [ [Ix1 Iy1 Ix2 Iy2..  ]  ....  [Dx Dy] [Dx Dy] ]
    //                              numRegionesDerecha
    // gridDim.x
    // unsigned int x0 = OLT[(OLTSize - gridDim.x*2) + blockIdx.x*2];

    //unsigned int x0 = OLT[OLTSize - (blockIdx.x*2) - 1];
    //unsigned int y0 = OLT[OLTSize - (blockIdx.x*2 + 1) - 1];
    unsigned int x0 = OLT[(OLTSize - gridDim.x*2) + blockIdx.x*2 + 1];
    unsigned int y0 = OLT[(OLTSize - gridDim.x*2) + (blockIdx.x*2)];

    unsigned int x = threadIdx.x + blockIdx.y * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.z * blockDim.y;
    __shared__ int dwell;
    if (threadIdx.x+threadIdx.y == 0){
        dwell = dwells[y0*(size_t)w + x0];
    }
    __syncthreads();
    if (x < d && y < d) {
        x += x0, y += y0;
        // if (dwells[y * w + x] != 666)
        dwells[y * (size_t)w + x] = dwell;
    }
}

/** the kernel to fill in per-pixel values of the portion of the Mandelbrot set
 */
__global__ void doBruteForce(int *dwells, unsigned int w, unsigned int h,
                                   complex cmin, complex cmax,
                                   int d, unsigned int MAX_DWELL, int *OLT, int OLTSize) {

    //unsigned int x0 = OLT[OLTSize - (blockIdx.x*2) - 1];
    //unsigned int y0 = OLT[OLTSize - (blockIdx.x*2 + 1) - 1];
    unsigned int x0 = OLT[(OLTSize - gridDim.x*2) + blockIdx.x*2 + 1];
    unsigned int y0 = OLT[(OLTSize - gridDim.x*2) + (blockIdx.x*2)];

    unsigned int x = threadIdx.x + blockIdx.y * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.z * blockDim.y;
    if (x < d && y < d) {
        x += x0, y += y0;
        // if (dwells[y * w + x] != 666)
        dwells[y * (size_t)w + x] = pixel_dwell(w, h, cmin, cmax, x, y, MAX_DWELL);
    }
}

void AdaptiveSerialKernelsNEW(int *dwells, unsigned int *h_nextSize,
                           unsigned int *d_nextSize, int **d_offsets1,
                           int **d_offsets2, int w, int h, complex cmin, complex cmax,
                           int d, int depth, unsigned int INIT_SUBDIV,
                           unsigned int SUBDIV, unsigned int MAX_DWELL,
                           unsigned int MIN_SIZE, unsigned int MAX_DEPTH) {

    dim3 b(BSX, BSY, 1), g(1, INIT_SUBDIV, INIT_SUBDIV);
    unsigned int *d_nbf;
    cucheck(cudaMalloc(&d_nbf, sizeof(int)));
    (cudaMemset(d_nbf, 0, sizeof(int)));

    unsigned int SUBDIV_ELEMS = SUBDIV * SUBDIV;
    unsigned int SUBDIV_ELEMS2 = SUBDIV_ELEMS * 2;
    unsigned int SUBDIV_ELEMSP = log2(SUBDIV) + 1;
    unsigned int SUBDIV_ELEMSX = SUBDIV - 1;
    size_t OLTSize = INIT_SUBDIV*INIT_SUBDIV*SUBDIV*SUBDIV*2;
    #ifdef DEBUG
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        printf("\n[level %2i].....", 1); fflush(stdout);
    #endif

    ASKNEW<<<g, b>>>(d_nextSize, d_nbf, *d_offsets1, *d_offsets2, dwells, h, w, cmin, cmax, d,
                  depth, SUBDIV, MAX_DWELL, MIN_SIZE, MAX_DEPTH, SUBDIV_ELEMS,
                  SUBDIV_ELEMS2, SUBDIV_ELEMSP, SUBDIV_ELEMSX, OLTSize);
    cucheck(cudaDeviceSynchronize());
    cudaMemcpy(h_nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost);
    #ifdef DEBUG
        dim3 gold = g;
    #endif
    if (*h_nextSize < g.x*g.y*g.z){
        g = dim3((g.x*g.y*g.z)-*h_nextSize, (d + b.x - 1)/b.x, (d + b.y - 1)/b.y);
        if (2 < MAX_DEPTH && d/SUBDIV > MIN_SIZE){
            doT<<<g, b>>>(dwells, w, d, *d_offsets2, OLTSize);
        } else {
            doBruteForce<<<g, b>>>(dwells, w, h, cmin, cmax, d, MAX_DWELL, *d_offsets2, OLTSize);
        }
    }
    #ifdef DEBUG
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop); // that's our time!
        //printf("[level %2i] h_nextSize  %i \n", 1, *h_nextSize); 
        printf("done %f secs --> P=%f r = %i x %i (grid %8i x %i x %i = %8i --> %i subdivided)\n", time/1000.0f, *h_nextSize/(float)(gold.x*INIT_SUBDIV*INIT_SUBDIV), d, d, gold.x, INIT_SUBDIV, INIT_SUBDIV, gold.x*INIT_SUBDIV*INIT_SUBDIV, *h_nextSize);
        cudaEventRecord(start, 0);
    #endif

    // LOOP
    for (int i = depth + 1; i < MAX_DEPTH && d / SUBDIV > MIN_SIZE; i++) {
        cucheck(cudaDeviceSynchronize());

        std::swap(*d_offsets1, *d_offsets2);

        cudaFree(*d_offsets2);
        OLTSize = *h_nextSize*SUBDIV*SUBDIV*SUBDIV*SUBDIV*2;
        //printf("OLTSize = %lu    --> %f GiBytes\n", OLTSize, 1.0*OLTSize*sizeof(int)/(1024*1024*1024.0));
        (cudaMalloc((void **)d_offsets2,  sizeof(int) * OLTSize));
        (cudaMemset(d_nextSize, 0, sizeof(int)));
        (cudaMemset(d_nbf, 0, sizeof(int)));
        d = d / SUBDIV;
        cucheck(cudaDeviceSynchronize());

        g = dim3(*h_nextSize, SUBDIV, SUBDIV);
         //printf("Running kernel with b(%i,%i) and g(%i, %i, %i) and d=%i\n", b.x, b.y, g.x, g.y, g.z, d);
        #ifdef DEBUG
            printf("[level %2i].....", i); fflush(stdout);
        #endif
        ASKNEW<<<g, b>>>(d_nextSize, d_nbf, *d_offsets1,  *d_offsets2, dwells, h, w, cmin, cmax, d,
                      i, SUBDIV, MAX_DWELL, MIN_SIZE, MAX_DEPTH, SUBDIV_ELEMS,
                      SUBDIV_ELEMS2, SUBDIV_ELEMSP, SUBDIV_ELEMSX, OLTSize);
        cucheck(cudaDeviceSynchronize());
        cudaMemcpy(h_nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost);
        #ifdef DEBUG
            dim3 gold = g;
        #endif
        if (*h_nextSize < g.x*g.y*g.z){
            g = dim3((g.x*g.y*g.z)-*h_nextSize, (d + b.x - 1)/b.x, (d + b.y - 1)/b.y);
            if (i+1 < MAX_DEPTH && d/SUBDIV > MIN_SIZE){
                doT<<<g, b>>>(dwells, w, d,  *d_offsets2, OLTSize);
            } else {
                doBruteForce<<<g, b>>>(dwells, w, h, cmin, cmax, d, MAX_DWELL,  *d_offsets2, OLTSize);
            }
        }
        #ifdef DEBUG
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop); // that's our time!
            //printf("[level %2i] h_nextSize  %i \n", i-1, *h_nextSize); 
            printf("done %f secs --> P=%f  r = %i x %i (grid %8i x %i x %i = %8i --> %i subdivided)\n", time/1000.0f, *h_nextSize/(float)(gold.x*SUBDIV*SUBDIV), d, d, gold.x, SUBDIV, SUBDIV, gold.x*SUBDIV*SUBDIV, *h_nextSize);
            cudaEventRecord(start, 0);
        #endif
    }
    cucheck(cudaFree(d_nbf))
}
