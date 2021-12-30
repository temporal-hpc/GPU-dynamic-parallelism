#pragma once

#include "complex.cuh"
#include "macros.cuh"
#include "mandelbrotHelper.cuh"

__global__ void kernel_ASK_SBR(unsigned int *d_ns, int *d_offs1, int *d_offs2, int *dwells,
                    int w, int h, complex cmin, complex cmax, int d, int depth,
                    unsigned int SUBDIV, unsigned int MAX_DWELL,
                    unsigned int MIN_SIZE, unsigned int MAX_DEPTH) {
    // check whether all boundary pixels have the same dwell
    //unsigned int use =
    //    blockIdx.x * SUBDIV_ELEMS2 + (blockIdx.z * SUBDIV + blockIdx.y) * 2;
    unsigned int use = blockIdx.x * 2;
    const unsigned int x0 = d_offs1[use] + blockIdx.y*d;
    const unsigned int y0 = d_offs1[use + 1]+ blockIdx.z*d;

    __shared__ unsigned int off_index;

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
        // RELLENAR (T)
        //printf("x0 = %i, y0 = %i\n", x0, y0);
        unsigned int x = threadIdx.x;
        unsigned int y = threadIdx.y;
        for (unsigned int ry = y; ry < d; ry += blockDim.y) {
            for (unsigned int rx = x; rx < d; rx += blockDim.x) {
                // CRISTOBAL TODO (comentar el if ya que se pregunta en los for)
                if (rx < d && ry < d) {
                    unsigned int rxx = rx + x0, ryy = ry + y0;
                    dwells[ryy * (size_t)w + rxx] = comm_dwell;
                }
            }
        }
    } else if (depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
        // SUBDIVIDIR
        //printf("asd\n");

        if (tid == 0) {
            off_index = atomicAdd(d_ns, 1);
            d_offs2[(off_index * 2)] = (x0);
            d_offs2[(off_index * 2) + 1] = (y0);
            //printf("offindex = %i, index = %i, blockx=%i, blocky=%i, blockz=%i\n", off_index, (off_index * 2), (blockIdx.x), (blockIdx.y), blockIdx.z);
            //printf("level %i --- x0 = %i, y0 = %i\n", depth, x0, y0);
        }
        /*__syncthreads();
        if (tid < 2) {
            d_offs2[(off_index * 2) + tid] = (x0)*((tid + 1) & 1) + (y0)*(tid & 1);
        }*/
    } else {
        // FUERZA BRUTA
        // return;
        //printf("x0 = %i, y0 = %i\n", x0, y0);

        unsigned int x = threadIdx.x;
        unsigned int y = threadIdx.y;
        for (unsigned int ry = y; ry < d; ry += blockDim.y) {
            for (unsigned int rx = x; rx < d; rx += blockDim.x) {
                // CRISTOBAL TODO (comentar el if ya que se pregunta en los for)
                if (rx < d && ry < d) {
                    unsigned int rxx = rx + x0, ryy = ry + y0;
                    dwells[ryy * (size_t)w + rxx] = pixel_dwell(w, h, cmin, cmax, rxx, ryy, MAX_DWELL);
                }
            }
        }
    }
}

void ASK_SBR(int *dwell, unsigned int *h_nextSize,
                           unsigned int *d_nextSize, int *d_offsets1,
                           int *d_offsets2, int w, int h, complex cmin, complex cmax,
                           int d, int depth, unsigned int INIT_SUBDIV,
                           unsigned int SUBDIV, unsigned int MAX_DWELL,
                           unsigned int MIN_SIZE, unsigned int MAX_DEPTH) {

    dim3 b(BSX, BSY, 1), g(1, INIT_SUBDIV, INIT_SUBDIV);

    #ifdef DEBUG
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        printf("\n[level %2i]...... d = %i ..... ", 1, d); fflush(stdout); 
    #endif
    kernel_ASK_SBR<<<g, b>>>(d_nextSize, d_offsets1, d_offsets2, dwell, h, w, cmin, cmax, d, depth, SUBDIV, MAX_DWELL, MIN_SIZE, MAX_DEPTH);
    cucheck(cudaDeviceSynchronize());
    for(int i = depth + 1; i < MAX_DEPTH && d / SUBDIV > MIN_SIZE; i++) {
        cudaMemcpy(h_nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost);

        #ifdef DEBUG
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop); // that's our time!
            printf("done %f secs --> P=%f (grid %8i x %i x %i = %8i --> %i subdivided)\n", time/1000.0f, *h_nextSize/(float)(g.x*g.y*g.z), g.x, g.y, g.z, g.x*g.y*g.z, *h_nextSize);
        #endif

        std::swap(d_offsets1, d_offsets2);

        cudaFree(d_offsets2);
        size_t OLTSize;
        
        OLTSize = *h_nextSize * (size_t)SUBDIV*SUBDIV*2;
        g = dim3(*h_nextSize, SUBDIV, SUBDIV);
       	d = d / SUBDIV;
        
        cucheck(cudaMalloc((void **)&d_offsets2, OLTSize*sizeof(int)));
		cudaDeviceSynchronize();
        cucheck(cudaMemset(d_nextSize, 0, sizeof(int)));

        #ifdef DEBUG
        	printf("OLTSize = %lu    --> %f GiBytes\n", OLTSize, 1.0*OLTSize*sizeof(int)/(1024*1024*1024.0));
            printf("[level %2i]...... d = %i, MIN_SIZE = %i ---- ", i, d, MIN_SIZE); fflush(stdout);
            cudaEventRecord(start, 0);
        #endif
        kernel_ASK_SBR<<<g, b>>>(d_nextSize, d_offsets1, d_offsets2, dwell, h, w, cmin, cmax, d, i, SUBDIV, MAX_DWELL, MIN_SIZE, MAX_DEPTH);
        cucheck(cudaDeviceSynchronize());
        //getchar();

    }
    cucheck(cudaFree(d_offsets1));
    cucheck(cudaFree(d_offsets2));
    #ifdef DEBUG
        cucheck(cudaMemcpy(h_nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop); // that's our time!
        printf("done %f secs --> P=%f (grid %8i x %i x %i = %8i --> %i subdivided)\n", time/1000.0f, *h_nextSize/(float)(g.x*g.y*g.z), g.x, g.y, g.z, g.x*g.y*g.z, *h_nextSize);
    #endif
}

