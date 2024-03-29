#pragma once

#define VAL(str) #str
#define TOSTRING(str) VAL(str)

std::string approchToString(int approach) {
    switch (approach) {
    case 0:
        return "exhaustive";
        break;
    case 1:
        return "DP-SBR";
        break;
    case 2:
        return "DP-MBR";
        break;
    case 3:
        return "ASK-SBR";
        break;
    case 4:
        return "ASK-MBR";
        break;
    }
    return "";
}
float rea_Ex(int* d_dwells, unsigned int w, unsigned int h, complex bottomLeftCorner, complex upperRightCorner,
    unsigned int g0, unsigned int r,
    unsigned int CA_MAXDWELL, unsigned int B,
    unsigned int MAX_DEPTH);
float rea_DP_SBR(int* d_dwells, unsigned int w, unsigned int h, complex bottomLeftCorner, complex upperRightCorner,
    unsigned int g0, unsigned int r,
    unsigned int CA_MAXDWELL, unsigned int B,
    unsigned int MAX_DEPTH);
float rea_DP_MBR(int* d_dwells, unsigned int w, unsigned int h, complex bottomLeftCorner, complex upperRightCorner,
    unsigned int g0, unsigned int r,
    unsigned int CA_MAXDWELL, unsigned int B,
    unsigned int MAX_DEPTH);
float rea_ASK_SBR(int* d_dwells, unsigned int w, unsigned int h, complex bottomLeftCorner, complex upperRightCorner,
    unsigned int g0, unsigned int r,
    unsigned int CA_MAXDWELL, unsigned int B,
    unsigned int MAX_DEPTH);
float rea_ASK_MBR(int* d_dwells, unsigned int w, unsigned int h, complex bottomLeftCorner, complex upperRightCorner,
    unsigned int g0, unsigned int r,
    unsigned int CA_MAXDWELL, unsigned int B,
    unsigned int MAX_DEPTH);

statistics doBenchmark(int approach, int* d_dwells, unsigned int w, unsigned int h,
    complex bottomLeftCorner, complex upperRightCorner,
    unsigned int g0, unsigned int r,
    unsigned int CA_MAXDWELL, unsigned int B,
    unsigned int MAX_DEPTH) {

    typedef float (*f)(int*, unsigned int, unsigned int, complex, complex, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);
    f func[5] = { rea_Ex, rea_DP_SBR, rea_DP_MBR, rea_ASK_SBR, rea_ASK_MBR };
    float elapsedTime = 0.0f;
    statistics stats;
    RunningStat meas;
#ifdef MEASURE_POWER
    GPUPowerBegin(w * (size_t)h, 100, 0, approchToString(approach) + std::string("-") + TOSTRING(BSX) + std::string("x") + TOSTRING(BSY) + std::string("-") + TOSTRING(ARCH));
#endif
    for (int k = 0; k < REALIZATIONS; k++) {
        elapsedTime = func[approach](d_dwells, w, h, bottomLeftCorner, upperRightCorner, g0, r, CA_MAXDWELL, B, MAX_DEPTH);
        meas.Push(elapsedTime);
    }
#ifdef MEASURE_POWER
    GPUPowerEnd();
#endif
    stats.mean = meas.Mean();
    stats.variance = meas.Variance();
    stats.stdev = meas.StandardDeviation();
    stats.sterr = meas.StandardDeviation() / ((double)sqrt(meas.NumDataValues()));
    return stats;
}

float rea_Ex(int* d_dwells, unsigned int w, unsigned int h, complex bottomLeftCorner, complex upperRightCorner,
    unsigned int g0, unsigned int r,
    unsigned int CA_MAXDWELL, unsigned int B,
    unsigned int MAX_DEPTH) {

    float elapsedTime;
    dim3 blockSize(BSX, BSY), gridSize(divup(w, blockSize.x), divup(h, blockSize.y));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < REPEATS; i++) {
        mandelbrot_k<<<gridSize, blockSize>>>(d_dwells, w, h, bottomLeftCorner, upperRightCorner, CA_MAXDWELL);
        cucheck(cudaDeviceSynchronize());
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}

float rea_ASK_SBR(int* d_dwells, unsigned int w, unsigned int h, complex bottomLeftCorner, complex upperRightCorner,
    unsigned int g0, unsigned int r,
    unsigned int CA_MAXDWELL, unsigned int B,
    unsigned int MAX_DEPTH) {

    int *h_offsets, *d_offsets1, *d_offsets2; // OLT
    unsigned int *h_OLTSize, *d_OLTSize; // OLT SIZE

    float elapsedTime = 0;

    h_OLTSize = (unsigned int*)malloc(sizeof(int));
    *h_OLTSize = g0 * g0 * 2;

    cucheck(cudaMalloc(&d_OLTSize, sizeof(int)));

    size_t initialOLTSize = *h_OLTSize * sizeof(int);

    h_offsets = (int*)malloc(*h_OLTSize * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < REPEATS; i++) {
        h_offsets[0] = 0;
        h_offsets[1] = 0;
        *h_OLTSize = 1;
        // these two pointers get a cudaFree inside the function for which they are arguments
        cucheck(cudaMalloc((void**)&d_offsets1, initialOLTSize));
        cucheck(cudaMalloc((void**)&d_offsets2, initialOLTSize));

        cucheck(cudaMemcpy(d_offsets1, h_offsets, initialOLTSize, cudaMemcpyHostToDevice));
        cucheck(cudaMemset(d_OLTSize, 0, sizeof(int)));
        // printf("INITIAL=%i\n", initialOLTSize);
        float iterationTime = 0;
        cudaEventRecord(start, 0);

        ASK_SBR(d_dwells, h_OLTSize, d_OLTSize, d_offsets1, d_offsets2, w, h, bottomLeftCorner, upperRightCorner, w / g0, 1, g0, r, CA_MAXDWELL, B, MAX_DEPTH);

        cucheck(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&iterationTime, start, stop); // that's our time!
        elapsedTime += iterationTime;
    }

    // free memory for this realization
    cucheck(cudaFree(d_OLTSize));
    free(h_OLTSize);
    free(h_offsets);

    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}

float rea_ASK_MBR(int* d_dwells, unsigned int w, unsigned int h,
    complex bottomLeftCorner, complex upperRightCorner,
    unsigned int g0, unsigned int r,
    unsigned int CA_MAXDWELL, unsigned int B,
    unsigned int MAX_DEPTH) {

    int* h_offsets; // OLT
    int *d_offsets1, *d_offsets2; // OLT
    unsigned int *h_OLTSize, *d_OLTSize; // OLT SIZE

    float elapsedTime = 0;

    h_OLTSize = (unsigned int*)malloc(sizeof(int));
    *h_OLTSize = g0 * g0 * 2;

    cucheck(cudaMalloc(&d_OLTSize, sizeof(int)));

    size_t initialOLTSize = *h_OLTSize * sizeof(int);

    h_offsets = (int*)malloc(*h_OLTSize * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < REPEATS; i++) {
        h_offsets[0] = 0;
        h_offsets[1] = 0;

        *h_OLTSize = 1;
        cucheck(cudaMalloc((void**)&d_offsets1, initialOLTSize));
        cucheck(cudaMalloc((void**)&d_offsets2, initialOLTSize));

        cucheck(cudaMemcpy(d_offsets1, h_offsets, initialOLTSize, cudaMemcpyHostToDevice));
        cucheck(cudaMemset(d_OLTSize, 0, sizeof(int)));

        float iterationTime = 0;
        cudaEventRecord(start, 0);

        ASK_MBR(d_dwells, h_OLTSize, d_OLTSize, &d_offsets1, &d_offsets2, w, h, bottomLeftCorner, upperRightCorner, w / g0, 1, g0, r, CA_MAXDWELL, B, MAX_DEPTH);

        cucheck(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&iterationTime, start, stop); // that's our time!
        elapsedTime += iterationTime;
        cucheck(cudaFree(d_offsets1));
        cucheck(cudaFree(d_offsets2));
    }

    // free memory for this realization
    cucheck(cudaFree(d_OLTSize));
    free(h_OLTSize);
    free(h_offsets);
    cucheck(cudaDeviceSynchronize());

    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}

float rea_DP_SBR(int* d_dwells, unsigned int w, unsigned int h,
    complex bottomLeftCorner, complex upperRightCorner,
    unsigned int g0, unsigned int r,
    unsigned int CA_MAXDWELL, unsigned int B,
    unsigned int MAX_DEPTH) {
    float elapsedTime;
    dim3 blockSize(BSX, BSY), gridSize(g0, g0);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
#ifdef RDC_TRUE
    for (int i = 0; i < REPEATS; i++) {
        dp_sbr_mandelbrot_block_k<<<gridSize, blockSize>>>(d_dwells, w, h, bottomLeftCorner, upperRightCorner, 0, 0, w / g0, 1, r, CA_MAXDWELL, B, MAX_DEPTH);
        cucheck(cudaDeviceSynchronize());
    }
#endif
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}

float rea_DP_MBR(int* d_dwells, unsigned int w, unsigned int h,
    complex bottomLeftCorner, complex upperRightCorner,
    unsigned int g0, unsigned int r,
    unsigned int CA_MAXDWELL, unsigned int B,
    unsigned int MAX_DEPTH) {
    float elapsedTime;
    dim3 blockSize(BSX, BSY), gridSize(g0, g0);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
#ifdef DEBUG
    printf("[DEBUG] DP_MBR  g0=%i  r=%i  B=%i\n", g0, r, B);
#endif
#ifdef RDC_TRUE
    for (int i = 0; i < REPEATS; i++) {
        dp_mbr_mandelbrot_block_k<<<gridSize, blockSize>>>(d_dwells, w, h, bottomLeftCorner, upperRightCorner, 0, 0, w / g0, 1, r, CA_MAXDWELL, B, MAX_DEPTH);
        cucheck(cudaDeviceSynchronize());
    }
#endif
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}

float doGridLines(int* d_dwells, unsigned int w, unsigned int h,
    complex bottomLeftCorner, complex upperRightCorner,
    unsigned int g0, unsigned int r,
    unsigned int CA_MAXDWELL, unsigned int B,
    unsigned int MAX_DEPTH) {

    int *h_offsets, *d_offsets1, *d_offsets2; // OLT
    unsigned int *h_OLTSize, *d_OLTSize; // OLT SIZE

    float elapsedTime = 0;

    h_OLTSize = (unsigned int*)malloc(sizeof(int));
    *h_OLTSize = g0 * g0 * r * r * 2;

    cucheck(cudaMalloc(&d_OLTSize, sizeof(int)));

    size_t initialOLTSize = *h_OLTSize * sizeof(int);

    h_offsets = (int*)malloc(*h_OLTSize * sizeof(int));

    for (int i = 0; i < g0 * g0 * 2; i += 2) {
        h_offsets[i] = ((i / 2) % g0) * (w / g0);
        h_offsets[i + 1] = ((i / 2) / g0) * (w / g0);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int j = 0; j < g0 * g0 * 2; j += 2) {
        h_offsets[j] = ((j / 2) % g0) * (w / g0);
        h_offsets[j + 1] = ((j / 2) / g0) * (w / g0);
        // printf("Offsets Iniciales: (%i) - %i, %i\n", i/2, h_offsets[i],
        // h_offsets[i+1]);
    }
    *h_OLTSize = 1;
    cucheck(cudaMalloc((void**)&d_offsets1, initialOLTSize));
    cucheck(cudaMalloc((void**)&d_offsets2, initialOLTSize));

    cucheck(cudaMemcpy(d_offsets1, h_offsets, initialOLTSize,
        cudaMemcpyHostToDevice));
    cucheck(cudaMemset(d_OLTSize, 0, sizeof(int)));

    float iterationTime = 0;
    cudaEventRecord(start, 0);

    GridLines(d_dwells, h_OLTSize, d_OLTSize, d_offsets1, d_offsets2,
        w, h, bottomLeftCorner, upperRightCorner,
        w / g0, 1, g0, r, CA_MAXDWELL,
        B, MAX_DEPTH);
    cucheck(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&iterationTime, start, stop); // that's our time!
    elapsedTime += iterationTime;

    // free memory for this realization
    cucheck(cudaFree(d_OLTSize));
    free(h_OLTSize);
    free(h_offsets);
    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}
