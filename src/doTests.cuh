#pragma once

float doExhaustive(int *d_dwells, unsigned int w, unsigned int h, complex bottomLeftCorner, complex upperRightCorner,
                              unsigned int g0, unsigned int r,
                              unsigned int CA_MAXDWELL, unsigned int B,
                              unsigned int MAX_DEPTH);
float doDynamicParallelism(int *d_dwells, unsigned int w, unsigned int h, complex bottomLeftCorner, complex upperRightCorner,
                              unsigned int g0, unsigned int r,
                              unsigned int CA_MAXDWELL, unsigned int B,
                              unsigned int MAX_DEPTH);
float doAdaptiveSerialKernels(int *d_dwells, unsigned int w, unsigned int h, complex bottomLeftCorner, complex upperRightCorner,
                              unsigned int g0, unsigned int r,
                              unsigned int CA_MAXDWELL, unsigned int B,
                              unsigned int MAX_DEPTH);
float doAdaptiveSerialKernelsNEW(int *d_dwells, unsigned int w, unsigned int h, complex bottomLeftCorner, complex upperRightCorner,
                              unsigned int g0, unsigned int r,
                              unsigned int CA_MAXDWELL, unsigned int B,
                              unsigned int MAX_DEPTH);



statistics doTest(int approach, int *d_dwells, unsigned int w, unsigned int h,
                              complex bottomLeftCorner, complex upperRightCorner,
                              unsigned int g0, unsigned int r,
                              unsigned int CA_MAXDWELL, unsigned int B,
                              unsigned int MAX_DEPTH) {

    typedef float (*f)(int*, unsigned int, unsigned int, complex, complex, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);
    f func[4] = {doExhaustive, doDynamicParallelism, doAdaptiveSerialKernels, doAdaptiveSerialKernelsNEW};
    float elapsedTime = 0.0f;
    statistics stats;
    RunningStat meas;
    for (int k = 0; k < REALIZATIONS; k++) {
        elapsedTime = func[approach](d_dwells, w, h, bottomLeftCorner, upperRightCorner, g0, r, CA_MAXDWELL, B, MAX_DEPTH);
        meas.Push(elapsedTime);
    }
    stats.mean = meas.Mean();
    stats.variance = meas.Variance();
    stats.stdev = meas.StandardDeviation();
    stats.sterr = meas.StandardDeviation()/((double)sqrt(meas.NumDataValues()));
    return stats;
}


float doExhaustive(int *d_dwells, unsigned int w, unsigned int h, complex bottomLeftCorner, complex upperRightCorner,
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
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    elapsedTime /= (REPEATS * 1000.f);
    return elapsedTime;
}


float doAdaptiveSerialKernels(int *d_dwells, unsigned int w, unsigned int h, complex bottomLeftCorner, complex upperRightCorner,
                              unsigned int g0, unsigned int r,
                              unsigned int CA_MAXDWELL, unsigned int B,
                              unsigned int MAX_DEPTH) {

    int *h_offsets, *d_offsets1, *d_offsets2; // OLT
    unsigned int *h_OLTSize, *d_OLTSize;      // OLT SIZE

    float elapsedTime = 0;

    h_OLTSize = (unsigned int *)malloc(sizeof(int));
    *h_OLTSize = g0 * g0 * 2;

    cucheck(cudaMalloc(&d_OLTSize, sizeof(int)));

    size_t initialOLTSize = *h_OLTSize * sizeof(int);

    h_offsets = (int *)malloc(*h_OLTSize * sizeof(int));


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < REPEATS; i++) {
        h_offsets[0] = 0;
        h_offsets[1] = 0;
        *h_OLTSize = 1;
        // these two pointers get a cudaFree inside the function for which they are arguments
        cucheck(cudaMalloc((void **)&d_offsets1, initialOLTSize));
        cucheck(cudaMalloc((void **)&d_offsets2, initialOLTSize));

        cucheck(cudaMemcpy(d_offsets1, h_offsets, initialOLTSize, cudaMemcpyHostToDevice));
        cucheck(cudaMemset(d_OLTSize, 0, sizeof(int)));
        //printf("INITIAL=%i\n", initialOLTSize);
        float iterationTime = 0;
        cudaEventRecord(start, 0);

        AdaptiveSerialKernels(d_dwells, h_OLTSize, d_OLTSize, d_offsets1, d_offsets2, w, h, bottomLeftCorner, upperRightCorner, w / g0, 1, g0, r, CA_MAXDWELL, B, MAX_DEPTH);

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

float doAdaptiveSerialKernelsNEW(int *d_dwells, unsigned int w, unsigned int h,
                              complex bottomLeftCorner, complex upperRightCorner,
                              unsigned int g0, unsigned int r,
                              unsigned int CA_MAXDWELL, unsigned int B,
                              unsigned int MAX_DEPTH) {

    int *h_offsets; // OLT
    int *d_offsets1, *d_offsets2; // OLT
    unsigned int *h_OLTSize, *d_OLTSize;      // OLT SIZE

    float elapsedTime = 0;

    h_OLTSize = (unsigned int *)malloc(sizeof(int));
    *h_OLTSize = g0 * g0 * 2;

    cucheck(cudaMalloc(&d_OLTSize, sizeof(int)));

    size_t initialOLTSize = *h_OLTSize * sizeof(int);

    h_offsets = (int *)malloc(*h_OLTSize * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < REPEATS; i++) {
        h_offsets[0] = 0;
        h_offsets[1] = 0;

        *h_OLTSize = 1;
        cucheck(cudaMalloc((void **)&d_offsets1, initialOLTSize));
        cucheck(cudaMalloc((void **)&d_offsets2, initialOLTSize));

        cucheck(cudaMemcpy(d_offsets1, h_offsets, initialOLTSize, cudaMemcpyHostToDevice));
        cucheck(cudaMemset(d_OLTSize, 0, sizeof(int)));

        float iterationTime = 0;
        cudaEventRecord(start, 0);

        AdaptiveSerialKernelsNEW(d_dwells, h_OLTSize, d_OLTSize, &d_offsets1, &d_offsets2,
                              w, h, bottomLeftCorner, upperRightCorner,
                              w / g0, 1, g0, r, CA_MAXDWELL,
                              B, MAX_DEPTH);
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

float doDynamicParallelism(int *d_dwells, unsigned int w, unsigned int h,
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
    for (int i = 0; i < REPEATS; i++) {
        mandelbrot_block_k<<<gridSize, blockSize>>>( d_dwells, w, h, bottomLeftCorner, upperRightCorner, 0, 0, w / g0, 1, r, CA_MAXDWELL, B, MAX_DEPTH);
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
                              unsigned int g0, unsigned int r,
                              unsigned int CA_MAXDWELL, unsigned int B,
                              unsigned int MAX_DEPTH) {

    int *h_offsets, *d_offsets1, *d_offsets2; // OLT
    unsigned int *h_OLTSize, *d_OLTSize;      // OLT SIZE

    float elapsedTime = 0;

    h_OLTSize = (unsigned int *)malloc(sizeof(int));
    *h_OLTSize = g0 * g0 * r * r * 2;

    cucheck(cudaMalloc(&d_OLTSize, sizeof(int)));

    size_t initialOLTSize = *h_OLTSize * sizeof(int);

    h_offsets = (int *)malloc(*h_OLTSize * sizeof(int));

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
    cucheck(cudaMalloc((void **)&d_offsets1, initialOLTSize));
    cucheck(cudaMalloc((void **)&d_offsets2, initialOLTSize));


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

