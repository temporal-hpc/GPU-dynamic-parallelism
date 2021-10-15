# GPU Dynamic Parallelism
This program serves as an experimental tool for comparing the efficiency of the GPU under heterogeneous parallel workloads. It uses the 
Mandelbrot Set as case study, offering three GPU-based approaches:
- Exhaustive approach:      The classic flat-kernel parallel 
- CUDA Dynamic Parallelism: Nvidia's recursive kernel approach
- Adaptive Serial Kernels:  An alternative iterative-kernel approach (proposed) 

## Compile
```
➜  GPU-dynamic-parallelism git:(master) make
nvcc -O3 -arch=sm_75 -rdc=true -lcudadevrt -Xcompiler -fopenmp -lpng src/main.cu -o bin/mandelbrot -Dno -Dno -DBSY=32 -DBSX=32 -DREPEATS=1
➜  GPU-dynamic-parallelism git:(master) 
```
See the `Makefile` for additional options such as printing progress info (`VERBOSE`) or exporting the subdivision grid image (`GRIDLINES`).

## run
```
➜  GPU-dynamic-parallelism git:(master) ✗ bin/mandelbrot

Execute as ./bin/gpuDP <Approach> <W> <H> <rmin> <rmax> <cmin> <cmax> <CA_MAXDWELL> <B> <g> <r> <MAX_DEPTH> <filename>
   Approach:
               0 - Exhaustive (classic one-pass approach)
               1 - Dynamic Parallelism (from Nvidia)
               2 - Adaptive Serial Kernels (alternative approach)
   -----------------------------------------------------------------------------------------------
   Parameters                  Example         Info
   -----------------------------------------------------------------------------------------------
   W ------------------------- 1024            Width in pixels.
   H ------------------------- 1024            Height in pixels.
   rmin rmax ----------------- [-1.5, 0.5]     Real part range.
   cmin cmax ----------------- [-1.0, 1.0]     Complex part range.
   CA_MAXDWELL --------------- 512             Maximum numbers of iterarions per element.
   B ------------------------- 32              Region Size for stopping subdivision (powers of 2).
   g ------------------------- 32              Initial numbers of regions (powers of 2).
   r ------------------------- 4               Subdivision scheme (powers of 2).
   MAX_DEPTH ----------------- 5               Maximum recursion depth.
   filename  ----------------- none            Chosen filename, 'none' to skip file output.
```
Example:
```
➜  GPU-dynamic-parallelism git:(master) ✗ ./bin/gpuDP 2 $((2**13)) $((2**13)) -1.5 0.5 -1.0 1.0 512    1 2 2   1000 none
Grid 8192 x 8192 --> 0.25 GiB
[level  1] 0.000670 secs -->  P_{ 1} = 1.000000   (grid        1 x 2 x 2 =        4 --> 4 subdivided)
[level  2] 0.001377 secs -->  P_{ 2} = 0.875000   (grid        4 x 2 x 2 =       16 --> 14 subdivided)
[level  3] 0.000541 secs -->  P_{ 3} = 0.892857   (grid       14 x 2 x 2 =       56 --> 50 subdivided)
[level  4] 0.000522 secs -->  P_{ 4} = 0.850000   (grid       50 x 2 x 2 =      200 --> 170 subdivided)
[level  5] 0.000844 secs -->  P_{ 5} = 0.814706   (grid      170 x 2 x 2 =      680 --> 554 subdivided)
[level  6] 0.001960 secs -->  P_{ 6} = 0.783394   (grid      554 x 2 x 2 =     2216 --> 1736 subdivided)
[level  7] 0.005089 secs -->  P_{ 7} = 0.747264   (grid     1736 x 2 x 2 =     6944 --> 5189 subdivided)
[level  8] 0.015636 secs -->  P_{ 8} = 0.742243   (grid     5189 x 2 x 2 =    20756 --> 15406 subdivided)
[level  9] 0.040155 secs -->  P_{ 9} = 0.741497   (grid    15406 x 2 x 2 =    61624 --> 45694 subdivided)
[level 10] 0.081200 secs -->  P_{10} = 0.739359   (grid    45694 x 2 x 2 =   182776 --> 135137 subdivided)
[level 11] 0.167651 secs -->  P_{11} = 0.728631   (grid   135137 x 2 x 2 =   540548 --> 393860 subdivided)
<level 12> 0.403871 secs -->  P_{12} = 0.000000   (grid   393860 x 2 x 2 =  1575440 --> 0 subdivided)
2, 32, 32, 8192, 8192, 512, 1000, 2, 1, 0.719721
```
