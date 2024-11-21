# GPU Dynamic Parallelism
<img src="https://github.com/temporal-hpc/GPU-dynamic-parallelism/raw/master/images/example.png" width=45% height=45%> <img src="https://github.com/temporal-hpc/GPU-dynamic-parallelism/raw/master/images/example-gridlines.png" width=45% height=45%>

This program serves as an experimental tool for comparing the efficiency of the GPU under heterogeneous parallel workloads. It uses the 
Mandelbrot Set as case study, offering three GPU-based approaches:
- Exhaustive approach:      The classic flat-kernel parallel pass
- CUDA Dynamic Parallelism: Nvidia's recursive kernel approach
- Adaptive Serial Kernels:  An alternative iterative-kernel approach (two variants proposed) 

## Compile
```
➜  GPU-dynamic-parallelism git:(master) make
nvcc -O3 -arch=sm_75 -rdc=true -lcudadevrt -Xcompiler -fopenmp -lpng src/main.cu -o bin/mandelbrot -Dno -Dno -DBSY=32 -DBSX=32 -DREPEATS=1
➜  GPU-dynamic-parallelism git:(master) 
```
See the `Makefile` for additional options such as printing progress info (`VERBOSE`) or exporting the subdivision grid image (`GRIDLINES`).

## Run
```
➜  GPU-dynamic-parallelism git:(master) ✗ bin/gpuDP

Execute as ./bin/gpuDP <Approach> <W> <H> <rmin> <rmax> <cmin> <cmax> <CA_MAXDWELL> <B> <g> <r> <MAX_DEPTH> <filename>
   Approach:
               0 - Exhaustive (classic one-pass approach)
               1 - Dynamic Parallelism (from Nvidia)
               2 - Adaptive Serial Kernels (one thread-block per region)
               3 - Adaptive Serial Kernels (multiple thread-blocks per region)
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
➜  GPU-dynamic-parallelism git:(master) ✗ ./bin/gpuDP 2 $((2**10)) $((2**10)) -1.5 0.5 -1.0 1.0 512    1 2 2  1000 example
Grid 1024 x 1024 --> 0.00 GiB
[level  1] 0.000190 secs -->  P_{ 1} = 1.000000   (grid        1 x 2 x 2 =        4 --> 4 subdivided)
[level  2] 0.000200 secs -->  P_{ 2} = 0.875000   (grid        4 x 2 x 2 =       16 --> 14 subdivided)
[level  3] 0.000186 secs -->  P_{ 3} = 0.892857   (grid       14 x 2 x 2 =       56 --> 50 subdivided)
[level  4] 0.000360 secs -->  P_{ 4} = 0.850000   (grid       50 x 2 x 2 =      200 --> 170 subdivided)
[level  5] 0.000837 secs -->  P_{ 5} = 0.795588   (grid      170 x 2 x 2 =      680 --> 541 subdivided)
[level  6] 0.001945 secs -->  P_{ 6} = 0.776802   (grid      541 x 2 x 2 =     2164 --> 1681 subdivided)
[level  7] 0.004726 secs -->  P_{ 7} = 0.724569   (grid     1681 x 2 x 2 =     6724 --> 4872 subdivided)
[level  8] 0.013105 secs -->  P_{ 8} = 0.704023   (grid     4872 x 2 x 2 =    19488 --> 13720 subdivided)
<level  9> 0.036247 secs -->  P_{ 9} = 0.000000   (grid    13720 x 2 x 2 =    54880 --> 0 subdivided)
2, 32, 32, 1024, 1024, 512, 1000, 2, 1, 0.057932
```

# Citation 

Felipe A. Quezada, Cristóbal A. Navarro, Miguel Romero, Cristhian Aguilera,
Modeling GPU Dynamic Parallelism for self similar density workloads,
Future Generation Computer Systems,
Volume 145,
2023,
Pages 239-253,
ISSN 0167-739X,
https://doi.org/10.1016/j.future.2023.03.046.
(https://www.sciencedirect.com/science/article/pii/S0167739X23001292)
Abstract: Dynamic Parallelism (DP) is a GPU programming abstraction that can make parallel computation more efficient for problems that exhibit heterogeneous workloads. With DP, GPU threads can launch kernels with more threads, recursively, producing a subdivision effect where resources are focused on the regions that exhibit more parallel work. Doing an optimal subdivision process is not trivial, as the combination of different parameters play a relevant role in the final performance of DP. Also, the current programming abstraction of DP relies on kernel recursion, which has performance overhead. This work presents a new subdivision cost model for problems that exhibit self similar density (SSD) workloads, useful for finding efficient subdivision schemes. Also, a new subdivision implementation free of recursion overhead is presented, named Adaptive Serial Kernels (ASK). Using the Mandelbrot set as a case study, the cost model shows that optimal performance is achieved when using {g∼32,r∼2,B∼32} for the initial subdivision, recurrent subdivision and stopping size, respectively. Experimental results agree with the theoretical parameters, confirming the usability of the cost model. In terms of performance, the ASK approach runs up to ∼60% faster than DP in the Mandelbrot set, and up to 12× faster than a basic exhaustive implementation, whereas DP is up to 7.5× faster. In terms of energy efficiency, ASK is up to ∼2× and ∼20× more energy efficient than DP and the exhaustive approach, respectively. These results put the subdivision cost model and the ASK approach as useful tools for analyzing the potential improvement of subdivision based approaches and for developing more efficient GPU-based libraries or fine-tune specific codes in research teams.
Keywords: GPU; Dynamic Parallelism; Subdivision; Heterogeneous workload; Kernel recursion overhead; Self similar density

