#pragma once

using namespace std;

void check_args_info(int argc){
    if (argc != 14) {
        cout << "Error. Execute as ./mandelbrot <Approach> <W> <H> <rmin> <rmax> <cmin> <cmax> <CA_MAXDWELL> <B> <G0> <R> <MAX_DEPTH> <filename>" << endl;
        cout << "Apporach:  0 - Brute Force" << endl;
        cout << "           1 - Dynamic Parallelism" << endl;
        cout << "           2 - Adaptive Serial Kernels" << endl;
        cout << "Default values:" << endl;
        cout << "   W ------------------------- Width in pixels." << endl;
        cout << "   H ------------------------- Height in pixels." << endl;
        cout << "   xlim -------- [-1.5, 0.5] - x range in the plane (real values)." << endl;
        cout << "   ylim -------- [-1.0, 1.0] - y range in the plane (imaginary values)." << endl;
        cout << "   CA_MAXDWELL --------------- 512 - Maximum numbers of iterarions per element." << endl;
        cout << "   B ------------------------- 32 - Size of the regions when exhaustive is triggered (powers of 2)." << endl;
        cout << "   G0 ------------------------ 32 - Initial numbers of regions (powers of 2)." << endl;
        cout << "   r ------------------------- 4 - Subdivision scheme (powers of 2)." << endl;
        cout << "   MAX_DEPTH ----------------- 5 - Maximum recursion depth." << endl;
        cout << "   filename  ----------------- 'none' - Use 'none' to skip file output." << endl;
        exit(-1);
    }
}
