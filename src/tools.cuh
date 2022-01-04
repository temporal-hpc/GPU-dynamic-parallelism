#pragma once

using namespace std;

void check_args_info(int argc){
    if (argc != 15) {
        cout << "\nExecute as ./bin/gpuDP <dev> <Approach> <W> <H> <rmin> <rmax> <cmin> <cmax> <CA_MAXDWELL> <g> <r> <B> <MAX_DEPTH> <filename>" << endl;
        cout << "        dev:   GPU ID (0,1,2...)" << endl;
        cout << "   Approach:" << endl;
        cout << "               0 - Ex: Exhaustive (classic one-pass approach)" << endl;
        cout << "               1 - DP-SBR: Dynamic Parallelism Single-Block per Region (SBR)" << endl;
        cout << "               2 - DP-MBR: Dynamic Parallelism Multiple-Blocks per Region (MBR)" << endl;
        cout << "               3 - ASK-SBR: Adaptive Serial Kernels Single-Block per Region (SBR)" << endl;
        cout << "               4 - ASK-MBR: Adaptive Serial Kernels Multiple-Block per Region (MBR)" << endl;
        cout << "   -----------------------------------------------------------------------------------------------" << endl;
        cout << "   Parameters                  Example         Info" << endl;
        cout << "   -----------------------------------------------------------------------------------------------" << endl;
        cout << "   W ------------------------- 1024            Width in pixels." << endl;
        cout << "   H ------------------------- 1024            Height in pixels." << endl;
        cout << "   rmin rmax ----------------- [-1.5, 0.5]     Real part range." << endl;
        cout << "   cmin cmax ----------------- [-1.0, 1.0]     Complex part range." << endl;
        cout << "   CA_MAXDWELL --------------- 512             Maximum numbers of iterarions per element." << endl;
        cout << "   g ------------------------- 32              Initial numbers of regions (powers of 2)." << endl;
        cout << "   r ------------------------- 4               Subdivision scheme (powers of 2)." << endl;
        cout << "   B ------------------------- 32              Region Size for stopping subdivision (powers of 2)." << endl;
        cout << "   MAX_DEPTH ----------------- 5               Maximum recursion depth." << endl;
        cout << "   filename  ----------------- none            Chosen filename, 'none' to skip file output." << endl;
        exit(-1);
    }
}
