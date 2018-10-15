// Compile with: nvcc -o out -Wno-deprecated-gpu-targets main.cu -std=c++1
#include "common.h"
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <math.h>
#include <string>

using namespace std;

__global__ void cudaWithBlocksAndThreads(int *MatA, int *MatB, int *MatC, const int nx, const int ny)
{
    // Codigo de clase
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int i, j;

    if (ix < nx)
    {
        for (i = 0; i < nx; i++)
        {
            for (j = 0; j < ny; j++)
            {
                MatC[ix * nx + i] += (MatA[ix * nx + j] * MatB[j * nx + i]);
            }
        }

    }

    return;
}

void initialData(int *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = i;
    }
}

int main(int argc, char const *argv[])
{
    printf("%s Starting...\n", argv[0]);

    // Set up data size of matrix
    /* int size = 0; */
    /* int algorithm = 0; */
    /* if(argc < 2) */
    /*     size = 1000; */
    /* else if (argc == 2) */
    /* { */
    /*     size = stoi(argv[1]); */
    /* } */
    /* else if (argc == 3) */
    /* { */
    /*     size = stoi(argv[1]); */
    /*     algorithm = stoi(argv[2]); */
    /* } */
    int size = 2000;

    int nx = size;
    int ny = size;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // Malloc host memory
    int *h_A, *h_B, *hostRefNoThreads, *hostRefThreads, *gpuRefThreads;
    h_A = (int *)malloc(nBytes);
    h_B = (int *)malloc(nBytes);
    hostRefNoThreads = (int *)malloc(nBytes);
    hostRefThreads = (int *)malloc(nBytes);
    gpuRefThreads = (int *)malloc(nBytes);

    // Initialize data at host side
    initialData(h_A, nxy);
    initialData(h_B, nxy);

    // Start Matrix Multiplication and timer
    // CPU No Threads
    if (algorithm == 0)
    {
        auto start =  chrono::high_resolution_clock::now();
        inCPUWithoutThreads(h_A, h_B, hostRefNoThreads, nx, ny);
        auto end =  chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> duration_ms = end - start;
        printf("inCPUWithoutThreads elapsed %f ms\n", duration_ms.count());
    }

    // CPU OMP Threads
    if (algorithm == 1)
    {
        auto start = chrono::high_resolution_clock::now();
        inCPUWithThreads(h_A, h_B, hostRefThreads, nx, ny);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> duration_ms = end - start;
        printf("inCPUWithThreads elapsed %f ms\n", duration_ms.count());
    }

    // Malloc device global memory
    int *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // invoke kernel at host side
    int dimx = 32;
    dim3 block(dimx, 1);
    dim3 grid((nx + block.x - 1) / block.x, 1);

    if (algorithm == 2)
    {
        auto start =  chrono::high_resolution_clock::now();
        cudaWithBlocksAndThreads<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
        SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
        auto end =  chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> duration_ms = end - start;
        printf("cudaWithBlocksAndThreads <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n",
               grid.x,
               grid.y,
               block.x, block.y, duration_ms.count());
    }

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    /* SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC"); */

    // check device results
    /* checkResult(hostRef, gpuRef, nxy); */

    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRefNoThreads);
    free(hostRefThreads);
    free(gpuRefThreads);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return 0;
}
