#include "common.h"
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <stdlib.h>
#define TILESIZE 32

using namespace std;

int N = 2000;
int numCRows;
int numARows = N;
int numBRows = N;
int numCColumns;
int numAColumns = N;
int numBColumns = N;

__global__ void cudaWithBlocksAndThreads(float *MatA, float *MatB, float *MatC, const int nx, const int ny)
{
    //Codigo de clase
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int i, j;

    if (ix < nx) {
        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                MatC[ix * nx + i] += (MatA[ix * nx + j] * MatB[j * nx + i]);
            }
        }
    }
    return;
}

__global__ void matrixMultiplyTiled(float * A, float * B, float * C,
                                    int numARows, int numAColumns,
                                    int numBRows, int numBColumns,
                                    int numCRows, int numCColumns) {
    __shared__ float sA[TILESIZE][TILESIZE];
    __shared__ float sB[TILESIZE][TILESIZE];

    int Row = blockDim.y*blockIdx.y + threadIdx.y;
    int Col = blockDim.x*blockIdx.x + threadIdx.x;

    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int k = 0; k < (((numAColumns - 1)/ TILESIZE) + 1); k++) {
        // Copy Data to Tile from Matrix
        if ( (Row < numARows) && (threadIdx.x + (k*TILESIZE)) < numAColumns) {
            sA[threadIdx.y][threadIdx.x] = A[(Row*numAColumns) + threadIdx.x + (k*TILESIZE)];
        }
        else {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        // Copy Data to Tile from Matrix (Global Memory to Shared Memory)
        if ( Col < numBColumns && (threadIdx.y + k*TILESIZE) < numBRows) {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k*TILESIZE)*numBColumns + Col];
        }
        else {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        // Multiplying Elements in tile
        for (int j = 0; j < TILESIZE; ++j) {
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    // Saving result to C
    if (Row < numCRows && Col < numCColumns) {
        C[Row*numCColumns + Col] = Cvalue;
    }
}

void matMultiplyOnCPU(float * A, float * B, float * C, int numARows,
                        int numAColumns, int numBRows, int numBColumns,
                        int numCRows, int numCColumns)
{
    for (int i=0; i < numARows; i ++) {
        for (int j = 0; j < numAColumns; j++) {
            C[i*numCColumns + j ] = 0.0;
            for (int k = 0; k < numCColumns; k++) {
                C[i*numCColumns + j ] += A[i*numAColumns + k] * B [k*numBColumns + j];
            }
        }
    }
    return;
}

int main(int argc, char ** argv) {
    float * hostA;
    float * hostB;
    float * hostC;
    float * hostC2;
    float * hostComputedC;
    float * deviceA;
    float * deviceB;
    float * deviceC;
    float * deviceA2;
    float * deviceB2;
    float * deviceC2;

    // Alloc CPU Memory
    hostA = (float *) malloc(sizeof(float)*numARows*numAColumns);
    hostB = (float *) malloc(sizeof(float)*numBRows*numBColumns);

    // Initialize matrix A and B
    for (int i = 0; i < numARows*numAColumns; i++) {
        hostA[i] = 1.0;
        hostB[i] = 1.0;
    }

    // Setting matrix C
    numCRows = numARows;
    numCColumns = numBColumns;

    hostC = (float *) malloc(sizeof(float)*numCRows*numCColumns);
    hostC2 = (float *) malloc(sizeof(float)*numCRows*numCColumns);
    hostComputedC = (float *) malloc(sizeof(float)*numCRows*numCColumns);

    // Allocating GPU memory TILED
    SAFE_CALL(cudaMalloc((void **)&deviceA, sizeof(float)*numARows*numAColumns), "Error Allocation GPU memory A");
    SAFE_CALL(cudaMalloc((void **)&deviceB, sizeof(float)*numBRows*numBColumns), "Error Allocation GPU memory B");
    SAFE_CALL(cudaMalloc((void **)&deviceC, sizeof(float)*numCRows*numCColumns), "Error Allocation GPU memory C");

    // Allocating GPU memory NOT TILED
    SAFE_CALL(cudaMalloc((void **)&deviceA2, sizeof(float)*numARows*numAColumns), "Error Allocation GPU memory A2");
    SAFE_CALL(cudaMalloc((void **)&deviceB2, sizeof(float)*numBRows*numBColumns), "Error Allocation GPU memory B2");
    SAFE_CALL(cudaMalloc((void **)&deviceC2, sizeof(float)*numCRows*numCColumns), "Error Allocation GPU memory C2");

    // Copy memory to the GPU TILED
    SAFE_CALL(cudaMemcpy(deviceA, hostA, sizeof(float)*numARows*numAColumns, cudaMemcpyHostToDevice), "Error Copying memory to GPU A");
    SAFE_CALL(cudaMemcpy(deviceB, hostB, sizeof(float)*numBRows*numBColumns, cudaMemcpyHostToDevice), "Error Copying memory to GPU B");

    // Copy memory to the GPU NOT TILED
    SAFE_CALL(cudaMemcpy(deviceA2, hostA, sizeof(float)*numARows*numAColumns, cudaMemcpyHostToDevice), "Error Copying memory to GPU A2");
    SAFE_CALL(cudaMemcpy(deviceB2, hostB, sizeof(float)*numBRows*numBColumns, cudaMemcpyHostToDevice), "Error Copying memory to GPU B2");

    // Initialize the grid and block dimensions
    // Blocks required
    dim3 dimGrid((numCColumns / TILESIZE) + 1, (numCRows / TILESIZE) + 1, 1);
    // Threads in each block
    dim3 dimBlock(TILESIZE, TILESIZE, 1);

    // Call TILED kernel
    auto start = chrono::high_resolution_clock::now();
    matrixMultiplyTiled<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    auto end =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end - start;
    printf("TILED GPU %f ms\n", duration_ms.count());
    cudaError_t err1 = cudaPeekAtLastError();

    // Sync device
    cudaDeviceSynchronize();

    // Copy the results in GPU memory back to the CPU
    SAFE_CALL(cudaMemcpy(hostC, deviceC, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost), "Error copying results from GPU to CPU");

    // Call CPU MatMult
    auto start2 =  chrono::high_resolution_clock::now();
    matMultiplyOnCPU(hostA, hostB, hostComputedC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    auto end2 =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms2 = end2 - start2;
    printf("CPU %f ms\n", duration_ms2.count());

    // Call NOT TILED Kernel
    auto start3 =  chrono::high_resolution_clock::now();
    cudaWithBlocksAndThreads<<<dimGrid, dimBlock>>>(deviceA2, deviceB2, deviceC2, numARows, numARows);
    auto end3 =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms3 = end3 - start3;
    printf("NOT TILED GPU %f ms\n", duration_ms3.count());

    // Sync device
    cudaDeviceSynchronize();

    // Copy the results in GPU memory back to the CPU
    SAFE_CALL(cudaMemcpy(hostC2, deviceC2, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost), "Error copying results from GPU to CPU");

    // Compare reults from CPU and GPU
    for (int i=0; i < numCColumns*numCRows; i++) {
        if (hostComputedC[i] != hostC[i]) {
            printf("Diferentes valores en Row = %d Col = %d CPU[] = %f --GPU[] %f\n", i / numCColumns, i % numCColumns, hostComputedC[i], hostC[i]);
            break;
        }
    }

    // Free the GPU memory
    SAFE_CALL(cudaFree(deviceA), "Error Freeing GPU Memory A");
    SAFE_CALL(cudaFree(deviceB), "Error Freeing GPU Memory B");
    SAFE_CALL(cudaFree(deviceC), "Error Freeing GPU Memory C");
    SAFE_CALL(cudaFree(deviceA2), "Error Freeing GPU Memory A2");
    SAFE_CALL(cudaFree(deviceB2), "Error Freeing GPU Memory B2");
    SAFE_CALL(cudaFree(deviceC2), "Error Freeing GPU Memory C2");

    //Free the Pointer Memory
    free(hostA);
    free(hostB);
    free(hostC);
    free(hostC2);
    free(hostComputedC);

    return 0;
}
