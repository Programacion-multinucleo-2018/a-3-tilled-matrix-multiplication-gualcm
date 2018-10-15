#include "common.h"
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <stdlib.h>
#define TILESIZE 2

int numARows;   // number of rows in the matrix A
int numAColumns;  // number of columns in the matrix A
int numBRows;   // number of rows in the matrix B
int numBColumns;  // number of columns in the matrix B
int numCRows;  // number of rows in the matrix C (you have to set this)
int numCColumns; // number of columns in the matrix C (you have to set this)


// Compute C = A * B
//*************************************************************
//Kernel for shared memory/ Tiled execution
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
                                    int numARows, int numAColumns,
                                    int numBRows, int numBColumns,
                                    int numCRows, int numCColumns)
{
    __shared__ float sA[TILESIZE][TILESIZE];   // Tile size to store elements in shared memory
    __shared__ float sB[TILESIZE][TILESIZE];

    int Row = blockDim.y*blockIdx.y + threadIdx.y; //To generate ids of threads.
    int Col = blockDim.x*blockIdx.x + threadIdx.x;
    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int k = 0; k < (((numAColumns - 1)/ TILESIZE) + 1); k++)
    {
        if ( (Row < numARows) && (threadIdx.x + (k*TILESIZE)) < numAColumns)//Copy Data to Tile from Matrix (Global Memory to Shared Memory)
        {
            sA[threadIdx.y][threadIdx.x] = A[(Row*numAColumns) + threadIdx.x + (k*TILESIZE)];
        }
        else
        {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        if ( Col < numBColumns && (threadIdx.y + k*TILESIZE) < numBRows)//Copy Data to Tile from Matrix (Global Memory to Shared Memory)
        {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k*TILESIZE)*numBColumns + Col];
        }
        else
        {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < TILESIZE; ++j)//Multiplying Elements present in tile
        {
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (Row < numCRows && Col < numCColumns)//Saving Final result into Matrix C
    {
        C[Row*numCColumns + Col] = Cvalue;
    }
}
//*************************************************************
void Print_Mat(int Row,int Col,float * Mat)//Function To print the Matrix
{
 for(int i=0;i<Row*Col;i++)
   {
   printf("%f  ",*(Mat+i));

   if((i%Col)==0 )
    {
     printf("\n");
    }
   }
}//Function close
//*************************************************************
//Normal CPU Matrix Multiplication
void matMultiplyOnHost(float * A, float * B, float * C, int numARows,
                        int numAColumns, int numBRows, int numBColumns,
                        int numCRows, int numCColumns)
{
    for (int i=0; i < numARows; i ++)
    {
        for (int j = 0; j < numAColumns; j++)
        {
            C[i*numCColumns + j ] = 0.0;
            for (int k = 0; k < numCColumns; k++)
            {
                C[i*numCColumns + j ] += A[i*numAColumns + k] * B [k*numBColumns + j];
            }
        }
    }
    return;
}
//*************************************************************
int main(int argc, char ** argv) {
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * hostComputedC;
    float * deviceA;
    float * deviceB;
    float * deviceC;

    // Please adjust rows and columns according to you need.

    printf("\nPlease Enter Rows and Columns of A:");
    scanf("%d %d",&numARows,&numAColumns);

    printf("\nPlease Enter Rows and Columns of B:");
    scanf("%d %d",&numBRows,&numBColumns);

    hostA = (float *) malloc(sizeof(float)*numARows*numAColumns);
    hostB = (float *) malloc(sizeof(float)*numBRows*numBColumns);

    for (int i = 0; i < numARows*numAColumns; i++)//Matrix Initialization
    {
        hostA[i]=1.0;
    }
    for (int i = 0; i < numBRows*numBColumns; i++)
    {
        hostB[i]=1.0;
    }

    printf("\nMatrix A Values:\n");
    Print_Mat(numARows,numAColumns,hostA);//Function Call

    printf("\n\nMatrix B Values:\n");
    Print_Mat(numBRows,numBColumns,hostB);//Function Call



    // Setting numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;

    hostC = (float *) malloc(sizeof(float)*numCRows*numCColumns);
    hostComputedC = (float *) malloc(sizeof(float)*numCRows*numCColumns);

    // Allocating GPU memory
    SAFE_CALL(cudaMalloc((void **)&deviceA, sizeof(float)*numARows*numAColumns));
    SAFE_CALL(cudaMalloc((void **)&deviceB, sizeof(float)*numBRows*numBColumns));
    SAFE_CALL(cudaMalloc((void **)&deviceC, sizeof(float)*numCRows*numCColumns));

    // Copy memory to the GPU
    SAFE_CALL(cudaMemcpy(deviceA, hostA, sizeof(float)*numARows*numAColumns, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(deviceB, hostB, sizeof(float)*numBRows*numBColumns, cudaMemcpyHostToDevice));

    // Initialize the grid and block dimensions

    dim3 dimGrid((numCColumns/TILESIZE) + 1, (numCRows/TILESIZE) + 1, 1);//Number of Blocks required
    dim3 dimBlock(TILESIZE, TILESIZE, 1);//Number of threads in each block

    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

    cudaDeviceSynchronize();//To synchronize the device

    // Copy the results in GPU memory back to the CPU
    SAFE_CALL(cudaMemcpy(hostC, deviceC, sizeof(float)*numCRows*numCColumns, cudaMemcpyDeviceToHost));

    printf("\nMatrix C From Device\n");
    Print_Mat(numCRows,numCColumns,hostC);//Function Call

    matMultiplyOnHost(hostA, hostB, hostComputedC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    printf("\nMatrix C From Host\n");
    Print_Mat(numCRows,numCColumns,hostComputedC);//Function Call

    for (int i=0; i < numCColumns*numCRows; i++)//Compare both the result matrices 1. MatrixMultiplyonHost 2. MatrixMultiplyonDevice
    {
        if (hostComputedC[i]  != hostC[i] )
        {
            printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i / numCColumns, i % numCColumns, hostComputedC[i], hostC[i]);
            break;
        }
    }

    printf("\n Number of Blocks Created:%d \n",((numCColumns/TILESIZE) + 1)*((numCColumns/TILESIZE) + 1));
    printf("\n Number of Threads Per Block: %d \n",(TILESIZE*TILESIZE));

    // Free the GPU memory
    SAFE_CALL(cudaFree(deviceA));
    SAFE_CALL(cudaFree(deviceB));
    SAFE_CALL(cudaFree(deviceC));
    //Free the Pointer Memory
    free(hostA);
    free(hostB);
    free(hostC);
    free(hostComputedC);

    return 0;
}
