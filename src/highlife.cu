#include <iostream>
#include <cuda_runtime.h>
#include "grid.h"

// Kernel
__global__ void computeHighLife(const Grid *grid, Grid *result)
{
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     int j = blockDim.y * blockIdx.y + threadIdx.y;

}

// Cuda main
extern "C"
int cuda_main(Grid *grid)
{
    Grid *result = new Grid(grid->getWidth(), grid->getHeight());
    *result = *grid;

    cudaMallocManaged(&grid, sizeof(Grid));
    cudaMallocManaged(&result, sizeof(result));

    const int THREADS_PER_BLOCK = 256;
    const int NUMBER_OF_BLOCKS = 1;

    std::cout << "CUDA can receive a Grid object" << std::endl;
    computeHighLife<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(grid, result);
    std::cout << "CUDA can send a Grid object" << std::endl;
    *grid = *result;
    return 0;
}
