#include <iostream>
#include <cuda_runtime.h>
#include "grid.h"

// Kernel
__global__ void computeHighLife(const Grid *grid, Grid *result)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < grid->getWidth() and j < grid->getHeight())
    {
        result->setAt(i, j, !(grid->getAt(i, j)));
    }

}

// Cuda main
extern "C"
int cuda_main(Grid *grid)
{
    Grid *result = new Grid(grid->getWidth(), grid->getHeight());
    *result = *grid;

    cudaMallocManaged(&grid, sizeof(Grid));
    cudaMallocManaged(&result, sizeof(result));

    const int blocksize = 32;
    dim3 threads(blocksize, blocksize);
    dim3 cudagrid(grid->getWidth() / threads.x, grid->getHeight() / threads.y);

    std::cout << "CUDA can receive a Grid object" << std::endl;
    computeHighLife<<< cudagrid, threads >>>(grid, result);
    std::cout << "CUDA can send a Grid object" << std::endl;
    *grid = *result;
    return 0;
}
