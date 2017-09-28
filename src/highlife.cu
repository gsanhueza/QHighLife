#include <iostream>
#include <cuda_runtime.h>
#include "grid.h"
#include "stdio.h"

// Kernel
__global__ void computeHighLife(bool **grid, bool *result, int width, int height)
{
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     int j = blockDim.y * blockIdx.y + threadIdx.y;

    result[threadIdx.x] = 0;
//     if (i < getWidth(grid) and j < getHeight(grid) and i >= 0 and j >= 0)
//     {
//         setAt(result, i, j, !getAt(grid, i, j));
//     }
}

// Cuda main
extern "C"
int cuda_main(Grid *grid)
{
    bool **h_grid = (bool **)malloc(grid->getWidth() * grid->getHeight() * sizeof(bool));
    bool *h_result = (bool *)malloc(grid->getWidth() * grid->getHeight() * sizeof(bool));
    bool **d_grid;
    cudaMalloc(&d_grid, grid->getWidth() * grid->getHeight() * sizeof(bool));
    bool *d_result;
    cudaMalloc(&d_result, grid->getWidth() * grid->getHeight() * sizeof(bool));

    h_grid = grid->getInnerGrid();

    for (int j = 0; j < grid->getHeight(); j++)
    {
        for (int i = 0; i < grid->getWidth(); i++)
        {
            h_result[j * grid->getHeight() + i] = 1;
        }
    }

    std::cout << "Host listo" << std::endl;

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_grid, h_grid, grid->getWidth() * grid->getHeight() * sizeof(bool), cudaMemcpyHostToDevice);

    int gridSize(grid->getWidth() * grid->getHeight());

    std::cout << "CUDA can receive a Grid object?" << std::endl;

    computeHighLife<<< 1, gridSize >>>(d_grid, d_result, grid->getWidth(), grid->getHeight());

    // h_result contains the result in host memory
    cudaMemcpy(h_result, d_result, grid->getWidth() * grid->getHeight() * sizeof(bool), cudaMemcpyDeviceToHost);

    std::cout << "CUDA can send a Grid object?" << std::endl;

    for (int j = 0; j < grid->getHeight(); j++)
    {
        for (int i = 0; i < grid->getWidth(); i++)
        {
            grid->setAt(i, j, h_result[j * grid->getHeight() + i]);
        }
    }

    // Final result
    return 0;
}
