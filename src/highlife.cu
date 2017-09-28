#include <iostream>
#include <cuda_runtime.h>
#include "grid.h"
#include "stdio.h"

// Helper 2D -> 1D array
__host__ __device__ int getPos(int i, int j, int n)
{
    return i + n * j;
}

__device__ int surroundingAliveCells(bool *grid, int i, int j, int w, int h)
{
    int count = 0;

    for (int y = max(0, j - 1); y <= min(j + 1, h - 1); y++)
    {
        for (int x = max(0, i - 1); x <= min(i + 1, w - 1); x++)
        {
            if (x == i and y == j) continue;                // Self check unrequired
            count += (grid[getPos(x, y, w)] ? 1 : 0);        // Count alive cells
        }
    }

    return count;
}

// Kernel
__global__ void computeHighLife(bool *grid, bool *result, int width, int height)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    int j = (blockDim.y * blockIdx.y) + threadIdx.y;

    if (i < width and j < height)                           // Caso no-multiplo de 2
    {
        // Not 2 or 3 cells surrounding this alive cell = Cell dies
        if (grid[getPos(i, j, width)] and not(surroundingAliveCells(grid, i, j, width, height) == 2 or surroundingAliveCells(grid, i, j, width, height) == 3))
        {
            result[getPos(i, j, width)] = 0;                // FIXME Nadie llega aquí
        }
        // Dead cell surrounded by 3 or 6 cells = Cell revives
        else if (not grid[getPos(i, j, width)] and (surroundingAliveCells(grid, i, j, width, height) == 3 or surroundingAliveCells(grid, i, j, width, height) == 6))
        {
            result[getPos(i, j, width)] = 1;
        }
        else{
            result[getPos(i, j, width)] = grid[getPos(i, j, width)];
        }
    }
}

// Cuda main
extern "C"
int cuda_main(Grid *grid)
{
    // Host data
    bool *h_grid   = (bool *)malloc(grid->getWidth() * grid->getHeight() * sizeof(bool));
    bool *h_result = (bool *)malloc(grid->getWidth() * grid->getHeight() * sizeof(bool));

    // Filling data
    for (int j = 0; j < grid->getHeight(); j++)
    {
        for (int i = 0; i < grid->getWidth(); i++)
        {
            h_grid[getPos(i, j, grid->getWidth())] = grid->getAt(i, j);
            h_result[getPos(i, j, grid->getWidth())] = 0;
        }
    }

    std::cout << "Host is ready." << std::endl;

    // Device data
    bool *d_grid;
    cudaMalloc(&d_grid, grid->getWidth() * grid->getHeight() * sizeof(bool));
    bool *d_result;
    cudaMalloc(&d_result, grid->getWidth() * grid->getHeight() * sizeof(bool));

    std::cout << "Device is initialized." << std::endl;

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_grid, h_grid, grid->getWidth() * grid->getHeight() * sizeof(bool), cudaMemcpyHostToDevice);

    // Set grid and bock dimensions
    const int THREADS = grid->getWidth() * grid->getHeight();
    const dim3 THREADS_PER_BLOCK(8, 8);                     // 64 threads per block
    const dim3 NUM_BLOCKS(  (grid->getWidth() + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x,
                            (grid->getHeight() + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y);

    std::cout << "THREADS = " << THREADS << std::endl;
    std::cout << "THREADS_PER_BLOCK.x = " << THREADS_PER_BLOCK.x << std::endl;
    std::cout << "THREADS_PER_BLOCK.y = " << THREADS_PER_BLOCK.y << std::endl;
    std::cout << "NUM_BLOCKS.x = " << NUM_BLOCKS.x << std::endl;
    std::cout << "NUM_BLOCKS.y = " << NUM_BLOCKS.y << std::endl;
    std::cout << std::endl;

    // TODO Detectar máx threads por 1 bloque
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Max Threads per Block = " << deviceProp.maxThreadsPerBlock << std::endl;

    std::cout << "CUDA can receive a Grid object?" << std::endl;

    computeHighLife<<< NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_grid, d_result, grid->getWidth(), grid->getHeight());

    // h_result contains the result in host memory
    cudaMemcpy(h_result, d_result, grid->getWidth() * grid->getHeight() * sizeof(bool), cudaMemcpyDeviceToHost);

    std::cout << "CUDA can send a Grid object?" << std::endl;

    for (int j = 0; j < grid->getHeight(); j++)
    {
        for (int i = 0; i < grid->getWidth(); i++)
        {
            grid->setAt(i, j, h_result[getPos(i, j, grid->getWidth())]);
            std::cout << h_result[getPos(i, j, grid->getWidth())];
        }
        std::cout << std::endl;
    }

    // Final result
    return 0;
}
