#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
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

// CUDA main
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

    // Set grid and block dimensions
    const int THREADS = grid->getWidth() * grid->getHeight();
    const dim3 THREADS_PER_BLOCK(8, 8);                     // 64 threads per block
    const dim3 NUM_BLOCKS(  (grid->getWidth() + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x,
                            (grid->getHeight() + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_grid, h_grid, grid->getWidth() * grid->getHeight() * sizeof(bool), cudaMemcpyHostToDevice);

    // Send kernel
    computeHighLife<<< NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_grid, d_result, grid->getWidth(), grid->getHeight());
    // Copy results from device memory to host memory
    cudaMemcpy(h_result, d_result, grid->getWidth() * grid->getHeight() * sizeof(bool), cudaMemcpyDeviceToHost);

    // Update grid
    for (int j = 0; j < grid->getHeight(); j++)
    {
        for (int i = 0; i < grid->getWidth(); i++)
        {
            grid->setAt(i, j, h_result[getPos(i, j, grid->getWidth())]);
        }
    }

    return 0;
}

// CUDA Stress test
extern "C"
int cuda_main_stress(Grid *grid, int timeInSeconds)
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

    // Set grid and block dimensions
    const int THREADS = grid->getWidth() * grid->getHeight();
    const dim3 THREADS_PER_BLOCK(8, 8);                     // 64 threads per block
    const dim3 NUM_BLOCKS(  (grid->getWidth() + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x,
        (grid->getHeight() + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y);

    std::chrono::time_point<std::chrono::high_resolution_clock> m_start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end = m_start + std::chrono::seconds(timeInSeconds);
    int iterations = 0;

    while (std::chrono::high_resolution_clock::now() < m_end)
    {
        // Copy vectors from host memory to device memory
        cudaMemcpy(d_grid, h_grid, grid->getWidth() * grid->getHeight() * sizeof(bool), cudaMemcpyHostToDevice);

        // Send kernel
        computeHighLife<<< NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_grid, d_result, grid->getWidth(), grid->getHeight());
        // Copy results from device memory to host memory
        cudaMemcpy(h_result, d_result, grid->getWidth() * grid->getHeight() * sizeof(bool), cudaMemcpyDeviceToHost);

        // Update grid
        for (int j = 0; j < grid->getHeight(); j++)
        {
            for (int i = 0; i < grid->getWidth(); i++)
            {
                h_grid[getPos(i, j, grid->getWidth())] = h_result[getPos(i, j, grid->getWidth())];
            }
        }

        ++iterations;
    }

    // Update grid
    for (int j = 0; j < grid->getHeight(); j++)
    {
        for (int i = 0; i < grid->getWidth(); i++)
        {
            grid->setAt(i, j, h_result[getPos(i, j, grid->getWidth())]);
        }
    }

    return iterations;
}
