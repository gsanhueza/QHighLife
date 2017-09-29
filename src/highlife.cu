#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include "grid.h"
#include "stdio.h"

// CUDA variables for setup
bool *h_grid;
bool *h_result;
bool *d_grid;
bool *d_result;

dim3 GRID_SIZE;
dim3 THREADS_PER_BLOCK;
dim3 NUM_BLOCKS;


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
            result[getPos(i, j, width)] = 0;
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

// CUDA setup
extern "C"
void cuda_setup(Grid *grid)
{
    // Host data
    h_grid   = new bool[grid->getWidth() * grid->getHeight()];
    h_result = new bool[grid->getWidth() * grid->getHeight()];

    // Device data
    cudaMalloc(&d_grid, grid->getWidth() * grid->getHeight() * sizeof(bool));
    cudaMalloc(&d_result, grid->getWidth() * grid->getHeight() * sizeof(bool));

    // Set grid dimensions
    GRID_SIZE.x = grid->getWidth();
    GRID_SIZE.y = grid->getHeight();

    // 64 threads per block
    THREADS_PER_BLOCK.x = 8;
    THREADS_PER_BLOCK.y = 8;

    // Set block dimensions
    NUM_BLOCKS.x = (GRID_SIZE.x + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x;
    NUM_BLOCKS.y = (GRID_SIZE.y + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y;
}

// CUDA main
extern "C"
int cuda_main(Grid *grid)
{
    // Data filling
    for (int j = 0; j < grid->getHeight(); j++)
    {
        for (int i = 0; i < grid->getWidth(); i++)
        {
            h_grid[getPos(i, j, grid->getWidth())] = grid->getAt(i, j);
            h_result[getPos(i, j, grid->getWidth())] = 0;
        }
    }

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_grid, h_grid, GRID_SIZE.x * GRID_SIZE.y * sizeof(bool), cudaMemcpyHostToDevice);

    // Send kernel
    computeHighLife<<< NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_grid, d_result, GRID_SIZE.x, GRID_SIZE.y);

    // Copy results from device memory to host memory
    cudaMemcpy(h_result, d_result, GRID_SIZE.x * GRID_SIZE.y * sizeof(bool), cudaMemcpyDeviceToHost);

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
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end = m_start + std::chrono::seconds(timeInSeconds);
    int iterations = 0;

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_grid, h_grid, grid->getWidth() * grid->getHeight() * sizeof(bool), cudaMemcpyHostToDevice);

    while (std::chrono::high_resolution_clock::now() < m_end)
    {
        // Optimization: We expect d_grid to be READONLY, and d_result to be WRITEONLY.
        // We start with d_grid == d_result.
        // When we finish the computation once, we (theoretically) want to update d_grid. => d_grid will be the same as d_result.
        // If we (temporarily) use d_result as d_grid in each second computation, we'll get the same "start".
        // Thus, our final results will be in d_grid. => We have to copy back d_grid to h_result to get the real result.
        // With this, we can avoid calling cudaMemcpy every iteration.

        // Send kernel: Results will be in d_result
        computeHighLife<<< NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_grid, d_result, grid->getWidth(), grid->getHeight());

        // Send kernel: Results will be in d_grid
        computeHighLife<<< NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_result, d_grid, grid->getWidth(), grid->getHeight());

        iterations += 2;
    }

    // Copy results from device memory to host memory (Check note above to see why our results are in d_grid instead of d_result.)
    cudaMemcpy(h_result, d_grid, GRID_SIZE.x * GRID_SIZE.y * sizeof(bool), cudaMemcpyDeviceToHost);

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
