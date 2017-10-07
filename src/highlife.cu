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
    // Positions
    int Nx = i;
    int Ex = (i + 1) % w;
    int Sx = i;
    int Wx = (i + w - 1) % w;

    int Ny = (j + h - 1) % h;
    int Ey = j;
    int Sy = (j + 1) % h;
    int Wy = j;

    // Cell values
    int N = grid[getPos(Nx, Ny, w)];
    int E = grid[getPos(Ex, Ey, w)];
    int S = grid[getPos(Sx, Sy, w)];
    int W = grid[getPos(Wx, Wy, w)];

    int NW = grid[getPos(Wx, Ny, w)];
    int NE = grid[getPos(Ex, Ny, w)];
    int SW = grid[getPos(Wx, Sy, w)];
    int SE = grid[getPos(Ex, Sy, w)];

    return NW + N + NE + W + E + SW + S + SE;
}

__device__ int surroundingAliveCellsIf(bool *grid, int i, int j, int w, int h)
{
    int count = 0;

    // Positions
    int Nx = i;
    int Ex = (i + 1) % w;
    int Sx = i;
    int Wx = (i + w - 1) % w;

    int Ny = (j + h - 1) % h;
    int Ey = j;
    int Sy = (j + 1) % h;
    int Wy = j;

    // Cell values
    if (grid[getPos(Nx, Ny, w)])
        count++;
    if (grid[getPos(Ex, Ey, w)])
        count++;
    if (grid[getPos(Sx, Sy, w)])
        count++;
    if (grid[getPos(Wx, Wy, w)])
        count++;

    if (grid[getPos(Wx, Ny, w)])
        count++;
    if (grid[getPos(Ex, Ny, w)])
        count++;
    if (grid[getPos(Wx, Sy, w)])
        count++;
    if (grid[getPos(Ex, Sy, w)])
        count++;

    return count;
}

// Kernels
__global__ void computeHighLife(bool *grid, bool *result, int width, int height)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    int j = (blockDim.y * blockIdx.y) + threadIdx.y;

    if (i < width and j < height)                           // Inside the matrix
    {
        bool currentCell = grid[getPos(i, j, width)];
        int surroundingAliveCellsNumber = surroundingAliveCells(grid, i, j, width, height);

        bool a = currentCell;
        bool b = surroundingAliveCellsNumber == 2;
        bool c = surroundingAliveCellsNumber == 3;
        bool d = surroundingAliveCellsNumber == 6;

        // Not 2 or 3 cells surrounding this alive cell = Cell dies
        if (a and not (b or c))
        {
            result[getPos(i, j, width)] = 0;
        }
        // Dead cell surrounded by 3 or 6 cells = Cell revives
        else if (not a and (c or d))
        {
            result[getPos(i, j, width)] = 1;
        }
        else
        {
            result[getPos(i, j, width)] = a;
        }
    }
}

__global__ void computeHighLifeIf(bool *grid, bool *result, int width, int height)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    int j = (blockDim.y * blockIdx.y) + threadIdx.y;

    if (i < width and j < height)                           // Inside the matrix
    {
        bool currentCell = grid[getPos(i, j, width)];
        int surroundingAliveCellsNumber = surroundingAliveCellsIf(grid, i, j, width, height);

        bool a = currentCell;
        bool b = surroundingAliveCellsNumber == 2;
        bool c = surroundingAliveCellsNumber == 3;
        bool d = surroundingAliveCellsNumber == 6;

        // Not 2 or 3 cells surrounding this alive cell = Cell dies
        if (a and not (b or c))
        {
            result[getPos(i, j, width)] = 0;
        }
        // Dead cell surrounded by 3 or 6 cells = Cell revives
        else if (not a and (c or d))
        {
            result[getPos(i, j, width)] = 1;
        }
        else
        {
            result[getPos(i, j, width)] = a;
        }
    }
}

// CUDA setup
extern "C"
void cuda_setup(Grid *grid)
{
    // Set grid dimensions
    GRID_SIZE.x = grid->getWidth();
    GRID_SIZE.y = grid->getHeight();

    // 64 threads per block
    THREADS_PER_BLOCK.x = 8;
    THREADS_PER_BLOCK.y = 8;

    // Set block dimensions
    NUM_BLOCKS.x = (GRID_SIZE.x + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x;
    NUM_BLOCKS.y = (GRID_SIZE.y + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y;
    // Host data
    h_grid   = new bool[GRID_SIZE.x * GRID_SIZE.y];
    h_result = new bool[GRID_SIZE.x * GRID_SIZE.y];

    // Device data
    cudaMalloc(&d_grid, GRID_SIZE.x * GRID_SIZE.y * sizeof(bool));
    cudaMalloc(&d_result, GRID_SIZE.x * GRID_SIZE.y * sizeof(bool));

}

// CUDA cleanup routine
extern "C"
void cuda_cleanup()
{
    delete h_grid;
    delete h_result;
    cudaFree(&d_grid);
    cudaFree(&d_result);
}

// CUDA main
extern "C"
int cuda_main(Grid *grid)
{
    // Data filling
    for (int j = 0; j < GRID_SIZE.y; j++)
    {
        for (int i = 0; i < GRID_SIZE.x; i++)
        {
            h_grid[getPos(i, j, GRID_SIZE.x)] = grid->getAt(i, j);
            h_result[getPos(i, j, GRID_SIZE.x)] = 0;
        }
    }

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_grid, h_grid, GRID_SIZE.x * GRID_SIZE.y * sizeof(bool), cudaMemcpyHostToDevice);

    // Send kernel
    computeHighLife<<< NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_grid, d_result, GRID_SIZE.x, GRID_SIZE.y);

    // Copy results from device memory to host memory
    cudaMemcpy(h_result, d_result, GRID_SIZE.x * GRID_SIZE.y * sizeof(bool), cudaMemcpyDeviceToHost);

    // Update grid
    for (int j = 0; j < GRID_SIZE.y; j++)
    {
        for (int i = 0; i < GRID_SIZE.x; i++)
        {
            grid->setAt(i, j, h_result[getPos(i, j, GRID_SIZE.x)]);
        }
    }

    return 0;
}

// CUDA Stress test
extern "C"
int cuda_main_stress(Grid *grid, int timeInSeconds)
{
    // Data filling
    for (int j = 0; j < GRID_SIZE.y; j++)
    {
        for (int i = 0; i < GRID_SIZE.x; i++)
        {
            h_grid[getPos(i, j, GRID_SIZE.x)] = grid->getAt(i, j);
            h_result[getPos(i, j, GRID_SIZE.x)] = 0;
        }
    }

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_grid, h_grid, GRID_SIZE.x * GRID_SIZE.y * sizeof(bool), cudaMemcpyHostToDevice);

    std::chrono::time_point<std::chrono::high_resolution_clock> m_start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end = m_start + std::chrono::seconds(timeInSeconds);
    int iterations = 0;

    while (std::chrono::high_resolution_clock::now() < m_end)
    {
        // Optimization: We expect d_grid to be READONLY, and d_result to be WRITEONLY.
        // We start with d_grid == d_result.
        // When we finish the computation once, we (theoretically) want to update d_grid. => d_grid will be the same as d_result.
        // If we (temporarily) use d_result as d_grid in each second computation, we'll get the same "start".
        // Thus, our final results will be in d_grid. => We have to copy back d_grid to h_result to get the real result.
        // With this, we can avoid calling cudaMemcpy every iteration.

        // Send kernel: Results will be in d_result
        computeHighLife<<< NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_grid, d_result, GRID_SIZE.x, GRID_SIZE.y);

        // Send kernel: Results will be in d_grid
        computeHighLife<<< NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_result, d_grid, GRID_SIZE.x, GRID_SIZE.y);

        iterations += 2;
    }

    // Copy results from device memory to host memory (Check note above to see why our results are in d_grid instead of d_result.)
    cudaMemcpy(h_result, d_grid, GRID_SIZE.x * GRID_SIZE.y * sizeof(bool), cudaMemcpyDeviceToHost);

    // Update grid
    for (int j = 0; j < GRID_SIZE.y; j++)
    {
        for (int i = 0; i < GRID_SIZE.x; i++)
        {
            grid->setAt(i, j, h_result[getPos(i, j, GRID_SIZE.x)]);
        }
    }

    return iterations;
}

// VARIANTS
extern "C"
int cuda_main_stress_if(Grid *grid, int timeInSeconds)
{
    // Data filling
    for (int j = 0; j < GRID_SIZE.y; j++)
    {
        for (int i = 0; i < GRID_SIZE.x; i++)
        {
            h_grid[getPos(i, j, GRID_SIZE.x)] = grid->getAt(i, j);
            h_result[getPos(i, j, GRID_SIZE.x)] = 0;
        }
    }

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_grid, h_grid, GRID_SIZE.x * GRID_SIZE.y * sizeof(bool), cudaMemcpyHostToDevice);

    std::chrono::time_point<std::chrono::high_resolution_clock> m_start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end = m_start + std::chrono::seconds(timeInSeconds);
    int iterations = 0;

    while (std::chrono::high_resolution_clock::now() < m_end)
    {
        // Optimization: We expect d_grid to be READONLY, and d_result to be WRITEONLY.
        // We start with d_grid == d_result.
        // When we finish the computation once, we (theoretically) want to update d_grid. => d_grid will be the same as d_result.
        // If we (temporarily) use d_result as d_grid in each second computation, we'll get the same "start".
        // Thus, our final results will be in d_grid. => We have to copy back d_grid to h_result to get the real result.
        // With this, we can avoid calling cudaMemcpy every iteration.

        // Send kernel: Results will be in d_result
        computeHighLifeIf<<< NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_grid, d_result, GRID_SIZE.x, GRID_SIZE.y);

        // Send kernel: Results will be in d_grid
        computeHighLifeIf<<< NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_result, d_grid, GRID_SIZE.x, GRID_SIZE.y);

        iterations += 2;
    }

    // Copy results from device memory to host memory (Check note above to see why our results are in d_grid instead of d_result.)
    cudaMemcpy(h_result, d_grid, GRID_SIZE.x * GRID_SIZE.y * sizeof(bool), cudaMemcpyDeviceToHost);

    // Update grid
    for (int j = 0; j < GRID_SIZE.y; j++)
    {
        for (int i = 0; i < GRID_SIZE.x; i++)
        {
            grid->setAt(i, j, h_result[getPos(i, j, GRID_SIZE.x)]);
        }
    }

    return iterations;
}

extern "C"
int cuda_main_stress_non_if(Grid *grid, int timeInSeconds)
{
    return cuda_main_stress(grid, timeInSeconds);
}

extern "C"
int cuda_main_stress_32(Grid *grid, int timeInSeconds)
{
    return cuda_main_stress(grid, timeInSeconds);
}

extern "C"
int cuda_main_stress_non_32(Grid *grid, int timeInSeconds)
{
    // 81 threads per block
    THREADS_PER_BLOCK.x = 9;
    THREADS_PER_BLOCK.y = 9;

    // Set block dimensions
    NUM_BLOCKS.x = (GRID_SIZE.x + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x;
    NUM_BLOCKS.y = (GRID_SIZE.y + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y;

    // Host data
    h_grid   = new bool[GRID_SIZE.x * GRID_SIZE.y];
    h_result = new bool[GRID_SIZE.x * GRID_SIZE.y];

    // Revert Device data from setup()
    cudaFree(&d_grid);
    cudaFree(&d_result);

    // Device data
    cudaMalloc(&d_grid, GRID_SIZE.x * GRID_SIZE.y * sizeof(bool));
    cudaMalloc(&d_result, GRID_SIZE.x * GRID_SIZE.y * sizeof(bool));

    // Data filling
    for (int j = 0; j < GRID_SIZE.y; j++)
    {
        for (int i = 0; i < GRID_SIZE.x; i++)
        {
            h_grid[getPos(i, j, GRID_SIZE.x)] = grid->getAt(i, j);
            h_result[getPos(i, j, GRID_SIZE.x)] = 0;
        }
    }

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_grid, h_grid, GRID_SIZE.x * GRID_SIZE.y * sizeof(bool), cudaMemcpyHostToDevice);

    std::chrono::time_point<std::chrono::high_resolution_clock> m_start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end = m_start + std::chrono::seconds(timeInSeconds);
    int iterations = 0;

    while (std::chrono::high_resolution_clock::now() < m_end)
    {
        // Optimization: We expect d_grid to be READONLY, and d_result to be WRITEONLY.
        // We start with d_grid == d_result.
        // When we finish the computation once, we (theoretically) want to update d_grid. => d_grid will be the same as d_result.
        // If we (temporarily) use d_result as d_grid in each second computation, we'll get the same "start".
        // Thus, our final results will be in d_grid. => We have to copy back d_grid to h_result to get the real result.
        // With this, we can avoid calling cudaMemcpy every iteration.

        // Send kernel: Results will be in d_result
        computeHighLife<<< NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_grid, d_result, GRID_SIZE.x, GRID_SIZE.y);

        // Send kernel: Results will be in d_grid
        computeHighLife<<< NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_result, d_grid, GRID_SIZE.x, GRID_SIZE.y);

        iterations += 2;
    }

    // Copy results from device memory to host memory (Check note above to see why our results are in d_grid instead of d_result.)
    cudaMemcpy(h_result, d_grid, GRID_SIZE.x * GRID_SIZE.y * sizeof(bool), cudaMemcpyDeviceToHost);

    // Update grid
    for (int j = 0; j < GRID_SIZE.y; j++)
    {
        for (int i = 0; i < GRID_SIZE.x; i++)
        {
            grid->setAt(i, j, h_result[getPos(i, j, GRID_SIZE.x)]);
        }
    }

    return iterations;
}
