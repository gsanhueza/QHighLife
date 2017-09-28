#include <iostream>
#include <cuda_runtime.h>
#include "grid.h"

__host__ __device__ bool getAt(Grid *grid, int i, int j)
{
    return grid->getAt(i, j);
}

__host__ __device__ void setAt(Grid *grid, int i, int j, bool value)
{
    grid->setAt(i, j, value);
}

__host__ __device__ int getWidth(Grid *grid)
{
    return grid->getWidth();
}

__host__ __device__ int getHeight(Grid *grid)
{
    return grid->getHeight();
}

// Kernel
__global__ void computeHighLife(Grid *grid, Grid *result)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    setAt(result, 1, 0, false);
//     if (i < getWidth(grid) and j < getHeight(grid) and i >= 0 and j >= 0)
//     {
//         setAt(result, i, j, !getAt(grid, i, j));
//     }
}

// Cuda main
extern "C"
int cuda_main(Grid *grid)
{
    Grid *mygrid;
    Grid *result;

    cudaMallocManaged(&mygrid, sizeof(mygrid));
    cudaMallocManaged(&result, sizeof(result));

    mygrid = new Grid(grid->getWidth(), grid->getHeight());
    result = new Grid(grid->getWidth(), grid->getHeight());

    *mygrid = *grid;
    *result = *grid;

    int blocksize = 32;
    dim3 threads(blocksize, blocksize);
    dim3 cudagrid(mygrid->getWidth() / threads.x, mygrid->getHeight() / threads.y);

    // FIXME Cuda puede hacer modificaciones (TODO), pero hay que ponerle ojo a los margenes, o hace segfault
    std::cout << "CUDA can receive a Grid object" << std::endl;
    std::cout << &mygrid << std::endl;
    std::cout << "mygrid.getAt(1, 0) was = " << std::boolalpha << mygrid->getAt(1, 0) << std::endl;
    std::cout << "result.getAt(1, 0) was = " << std::boolalpha << result->getAt(1, 0) << std::endl;
    computeHighLife<<< cudagrid, threads >>>(mygrid, result);
    std::cout << "CUDA can send a Grid object" << std::endl;
    std::cout << "mygrid.getAt(1, 0) is = " << std::boolalpha << mygrid->getAt(1, 0) << std::endl;
    std::cout << "result.getAt(1, 0) is = " << std::boolalpha << result->getAt(1, 0) << std::endl;

    // Final result
    *grid = *result;
    return 0;
}
