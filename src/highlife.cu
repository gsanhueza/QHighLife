#include <iostream>
#include <cuda_runtime.h>
#include "grid.h"

// Kernel
__global__ void computeHighLife(bool **grid, bool **result, int width, int height)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    result[1][0] = false;
//     if (i < getWidth(grid) and j < getHeight(grid) and i >= 0 and j >= 0)
//     {
//         setAt(result, i, j, !getAt(grid, i, j));
//     }
}

// Cuda main
extern "C"
int cuda_main(Grid *grid)
{
    bool **innerGrid;
    bool **innerResult;

    cudaMallocManaged(&innerGrid, grid->getWidth() * grid->getHeight() * sizeof(bool));
    cudaMallocManaged(&innerGrid, grid->getWidth() * grid->getHeight() * sizeof(bool));

    innerGrid = grid->getInnerGrid();
    innerResult = grid->getInnerGrid();

    int blocksize = 32;
    dim3 threads(blocksize, blocksize);
    dim3 cudagrid(grid->getWidth() / threads.x, grid->getHeight() / threads.y);

    // FIXME Cuda puede hacer modificaciones (TODO), pero hay que ponerle ojo a los margenes, o hace segfault
    std::cout << "CUDA can receive a Grid object" << std::endl;
    std::cout << "mygrid.getAt(1, 0) was = " << std::boolalpha << innerGrid[1][0] << std::endl;
    std::cout << "result.getAt(1, 0) was = " << std::boolalpha << innerResult[1][0] << std::endl;
    computeHighLife<<< 1, 1 >>>(innerGrid, innerResult, grid->getWidth(),  grid->getHeight());
//     computeHighLife<<< cudagrid, threads >>>(innerGrid, innerResult, grid->getWidth(),  grid->getHeight());
    std::cout << "CUDA can send a Grid object" << std::endl;
    std::cout << "mygrid.getAt(1, 0) is = " << std::boolalpha << innerGrid[1][0] << std::endl;
    std::cout << "result.getAt(1, 0) is = " << std::boolalpha << innerResult[1][0] << std::endl;

    // Final result
    grid->setInnerGrid(innerResult);
    return 0;
}
