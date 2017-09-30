// Helper 2D -> 1D array
int getPos(int i, int j, int n)
{
    return i + n * j;
}

// Neighbour counter
int surroundingAliveCells(global bool *grid, int i, int j, int w, int h)
{
    int count = 0;

    for (int y = max(0, j - 1); y <= min(j + 1, h - 1); y++)
    {
        for (int x = max(0, i - 1); x <= min(i + 1, w - 1); x++)
        {
            if (x == i && y == j) continue;             // Self check unrequired
            count += (grid[getPos(x, y, w)]);           // Count alive cells
        }
    }

    return count;
}

// Kernel
kernel void computeHighLife(global bool *grid, global bool *result, int width, int height)
{
//     int i = (blockDim.x * blockIdx.x) + threadIdx.x; // CUDA Style
//     int j = (blockDim.y * blockIdx.y) + threadIdx.y;

//     int i = (get_local_size(0) * get_group_id(0)) + get_local_id(0); // OpenCL, CUDA Style
//     int j = (get_local_size(1) * get_group_id(1)) + get_local_id(1);

    int i = get_global_id(0);                               // Get x=0 dimension
    int j = get_global_id(1);                               // Get y=1 dimension

    if (i < width && j < height)                            // Iterate only over our data
    {
        // Not 2 or 3 cells surrounding this alive cell = Cell dies
        if (grid[getPos(i, j, width)] && !(surroundingAliveCells(grid, i, j, width, height) == 2 || surroundingAliveCells(grid, i, j, width, height) == 3))
        {
            result[getPos(i, j, width)] = 0;
        }
        // Dead cell surrounded by 3 or 6 cells = Cell revives
        else if (!grid[getPos(i, j, width)] && (surroundingAliveCells(grid, i, j, width, height) == 3 || surroundingAliveCells(grid, i, j, width, height) == 6))
        {
            result[getPos(i, j, width)] = 1;
        }
        else{
            result[getPos(i, j, width)] = grid[getPos(i, j, width)];
        }
    }
}
