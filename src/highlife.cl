// Helper 2D -> 1D array
int getPos(int i, int j, int n)
{
    return i + n * j;
}

// Neighbour counter
int surroundingAliveCells(global bool *grid, int i, int j, int w, int h)
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
