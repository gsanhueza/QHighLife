int surroundingAliveCells(global bool *grid, int k, int w, int h)
{
    int count = 0;

    for (int y = max(k % w, k - w); y <= min(k + w, w * h - (w - (k % w))); y += w)
    {
        for (int x = max((k / w) * w, k - 1); x <= min(k + 1,(k / w) * w + (w - 1)); x++)
        {
            if (x == y) continue;                // Self check unrequired
            count += (grid[k] ? 1 : 0);        // Count alive cells
        }
    }

    return count;
}

// Kernel
kernel void computeHighLife(global bool *grid, global bool *result, int width, int height)
{
//     int i = (blockDim.x * blockIdx.x) + threadIdx.x;
//     int j = (blockDim.y * blockIdx.y) + threadIdx.y;

    int k = get_global_id(0);

    if (k < width * height)                           // Caso no-multiplo de 2
    {
//         // Not 2 or 3 cells surrounding this alive cell = Cell dies
//         if (grid[k] && !(surroundingAliveCells(grid, k, width, height) == 2 || surroundingAliveCells(grid, k, width, height) == 3))
//         {
//             result[k] = 0;
//         }
//         // Dead cell surrounded by 3 or 6 cells = Cell revives
//         else if (!grid[k] && (surroundingAliveCells(grid, k, width, height) == 3 || surroundingAliveCells(grid, k, width, height) == 6))
//         {
//             result[k] = 1;
//         }
//         else{
//             result[k] = grid[k];
//         }
        result[k] = !grid[k];
    }

}
