#include "cudamodel.h"
extern "C"
void cuda_main(Grid *grid);

extern "C"
int cuda_main_stress(Grid *grid, int timeInSeconds);

CUDAModel::CUDAModel(int width, int height) :
    Model(width, height)
{
}

CUDAModel::~CUDAModel()
{
}

void CUDAModel::run()
{
    cuda_main(m_grid);
}

int CUDAModel::runStressTest(int timeInSeconds)
{
    return cuda_main_stress(m_grid, timeInSeconds);
}
