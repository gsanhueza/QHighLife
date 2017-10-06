#include "cudamodel.h"
extern "C"
void cuda_main(Grid *grid);

extern "C"
int cuda_main_stress(Grid *grid, int timeInSeconds);

extern "C"
int cuda_main_stress_if(Grid *grid, int timeInSeconds);

extern "C"
int cuda_main_stress_non_if(Grid *grid, int timeInSeconds);

extern "C"
int cuda_main_stress_32(Grid *grid, int timeInSeconds);

extern "C"
int cuda_main_stress_non_32(Grid *grid, int timeInSeconds);

extern "C"
void cuda_setup(Grid *grid);

extern "C"
void cuda_cleanup();

CUDAModel::CUDAModel(int width, int height) :
    Model(width, height)
{
}

CUDAModel::~CUDAModel()
{
    cuda_cleanup();
}

void CUDAModel::setup()
{
    cuda_setup(m_grid);
}

void CUDAModel::run()
{
    cuda_main(m_grid);
}

int CUDAModel::runStressTest(int timeInSeconds)
{
    return cuda_main_stress(m_grid, timeInSeconds);
}

int CUDAModel::runStressTestVariantIf(int timeInSeconds)
{
    return cuda_main_stress_if(m_grid, timeInSeconds);
}

int CUDAModel::runStressTestVariantNonIf(int timeInSeconds)
{
    return cuda_main_stress_non_if(m_grid, timeInSeconds);
}

int CUDAModel::runStressTestVariant32(int timeInSeconds)
{
    return cuda_main_stress_32(m_grid, timeInSeconds);
}

int CUDAModel::runStressTestVariantNon32(int timeInSeconds)
{
    return cuda_main_stress_non_32(m_grid, timeInSeconds);
}
