#include "cudamodel.h"
extern "C"
void cuda_main();

CUDAModel::CUDAModel()
{
}

CUDAModel::~CUDAModel()
{
}

void CUDAModel::run()
{
    cuda_main();
}
