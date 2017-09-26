#include "cudamodel.h"
extern "C"
void cuda_main();

CUDAModel::CUDAModel(unsigned int width, unsigned int height) :
    Model(width, height)
{
}

CUDAModel::~CUDAModel()
{
}

void CUDAModel::run()
{
    cuda_main();
}
