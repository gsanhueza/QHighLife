#include "cudamodel.h"
extern "C"
void cuda_main(Grid *grid);

CUDAModel::CUDAModel(int width, int height) :
    Model(width, height)
{
}

CUDAModel::~CUDAModel()
{
}

void CUDAModel::run()
{
    out << "TODO: Implement CUDA Model" << endl;
    cuda_main(m_grid);
}
