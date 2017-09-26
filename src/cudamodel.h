#ifndef _CUDAMODEL_H_
#define _CUDAMODEL_H_

#include "model.h"

class CUDAModel : public Model
{
public:
    CUDAModel(unsigned int width, unsigned int height);
    ~CUDAModel();

    virtual void run();
};

#endif
