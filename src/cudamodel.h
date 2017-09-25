#ifndef _CUDAMODEL_H_
#define _CUDAMODEL_H_

#include "model.h"

class CUDAModel : public Model
{
public:
    CUDAModel();
    ~CUDAModel();

    virtual void run();
};

#endif
