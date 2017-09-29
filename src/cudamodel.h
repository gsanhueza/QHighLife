#ifndef _CUDAMODEL_H_
#define _CUDAMODEL_H_

#include "model.h"

class CUDAModel : public Model
{
public:
    CUDAModel(int width, int height);
    ~CUDAModel();

    virtual void setup();
    virtual void run();
    virtual int runStressTest(int timeInSeconds);

protected:
    virtual void setup();
};

#endif
