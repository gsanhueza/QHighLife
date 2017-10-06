#ifndef _CUDAMODEL_H_
#define _CUDAMODEL_H_

#include "model.h"

class CUDAModel : public Model
{
public:
    CUDAModel(int width, int height);
    ~CUDAModel();

//     virtual void setup();
    virtual void run();
    virtual int runStressTest(int timeInSeconds);
    virtual int runStressTestVariantIf(int timeInSeconds);
    virtual int runStressTestVariantNonIf(int timeInSeconds);
    virtual int runStressTestVariant32(int timeInSeconds);
    virtual int runStressTestVariantNon32(int timeInSeconds);

protected:
    virtual void setup();
};

#endif
