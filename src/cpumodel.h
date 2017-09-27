#ifndef _CPUMODEL_H_
#define _CPUMODEL_H_

#include "model.h"
#include <cmath>

class CPUModel : public Model
{
public:
    CPUModel(int width, int height);
    ~CPUModel();

    virtual void run();

private:
    int surroundingAliveCells(int i, int j);
};

#endif
