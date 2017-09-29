#ifndef _CPUMODEL_H_
#define _CPUMODEL_H_

#include "model.h"
#include <cmath>

class CPUModel : public Model
{
public:
    CPUModel(int width, int height);
    ~CPUModel();

//     virtual void setup();
    virtual void run();
    virtual int runStressTest(int timeInSeconds);

protected:
    virtual void setup();

private:
    int surroundingAliveCells(int i, int j);
    Grid *m_result;
};

#endif
