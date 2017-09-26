#ifndef _CPUMODEL_H_
#define _CPUMODEL_H_

#include "model.h"

class CPUModel : public Model
{
public:
    CPUModel(unsigned int width, unsigned int height);
    ~CPUModel();

    virtual void run();
};

#endif
