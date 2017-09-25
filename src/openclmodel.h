#ifndef _OPENCLMODEL_H_
#define _OPENCLMODEL_H_

#include "model.h"

class OpenCLModel : public Model
{
public:
    OpenCLModel();
    ~OpenCLModel();

    virtual void run();
};

#endif
