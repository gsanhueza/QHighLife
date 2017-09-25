#ifndef _MODEL_H_
#define _MODEL_H_

#include "grid.h"

class Model
{
public:
    Model();
    ~Model();

    virtual void run() = 0;

protected:
    Grid m_grid;

};

#endif
