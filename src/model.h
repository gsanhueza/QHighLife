#ifndef _MODEL_H_
#define _MODEL_H_

#include "grid.h"

class Model
{
public:
    Model(unsigned int width, unsigned int height);
    virtual ~Model();

    virtual void run() = 0;

protected:
    Grid *m_grid;

};

#endif
