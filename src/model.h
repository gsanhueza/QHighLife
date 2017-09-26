#ifndef _MODEL_H_
#define _MODEL_H_

#include <QStringRef>
#include "grid.h"
#include "gridreader.h"

class Model
{
public:
    Model(unsigned int width, unsigned int height);
    virtual ~Model();

    virtual void run() = 0;
    bool loadGrid(QString filepath);

protected:
    Grid *m_grid;
    GridReader m_gridreader;
};

#endif
