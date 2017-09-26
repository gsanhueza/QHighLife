#ifndef _MODEL_H_
#define _MODEL_H_

#include <QStringRef>
#include <QVector>
#include "grid.h"
#include "gridreader.h"

class Model
{
public:
    Model(unsigned int width, unsigned int height);
    virtual ~Model();

    virtual void run() = 0;
    virtual void setLoadedGrid(QVector<QString> data);

protected:
    Grid *m_grid = nullptr;
};

#endif
