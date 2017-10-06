#ifndef _MODEL_H_
#define _MODEL_H_

#include <QStringRef>
#include <QTextStream>
#include <QVector>
#include "grid.h"
#include "gridreader.h"

class Model
{
public:
    Model(int width, int height);
    virtual ~Model();

    virtual void setup() = 0;
    virtual void run() = 0;
    virtual int runStressTest(int timeInSeconds) = 0;
    virtual int runStressTestVariantIf(int timeInSeconds) = 0;
    virtual int runStressTestVariantNonIf(int timeInSeconds) = 0;
    virtual int runStressTestVariant32(int timeInSeconds) = 0;
    virtual int runStressTestVariantNon32(int timeInSeconds) = 0;
    virtual void setLoadedGrid(QVector<QString> data);
    Grid* getGrid();

    Grid *m_grid;
    QTextStream out;
};

#endif
