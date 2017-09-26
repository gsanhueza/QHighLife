#ifndef _GRIDREADER_H_
#define _GRIDREADER_H_

#include <QFile>
#include <QStringRef>
#include <QTextStream>

#include "grid.h"

class GridReader
{
public:
    GridReader();
    ~GridReader();

    bool loadFile(Grid *grid, QString filepath);
};

#endif
