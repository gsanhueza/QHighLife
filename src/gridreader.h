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

    bool loadFile(QString filepath);
    unsigned int getDetectedWidth() const;
    unsigned int getDetectedHeight() const;

private:
    unsigned int m_detectedWidth;
    unsigned int m_detectedHeight;
};

#endif
