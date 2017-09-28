#ifndef _GRIDREADER_H_
#define _GRIDREADER_H_

#include <QFile>
#include <QStringRef>
#include <QTextStream>
#include <QVector>

#include "grid.h"

class GridReader
{
public:
    GridReader();
    ~GridReader();

    bool loadFile(QString filepath);
    int getDetectedWidth() const;
    int getDetectedHeight() const;
    QVector<QString> getData() const;

private:
    int m_detectedWidth;
    int m_detectedHeight;
    QVector<QString> m_data;
};

#endif
