#ifndef _GRIDREADER_H_
#define _GRIDREADER_H_

#include <QFile>
#include <QStringRef>
#include <QTextStream>

#include <vector>
#include <sstream>

using namespace std;

class GridReader
{
public:
    GridReader();
    ~GridReader();

    bool loadFile(QString filepath);
};

#endif
