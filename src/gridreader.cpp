#include "gridreader.h"
#include <iostream>

GridReader::GridReader()
{
}

GridReader::~GridReader()
{
}

bool GridReader::loadFile (Grid *grid, QString filepath)
{
    if (filepath == "")
    {
        return false;
    }

    QFile file(filepath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        std::cout << "Can't open " << filepath.toStdString() << std::endl;
        return false;
    }

    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine();
        std::cout << line.toStdString() << std::endl;
    }

    return true;
}
