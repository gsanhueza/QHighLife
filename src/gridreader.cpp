#include "gridreader.h"
#include <iostream>

GridReader::GridReader() :
    m_detectedWidth(0),
    m_detectedHeight(0)
{
}

GridReader::~GridReader()
{
}

bool GridReader::loadFile(QString filepath)
{
    std::cout << "GRIDREADER: Cargando " << filepath.toStdString() << std::endl;
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
        m_data.append(line);
        m_detectedWidth = line.size();
        m_detectedHeight++;
    }

    return true;
}

unsigned int GridReader::getDetectedWidth() const
{
    return m_detectedWidth;
}

unsigned int GridReader::getDetectedHeight() const
{
    return m_detectedHeight;
}

QVector<QString> GridReader::getData() const
{
    return m_data;
}
