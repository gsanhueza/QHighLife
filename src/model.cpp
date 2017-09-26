#include "model.h"
#include <iostream>

Model::Model(unsigned int width, unsigned int height)
{
    m_grid = new Grid(width, height);
}

Model::~Model()
{
    delete m_grid;
}

void Model::setLoadedGrid(QVector<QString> data)
{
    for (int i = 0; i < data.size(); i++)                   // Each line has its width
    {
        for (int j = 0; j < data.at(i).size(); j++)
        {
            m_grid->setAt(i, j, (data.at(i).at(j) == QChar('1')));
        }
    }
}
