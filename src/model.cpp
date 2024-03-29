#include "model.h"

Model::Model(int width, int height) :
    m_grid(nullptr),
    out(stdout)
{
    m_grid = new Grid(width, height);
}

Model::~Model()
{
    delete m_grid;
}

void Model::setLoadedGrid(QVector<QString> data)
{
    for (int j = 0; j < data.size(); j++)
    {
        for (int i = 0; i < data.at(j).size(); i++)
        {
            m_grid->setAt(i, j, (data.at(j).at(i) == QChar('1')));
        }
    }
}

Grid* Model::getGrid()
{
    return m_grid;
}
