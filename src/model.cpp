#include "model.h"

Model::Model(unsigned int width, unsigned int height)
{
    m_grid = new Grid(width, height);
}

Model::~Model()
{
    delete m_grid;
}

bool Model::loadGrid(QString filepath)
{
    return m_gridreader.loadFile(filepath);
}
