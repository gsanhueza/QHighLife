#include "cpumodel.h"
#include <iostream>

using namespace std;

CPUModel::CPUModel(unsigned int width, unsigned int height) :
    Model(width, height)
{
}

CPUModel::~CPUModel()
{
}

// TODO Implementar CPU
void CPUModel::run()
{
    Grid result(m_grid->getWidth(), m_grid->getHeight());

    for (unsigned int i = 0; i < result.getWidth(); i++)
    {
        for (unsigned int j = 0; j < result.getHeight(); j++)
        {
            bool isAlive = m_grid->getAt(i, j);
            if (isAlive)
            {
                result.setAt(i, j, not isAlive);
            }
        }
    }
    *m_grid = result;

    // TODO Sacar check
    for (unsigned int i = 0; i < m_grid->getWidth(); i++)
    {
        for (unsigned int j = 0; j < m_grid->getHeight(); j++)
        {
            std::cout << m_grid->getAt(i, j);
        }
        std::cout << std::endl;
    }
}
