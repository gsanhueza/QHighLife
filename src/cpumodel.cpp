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
    // TODO Sacar check
    for (unsigned int j = 0; j < m_grid->getHeight(); j++)
    {
        for (unsigned int i = 0; i < m_grid->getWidth(); i++)
        {
            std::cout << m_grid->getAt(i, j);
        }
        std::cout << std::endl;
    }
}
