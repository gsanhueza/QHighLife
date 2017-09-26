#include "cpumodel.h"
#include <iostream>

using namespace std;

CPUModel::CPUModel(unsigned int width, unsigned int height) :
    Model(width, height)
{
}

CPUModel::~CPUModel()
{
    delete m_grid;
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

    cout << "CPUModel run at (4, 2): " << boolalpha << m_grid->getAt(4, 2) << endl;
}
