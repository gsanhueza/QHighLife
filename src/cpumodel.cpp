#include "cpumodel.h"

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
    out << "TODO: Implement CPU Model" << endl;

    Grid result(m_grid->getWidth(), m_grid->getHeight());
    for (unsigned int j = 0; j < m_grid->getHeight(); j++)
    {
        for (unsigned int i = 0; i < m_grid->getWidth(); i++)
        {
            result.setAt(i, j, not m_grid->getAt(i, j));
        }
    }

    *m_grid = result;
}
