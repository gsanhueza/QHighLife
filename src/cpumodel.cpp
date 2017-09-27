#include "cpumodel.h"

using namespace std;

CPUModel::CPUModel(int width, int height) :
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
    result = *m_grid;

    for (int j = 0; j < m_grid->getHeight(); j++)
    {
        for (int i = 0; i < m_grid->getWidth(); i++)
        {
            // Not 2 or 3 cells surrounding this alive cell = Cell dies
            if (m_grid->getAt(i, j) and not (surroundingAliveCells(i, j) == 2 or surroundingAliveCells(i, j) == 3))
            {
                result.setAt(i, j, false);
            }
            // Dead cell surrounded by 3 or 6 cells = Cell revives
            else if (not m_grid->getAt(i, j) and (surroundingAliveCells(i, j) == 3 or surroundingAliveCells(i, j) == 6))
            {
                result.setAt(i, j, true);
            }
        }
    }

    *m_grid = result;
}

int CPUModel::surroundingAliveCells(int i, int j)
{
    int count = 0;

    for (int y = std::max(0, j - 1); y <= std::min(j + 1, m_grid->getHeight() - 1); y++)
    {
        for (int x = std::max(0, i - 1); x <= std::min(i + 1, m_grid->getWidth() - 1); x++)
        {
            if (x == i and y == j) continue;                // Self check unrequired
            count += (m_grid->getAt(x, y) ? 1 : 0);         // Count alive cells
        }
    }
    return count;
}
