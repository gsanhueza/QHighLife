#include "cpumodel.h"

using namespace std;

CPUModel::CPUModel(int width, int height) :
    Model(width, height),
    m_result(new Grid(width, height))
{
}

CPUModel::~CPUModel()
{
    delete m_result;
}

void CPUModel::setup()
{
}

void CPUModel::run()
{
    for (int j = 0; j < m_grid->getHeight(); j++)
    {
        for (int i = 0; i < m_grid->getWidth(); i++)
        {
            // Not 2 or 3 cells surrounding this alive cell = Cell dies
            if (m_grid->getAt(i, j) and not (surroundingAliveCells(i, j) == 2 or surroundingAliveCells(i, j) == 3))
            {
                m_result->setAt(i, j, false);
            }
            // Dead cell surrounded by 3 or 6 cells = Cell revives
            else if (not m_grid->getAt(i, j) and (surroundingAliveCells(i, j) == 3 or surroundingAliveCells(i, j) == 6))
            {
                m_result->setAt(i, j, true);
            }
            else {
                m_result->setAt(i, j, m_grid->getAt(i, j));
            }
        }
    }

    *m_grid = *m_result;
}

int CPUModel::runStressTest(int timeInSeconds)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end = m_start + std::chrono::seconds(timeInSeconds);
    int iterations = 0;

    while (std::chrono::high_resolution_clock::now() < m_end)
    {
        run();
        ++iterations;
    }

    return iterations;
}

int CPUModel::surroundingAliveCells(int i, int j)
{
    int h = m_grid->getHeight();
    int w = m_grid->getWidth();

    // Positions
    int Nx = i;
    int Ex = (i + 1) % w;
    int Sx = i;
    int Wx = (i + w - 1) % w;

    int Ny = (j + h - 1) % h;
    int Ey = j;
    int Sy = (j + 1) % h;
    int Wy = j;

    // Cell values
    int N = m_grid->getAt(Nx, Ny);
    int E = m_grid->getAt(Ex, Ey);
    int S = m_grid->getAt(Sx, Sy);
    int W = m_grid->getAt(Wx, Wy);

    int NW = m_grid->getAt(Wx, Ny);
    int NE = m_grid->getAt(Ex, Ny);
    int SW = m_grid->getAt(Wx, Sy);
    int SE = m_grid->getAt(Ex, Sy);

    return NW + N + NE + W + E + SW + S + SE;
}
