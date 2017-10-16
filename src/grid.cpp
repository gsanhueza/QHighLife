#include "grid.h"

Grid::Grid(int width, int height) :
    m_height(height),
    m_width(width)
{
    m_grid = new Cell[m_width * m_height];

    for (int i = 0; i < m_width * m_height; i++)
    {
        m_grid[i] = false;
    }
}

Grid::~Grid()
{
    delete [] m_grid;
}

Grid& Grid::operator=(const Grid& other)
{
    for (int i = 0; i < m_width * m_height; i++)
    {
        m_grid[i] = other.m_grid[i];
    }
    return *this;
}

bool Grid::operator==(const Grid& other) const
{
    for (int i = 0; i < m_width * m_height; i++)
    {
        if (m_grid[i] != other.m_grid[i])
        {
            return false;
        }
    }
    return true;
}

inline int Grid::getPosAt(int i, int j, int n) const
{
    return i + n * j;
}

int Grid::getWidth() const
{
    return m_width;
}

int Grid::getHeight() const
{
    return m_height;
}

bool Grid::getAt(int x, int y) const
{
    return m_grid[getPosAt(x, y, m_width)];
}

void Grid::setAt(int x, int y, Cell value)
{
    m_grid[getPosAt(x, y, m_width)] = value;
}
