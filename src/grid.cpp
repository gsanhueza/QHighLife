#include "grid.h"

Grid::Grid(unsigned int width, unsigned int height) :
    m_height(height),
    m_width(width)
{
    m_grid = new Cell*[height];
    if (m_height)
    {
        m_grid[0] = new Cell[m_width * m_height];
        for (unsigned int i = 0; i < m_width; i++)
        {
            m_grid[i] = m_grid[0] + i * m_width;
        }
    }
}

Grid::~Grid()
{
    if (m_height)
    {
        delete [] m_grid[0];
    }
    delete [] m_grid;
}

Grid& Grid::operator=(const Grid& other)
{
    for (unsigned int i = 0; i < m_height; i++)
    {
        for (unsigned int j = 0; j < m_height; j++)
        {
            m_grid[i][j] = other.m_grid[i][j];
        }
    }
    return *this;
}

bool Grid::operator==(const Grid& other) const
{
    for (unsigned int i = 0; i < m_height; i++)
    {
        for (unsigned int j = 0; j < m_height; j++)
        {
            if (m_grid[i][j] != other.m_grid[i][j])
            {
                return false;
            }
        }
    }
    return true;
}

bool Grid::getAt(unsigned int x, unsigned int y) const
{
    return m_grid[x][y];
}

void Grid::setAt(unsigned int x, unsigned int y, Cell value)
{
    m_grid[x][y] = value;
}
