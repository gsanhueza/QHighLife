#include "grid.h"

Grid::Grid(int width, int height) :
    m_height(height),
    m_width(width)
{
    m_grid = new Cell*[m_width];

    for (int i = 0; i < m_width; i++)
    {
        m_grid[i] = new Cell[m_height];
    }

    for (int i = 0; i < m_width; i++)
    {
        for (int j = 0; j < m_height; j++)
        {
            m_grid[i][j] = false;
        }
    }
}

Grid::~Grid()
{
    for (int i = 0; i < m_width; ++i)
    {
        delete m_grid[i];
    }
    delete [] m_grid;
}

Grid& Grid::operator=(const Grid& other)
{
    for (int i = 0; i < m_width; i++)
    {
        for (int j = 0; j < m_height; j++)
        {
            m_grid[i][j] = other.m_grid[i][j];
        }
    }
    return *this;
}

bool Grid::operator==(const Grid& other) const
{
    for (int i = 0; i < m_width; i++)
    {
        for (int j = 0; j < m_height; j++)
        {
            if (m_grid[i][j] != other.m_grid[i][j])
            {
                return false;
            }
        }
    }
    return true;
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
    return m_grid[x][y];
}

void Grid::setAt(int x, int y, Cell value)
{
    m_grid[x][y] = value;
}
