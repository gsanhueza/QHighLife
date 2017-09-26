#include "grid.h"
#include <iostream>

Grid::Grid(unsigned int width, unsigned int height) :
    m_height(height),
    m_width(width)
{
    m_grid = new Cell*[m_width];

    for (unsigned int i = 0; i < m_width; i++)
    {
        m_grid[i] = new Cell[m_height];
    }

    for (unsigned int i = 0; i < m_width; i++)
    {
        for (unsigned int j = 0; j < m_height; j++)
        {
            m_grid[i][j] = false;
        }
    }
}

Grid::~Grid()
{
    for (unsigned int i = 0; i < m_width; ++i)
    {
        delete m_grid[i];
    }
    delete [] m_grid;
}

Grid& Grid::operator=(const Grid& other)
{
    for (unsigned int i = 0; i < m_width; i++)
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
    for (unsigned int i = 0; i < m_width; i++)
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

unsigned int Grid::getWidth() const
{
    return m_width;
}

unsigned int Grid::getHeight() const
{
    return m_height;
}

bool Grid::getAt(unsigned int x, unsigned int y) const
{
    return m_grid[x][y];
}

void Grid::setAt(unsigned int x, unsigned int y, Cell value)
{
    m_grid[x][y] = value;
}
