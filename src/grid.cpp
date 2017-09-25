#include "grid.h"

Grid::Grid()
{

}

Grid::Grid(const Grid& other)
{

}

Grid::~Grid()
{

}

Grid& Grid::operator=(const Grid& other)
{
    return *this;
}

bool Grid::operator==(const Grid& other) const
{
    return false;
}

bool Grid::getAt(unsigned int x, unsigned int y) const
{
    return false;
}

void Grid::setAt(unsigned int x, unsigned int y, bool value)
{
}
