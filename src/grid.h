#ifndef _GRID_H_
#define _GRID_H_

typedef bool Cell;

class Grid
{
public:
    Grid(unsigned int width, unsigned int height);
    ~Grid();
    Grid& operator=(const Grid& other);
    bool operator==(const Grid& other) const;

    bool getAt(unsigned int x, unsigned int y) const;
    void setAt(unsigned int x, unsigned int y, Cell value);

private:
    bool **m_grid;
    unsigned int m_height;
    unsigned int m_width;
};

#endif
