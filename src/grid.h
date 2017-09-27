#ifndef _GRID_H_
#define _GRID_H_

typedef bool Cell;

class Grid
{
public:
    Grid(int width, int height);
    ~Grid();
    Grid& operator=(const Grid& other);
    bool operator==(const Grid& other) const;

    int getWidth() const;
    int getHeight() const;

    bool getAt(int x, int y) const;
    void setAt(int x, int y, Cell value);

private:
    bool **m_grid;
    int m_height;
    int m_width;
};

#endif
