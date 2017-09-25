#ifndef _GRID_H_
#define _GRID_H_

class Grid
{
public:
    Grid();
    Grid(const Grid& other);
    ~Grid();
    Grid& operator=(const Grid& other);
    bool operator==(const Grid& other) const;

    bool getAt(unsigned int x, unsigned int y) const;
    void setAt(unsigned int x, unsigned int y, bool value);

private:
    bool *m_grid;
    unsigned int m_height;
    unsigned int m_width;
};

#endif
