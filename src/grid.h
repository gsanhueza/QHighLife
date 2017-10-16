/*
 * QHighLife is a High Life cellular-automata computing and visualization application using CPU and GPU.
 * Copyright (C) 2017  Gabriel Sanhueza <gabriel_8032@hotmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
    inline int getPosAt(int i, int j, int n) const;

private:
    bool *m_grid;
    int m_height;
    int m_width;
};

#endif
