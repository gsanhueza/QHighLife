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

#ifndef _CANVAS_H_
#define _CANVAS_H_

#include <QWidget>
#include <QVector>
#include <QTextStream>
#include "grid.h"
#include "gridreader.h"

class Canvas : public QWidget
{
    Q_OBJECT;

public:
    Canvas(QWidget* parent = nullptr);
    ~Canvas();

public slots:
    void receiveGridReader(GridReader *gridReader);
    void receiveGrid(Grid *grid);

protected:
    void paintEvent(QPaintEvent *e) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    void doPainting();

    QVector<QString> m_data;
    Grid* m_grid;
    int m_gridWidth;
    int m_gridHeight;
    int m_cellWidth;
    int m_cellHeight;

    QTextStream out;
};

#endif
