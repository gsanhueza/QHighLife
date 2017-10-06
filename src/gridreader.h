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

#ifndef _GRIDREADER_H_
#define _GRIDREADER_H_

#include <QFile>
#include <QStringRef>
#include <QTextStream>
#include <QVector>

#include "grid.h"

class GridReader
{
public:
    GridReader();
    ~GridReader();

    bool loadFile(QString filepath);
    int getDetectedWidth() const;
    int getDetectedHeight() const;
    QVector<QString> getData() const;

private:
    int m_detectedWidth;
    int m_detectedHeight;
    QVector<QString> m_data;
};

#endif
