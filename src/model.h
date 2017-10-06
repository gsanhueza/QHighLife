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

#ifndef _MODEL_H_
#define _MODEL_H_

#include <QStringRef>
#include <QTextStream>
#include <QVector>
#include "grid.h"
#include "gridreader.h"

class Model
{
public:
    Model(int width, int height);
    virtual ~Model();

    virtual void setup() = 0;
    virtual void run() = 0;
    virtual int runStressTest(int timeInSeconds) = 0;
    virtual int runStressTestVariantIf(int timeInSeconds) = 0;
    virtual int runStressTestVariantNonIf(int timeInSeconds) = 0;
    virtual int runStressTestVariant32(int timeInSeconds) = 0;
    virtual int runStressTestVariantNon32(int timeInSeconds) = 0;
    virtual void setLoadedGrid(QVector<QString> data);
    Grid* getGrid();

    Grid *m_grid;
    QTextStream out;
};

#endif
