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

    /**
    * @brief Setups the model (loads the grid in the model).
    *
    */
    virtual void setup() = 0;

    /**
    * @brief Runs the implementation.
    *
    */
    virtual void run() = 0;

    /**
    * @brief Runs the original implementation for the stress test.
    *
    * @param timeInSeconds p_timeInSeconds: Amount of seconds for the stress test.
    */
    virtual int runStressTest(int timeInSeconds) = 0;

    /**
    * @brief Runs the "Variant If" implementation for the stress test.
    *
    * @param timeInSeconds p_timeInSeconds: Amount of seconds for the stress test.
    */
    virtual int runStressTestVariantIf(int timeInSeconds) = 0;

    /**
    * @brief Runs the "Variant Non If" implementation for the stress test.
    * In this case, it's the original implementation.
    *
    * @param timeInSeconds p_timeInSeconds: Amount of seconds for the stress test.
    */
    virtual int runStressTestVariantNonIf(int timeInSeconds) = 0;

    /**
    * @brief Runs the "Variant 32" implementation for the stress test.
    * In this case, it's the original implementation.
    *
    * @param timeInSeconds p_timeInSeconds: Amount of seconds for the stress test.
    */
    virtual int runStressTestVariant32(int timeInSeconds) = 0;

    /**
    * @brief Runs the "Variant Non 32" implementation for the stress test.
    *
    * @param timeInSeconds p_timeInSeconds: Amount of seconds for the stress test.
    */
    virtual int runStressTestVariantNon32(int timeInSeconds) = 0;

    /**
    * @brief Sets the grid with the loaded data in data.
    *
    * @param data p_data: Each row of the loaded grid.
    */
    virtual void setLoadedGrid(QVector<QString> data);

    /**
    * @brief Returns the current grid.
    *
    * @return Grid* Current grid.
    */
    Grid* getGrid();

    Grid *m_grid;
    QTextStream out;
};

#endif
