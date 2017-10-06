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

#ifndef _CPUMODEL_H_
#define _CPUMODEL_H_

#include "model.h"
#include <cmath>

class CPUModel : public Model
{
public:
    CPUModel(int width, int height);
    ~CPUModel();

//     virtual void setup();
    virtual void run();
    virtual int runStressTest(int timeInSeconds);
    virtual int runStressTestVariantIf(int timeInSeconds);
    virtual int runStressTestVariantNonIf(int timeInSeconds);
    virtual int runStressTestVariant32(int timeInSeconds);
    virtual int runStressTestVariantNon32(int timeInSeconds);

protected:
    virtual void setup();

private:
    int surroundingAliveCells(int i, int j);
    Grid *m_result;
};

#endif
