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

#ifndef _OPENCLMODEL_H_
#define _OPENCLMODEL_H_

#include "model.h"

#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.cpp>
#else
#include <CL/cl.hpp>
#endif

class OpenCLModel : public Model
{
public:
    OpenCLModel(int width, int height);
    ~OpenCLModel();

    virtual void setup();
    virtual void run();
    virtual int runStressTest(int timeInSeconds);
    virtual int runStressTestVariantIf(int timeInSeconds);
    virtual int runStressTestVariantNonIf(int timeInSeconds);
    virtual int runStressTestVariant32(int timeInSeconds);
    virtual int runStressTestVariantNon32(int timeInSeconds);

private:
    int m_platform_id;
    int m_device_id;

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::CommandQueue queue;
    cl::Buffer buffer_grid;
    cl::Buffer buffer_result;
    cl::Program program;

    bool *host_grid;
    bool *host_result;
};

#endif
