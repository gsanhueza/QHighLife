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
