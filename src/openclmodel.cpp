#include "openclmodel.h"

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <stdlib.h>

OpenCLModel::OpenCLModel(int width, int height) :
    Model(width, height)
{
}

OpenCLModel::~OpenCLModel()
{
}

// Helper 2D -> 1D array
int getPosCL(int i, int j, int n)
{
    return i + n * j;
}

void OpenCLModel::setup()
{
    m_platform_id = 0;
    m_device_id = 0;

    host_grid   = new bool[m_grid->getWidth() * m_grid->getHeight()];
    host_result = new bool[m_grid->getWidth() * m_grid->getHeight()];

    // Filling data
    for (int j = 0; j < m_grid->getHeight(); j++)
    {
        for (int i = 0; i < m_grid->getWidth(); i++)
        {
            host_grid[getPosCL(i, j, m_grid->getWidth())] = m_grid->getAt(i, j);
        }
    }

    // Query for platforms
    cl::Platform::get(&platforms);

    // Get a list of devices on this platform
    platforms[m_platform_id].getDevices(CL_DEVICE_TYPE_GPU|CL_DEVICE_TYPE_CPU, &devices); // Select the platform.

    // Create a context
    cl::Context context(devices);

    // Create a command queue
    queue = cl::CommandQueue( context, devices[m_device_id] );   // Select the device.

    // Create the memory buffers
    buffer_grid   = cl::Buffer(context, CL_MEM_READ_ONLY, m_grid->getWidth() * m_grid->getHeight() * sizeof(bool));
    buffer_result = cl::Buffer(context, CL_MEM_WRITE_ONLY, m_grid->getWidth() * m_grid->getHeight() * sizeof(bool));

    // Copy the input data to the input buffers using the command queue.
    queue.enqueueWriteBuffer( buffer_grid, CL_FALSE, 0, m_grid->getWidth() * m_grid->getHeight() * sizeof(bool), host_grid );

    // Read the program source
    std::ifstream sourceFile("../src/highlife.cl");
    std::string sourceCode( std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));

    // Make program from the source code
    program = cl::Program(context, source);

    // Build the program for the devices
    program.build(devices);
}

void OpenCLModel::run()
{
    setup();

    // Make kernel
    cl::Kernel highlife_kernel(program, "computeHighLife");

    // Set kernel dimensions
    cl::NDRange global( m_grid->getWidth(), m_grid->getHeight() );
    cl::NDRange local( 8, 8 );                          // 64 workitems per workgroup

    /* START_LAUNCH_KERNEL */
    // Set the kernel arguments
    highlife_kernel.setArg(0, buffer_grid);
    highlife_kernel.setArg(1, buffer_result);
    highlife_kernel.setArg(2, m_grid->getWidth());
    highlife_kernel.setArg(3, m_grid->getHeight());

    // Execute the kernel
    queue.enqueueNDRangeKernel( highlife_kernel, cl::NullRange, global, local );
    /* END_LAUNCH_KERNEL */

    // Copy the output data back to the host
    queue.enqueueReadBuffer( buffer_result, CL_TRUE, 0, m_grid->getWidth() * m_grid->getHeight() * sizeof(bool), host_result );

    // Set the result
    for (int j = 0; j < m_grid->getHeight(); j++)
    {
        for (int i = 0; i < m_grid->getWidth(); i++)
        {
            m_grid->setAt(i, j, host_result[getPosCL(i, j, m_grid->getWidth())]);
        }
    }


}

int OpenCLModel::runStressTest(int timeInSeconds)
{
    out << "FIXME: Implement OpenCL Model stress test" << endl;
    return 0;
}
