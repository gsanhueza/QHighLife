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
    delete host_grid;
    delete host_result;
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

    // Query for platforms
    cl::Platform::get(&platforms);

    // Get a list of devices on this platform
    platforms[m_platform_id].getDevices(CL_DEVICE_TYPE_GPU|CL_DEVICE_TYPE_CPU, &devices); // Select the platform.

    // Create a context
    cl::Context context(devices);

    // Create a command queue
    queue = cl::CommandQueue( context, devices[m_device_id] );   // Select the device.

    // Create the memory buffers
    buffer_grid   = cl::Buffer(context, CL_MEM_READ_WRITE, m_grid->getWidth() * m_grid->getHeight() * sizeof(bool));
    buffer_result = cl::Buffer(context, CL_MEM_READ_WRITE, m_grid->getWidth() * m_grid->getHeight() * sizeof(bool));

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
    // Filling data
    for (int j = 0; j < m_grid->getHeight(); j++)
    {
        for (int i = 0; i < m_grid->getWidth(); i++)
        {
            host_grid[getPosCL(i, j, m_grid->getWidth())] = m_grid->getAt(i, j);
        }
    }

    // Copy the input data to the input buffers using the command queue.
    queue.enqueueWriteBuffer( buffer_grid, CL_FALSE, 0, m_grid->getWidth() * m_grid->getHeight() * sizeof(bool), host_grid );

    // Make kernel (Check arguments in template)
    cl::make_kernel<cl::Buffer, cl::Buffer, int, int> highlife_kernel(program, "computeHighLife");

    // Set kernel dimensions
    cl::NDRange global( m_grid->getWidth(), m_grid->getHeight() );
    cl::NDRange local( 8, 8 );                          // 64 workitems per workgroup

    // Set the kernel arguments and execute the kernel
    try {
        highlife_kernel(cl::EnqueueArgs(queue, global, local), buffer_grid, buffer_result, m_grid->getWidth(), m_grid->getHeight());
    }
    catch (cl::Error e)
    {
        std::cerr << "Error: " << e.what() << ". Input size is not divisible by kernel range." << std::endl;
    }

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
    int width = m_grid->getWidth();
    int height = m_grid->getHeight();

    // Filling data
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            host_grid[getPosCL(i, j, width)] = m_grid->getAt(i, j);
        }
    }

    // Copy the input data to the input buffers using the command queue.
    queue.enqueueWriteBuffer( buffer_grid, CL_FALSE, 0, width * height * sizeof(bool), host_grid );

    // Make kernel (Check arguments in template)
    cl::make_kernel<cl::Buffer&, cl::Buffer&, int, int> highlife_kernel(program, "computeHighLife");

    // Set kernel dimensions
    cl::NDRange global( width, height );
    cl::NDRange local( 8, 8 );                          // 64 workitems per workgroup

    std::chrono::time_point<std::chrono::high_resolution_clock> m_start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end = m_start + std::chrono::seconds(timeInSeconds);
    int iterations = 0;

    cl::EnqueueArgs eargs(queue, global, local);

    while (std::chrono::high_resolution_clock::now() < m_end)
    {
        // Optimization: We expect buffer_grid to be READONLY, and buffer_result to be WRITEONLY.
        // We start with buffer_grid == buffer_result.
        // When we finish the computation once, we (theoretically) want to update buffer_grid. => buffer_grid will be the same as buffer_result.
        // If we (temporarily) use buffer_result as buffer_grid in each second computation, we'll get the same "start".
        // Thus, our final results will be in buffer_grid. => We have to copy back buffer_grid to h_result to get the real result.
        // With this, we can avoid calling enqueueReadBuffer every iteration.

        highlife_kernel(eargs, buffer_grid, buffer_result, width, height);
        highlife_kernel(eargs, buffer_result, buffer_grid, width, height);

        iterations += 2;
    }

    // Copy the output data back to the host (Check note above to see why our results are in buffer_grid instead of buffer_result.)
    queue.enqueueReadBuffer( buffer_grid, CL_TRUE, 0, width * height * sizeof(bool), host_result );

    // Set the result
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            m_grid->setAt(i, j, host_result[getPosCL(i, j, width)]);
        }
    }

    return iterations;
}

int OpenCLModel::runStressTestVariantIf(int timeInSeconds)
{
    // Filling data
    for (int j = 0; j < m_grid->getHeight(); j++)
    {
        for (int i = 0; i < m_grid->getWidth(); i++)
        {
            host_grid[getPosCL(i, j, m_grid->getWidth())] = m_grid->getAt(i, j);
        }
    }

    // Copy the input data to the input buffers using the command queue.
    queue.enqueueWriteBuffer( buffer_grid, CL_FALSE, 0, m_grid->getWidth() * m_grid->getHeight() * sizeof(bool), host_grid );

    // Make kernel (Check arguments in template)
    cl::make_kernel<cl::Buffer, cl::Buffer, int, int> highlife_kernel(program, "computeHighLife");

    // Set kernel dimensions
    cl::NDRange global( m_grid->getWidth(), m_grid->getHeight() );
    cl::NDRange local( 8, 8 );                              // 64 workitems per workgroup

    std::chrono::time_point<std::chrono::high_resolution_clock> m_start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end = m_start + std::chrono::seconds(timeInSeconds);
    int iterations = 0;

    while (std::chrono::high_resolution_clock::now() < m_end)
    {
        // Optimization: We expect buffer_grid to be READONLY, and buffer_result to be WRITEONLY.
        // We start with buffer_grid == buffer_result.
        // When we finish the computation once, we (theoretically) want to update buffer_grid. => buffer_grid will be the same as buffer_result.
        // If we (temporarily) use buffer_result as buffer_grid in each second computation, we'll get the same "start".
        // Thus, our final results will be in buffer_grid. => We have to copy back buffer_grid to h_result to get the real result.
        // With this, we can avoid calling enqueueReadBuffer every iteration.

        highlife_kernel(cl::EnqueueArgs(queue, global, local), buffer_grid, buffer_result, m_grid->getWidth(), m_grid->getHeight());
        highlife_kernel(cl::EnqueueArgs(queue, global, local), buffer_result, buffer_grid, m_grid->getWidth(), m_grid->getHeight());

        iterations += 2;
    }

    // Copy the output data back to the host (Check note above to see why our results are in buffer_grid instead of buffer_result.)
    queue.enqueueReadBuffer( buffer_grid, CL_TRUE, 0, m_grid->getWidth() * m_grid->getHeight() * sizeof(bool), host_result );

    // Set the result
    for (int j = 0; j < m_grid->getHeight(); j++)
    {
        for (int i = 0; i < m_grid->getWidth(); i++)
        {
            m_grid->setAt(i, j, host_result[getPosCL(i, j, m_grid->getWidth())]);
        }
    }

    return iterations;
}

int OpenCLModel::runStressTestVariantNonIf(int timeInSeconds)
{
    return runStressTest(timeInSeconds);
}

int OpenCLModel::runStressTestVariant32(int timeInSeconds)
{
    return runStressTest(timeInSeconds);
}

int OpenCLModel::runStressTestVariantNon32(int timeInSeconds)
{
    // Filling data
    for (int j = 0; j < m_grid->getHeight(); j++)
    {
        for (int i = 0; i < m_grid->getWidth(); i++)
        {
            host_grid[getPosCL(i, j, m_grid->getWidth())] = m_grid->getAt(i, j);
        }
    }

    // Copy the input data to the input buffers using the command queue.
    queue.enqueueWriteBuffer( buffer_grid, CL_FALSE, 0, m_grid->getWidth() * m_grid->getHeight() * sizeof(bool), host_grid );

    // Make kernel (Check arguments in template)
    cl::make_kernel<cl::Buffer, cl::Buffer, int, int> highlife_kernel(program, "computeHighLife");

    // Set kernel dimensions
    cl::NDRange global( m_grid->getWidth(), m_grid->getHeight() );
    cl::NDRange local( 9, 9 );                              // 81 workitems per workgroup

    std::chrono::time_point<std::chrono::high_resolution_clock> m_start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end = m_start + std::chrono::seconds(timeInSeconds);
    int iterations = 0;

    while (std::chrono::high_resolution_clock::now() < m_end)
    {
        // Optimization: We expect buffer_grid to be READONLY, and buffer_result to be WRITEONLY.
        // We start with buffer_grid == buffer_result.
        // When we finish the computation once, we (theoretically) want to update buffer_grid. => buffer_grid will be the same as buffer_result.
        // If we (temporarily) use buffer_result as buffer_grid in each second computation, we'll get the same "start".
        // Thus, our final results will be in buffer_grid. => We have to copy back buffer_grid to h_result to get the real result.
        // With this, we can avoid calling enqueueReadBuffer every iteration.
        try {
            highlife_kernel(cl::EnqueueArgs(queue, global, local), buffer_grid, buffer_result, m_grid->getWidth(), m_grid->getHeight());
            highlife_kernel(cl::EnqueueArgs(queue, global, local), buffer_result, buffer_grid, m_grid->getWidth(), m_grid->getHeight());
          }
        catch (cl::Error e)
        {
            std::cerr << "Error: " << e.what() << ". Input size is not divisible by kernel range." << std::endl;
            break;
        }

        iterations += 2;
    }

    // Copy the output data back to the host (Check note above to see why our results are in buffer_grid instead of buffer_result.)
    queue.enqueueReadBuffer( buffer_grid, CL_TRUE, 0, m_grid->getWidth() * m_grid->getHeight() * sizeof(bool), host_result );

    // Set the result
    for (int j = 0; j < m_grid->getHeight(); j++)
    {
        for (int i = 0; i < m_grid->getWidth(); i++)
        {
            m_grid->setAt(i, j, host_result[getPosCL(i, j, m_grid->getWidth())]);
        }
    }

    return iterations;
}
