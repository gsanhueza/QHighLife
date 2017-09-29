#ifndef _OPENCLMODEL_H_
#define _OPENCLMODEL_H_

#include "model.h"

class OpenCLModel : public Model
{
public:
    OpenCLModel(int width, int height);
    ~OpenCLModel();

    virtual void run();
    virtual int runStressTest(int timeInSeconds);

protected:
    virtual void setup();

private:
    int m_platform_id;
    int m_device_id;

    bool *host_grid;
    bool *host_result;
};

#endif
