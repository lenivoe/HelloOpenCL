#ifndef OPENCLQUEUE_H
#define OPENCLQUEUE_H

#include <iostream>

#include "OpenCLUtility.h"

namespace OpenCL {

class COpenCLQueue
{
public:
    inline COpenCLQueue(size_t _platform_ind, size_t _device_type, size_t _device_ind);

    inline const cl::Device& Device() const { return m_Device; }
    inline const cl::Context& Context() const { return m_Context; }
    inline const cl::CommandQueue& Queue() const { return m_Queue; }
private:
    cl::Device m_Device;
    cl::Context m_Context;
    cl::CommandQueue m_Queue;
};

inline COpenCLQueue::COpenCLQueue(size_t _platform_ind, size_t _device_type, size_t _device_ind) {
    using namespace OpenCL;
    using namespace Utility;

    // get platforms
    std::cout << "..get platforms" << std::endl;
    cl::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // get devices
    std::cout << "..get devices" << std::endl;
    cl::vector<cl::Device> devices;
    platforms[_platform_ind].getDevices(_device_type, &devices);

    // device info
    cl::Device& cur_device = devices[_device_ind];
    std::cout << "....device name: " << cur_device.getInfo<CL_DEVICE_NAME>() << std::endl <<
                 "....device type: " << CLDeviceType(cur_device.getInfo<CL_DEVICE_TYPE>()) << std::endl <<
                 "....device max work group size: " << cur_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;

    // create context and queue for selected device
    std::cout << "..create context and queue" << std::endl;
    m_Device = devices[_device_ind];
    m_Context = cl::Context({ m_Device });
    m_Queue = cl::CommandQueue(m_Context, m_Device);
}

}

#endif // OPENCLQUEUE_H
