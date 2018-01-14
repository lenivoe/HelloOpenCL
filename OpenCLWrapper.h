#ifndef OPENCLWRAPPER_H
#define OPENCLWRAPPER_H

#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 100
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp> // вызывает ошибки линковки, использовать только в inline функциях

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>


class COpenCLWrapper {
public:
    ~COpenCLWrapper();

    static inline void Init(size_t _platform_ind = 0, size_t _device_type = CL_DEVICE_TYPE_GPU, size_t _device_ind = 0);
    static inline cl::Kernel BuildKernel(const cl::string& _file_name, const cl::string& _kernel_name = "main");
    static inline void MatMul(cl::Kernel& _kernel,
                     const float* _input_matx1,
                     const float* _input_matx2,
                     float* _output_matx,
                     std::size_t _rows1,
                     std::size_t _cols1_rows2,
                     std::size_t _cols2);
private:
    inline COpenCLWrapper(const cl::Device& _device) :
        m_ContextDevices({ _device }), m_Context(m_ContextDevices), m_Queue(m_Context, _device) { }

    static COpenCLWrapper* m_Inst;

    cl::vector<cl::Device> m_ContextDevices;
    cl::Context m_Context;
    cl::CommandQueue m_Queue;
};

//
// public
//

inline void COpenCLWrapper::Init(size_t _platform_ind, size_t _device_type, size_t _device_ind) {
    // get platforms
    std::cout << "..get platforms" << std::endl;
    cl::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // get devices
    std::cout << "..get devices" << std::endl;
    cl::vector<cl::Device> devices;
    platforms[_platform_ind].getDevices(_device_type, &devices);

    // create context and queue for selected device
    std::cout << "..create context and queue" << std::endl;
    m_Inst = new COpenCLWrapper(devices[_device_ind]);
}

inline cl::Kernel COpenCLWrapper::BuildKernel(const cl::string& _file_name, const cl::string& _kernel_name) {
    // load OpenCL source code
    std::cout << "..read sources" << std::endl;
    std::ifstream sourceFile(_file_name);
    std::string sourceStr(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

    // build OpenCL program
    std::cout << "..set sources" << std::endl;
    cl::Program::Sources source = { sourceStr };
    std::cout << "..create program" << std::endl;
    cl::Program program = cl::Program(m_Inst->m_Context, source);
    std::cout << "..build program" << std::endl;
    program.build(m_Inst->m_ContextDevices);
    // make the kernel
    std::cout << "..create kernel" << std::endl;
    return cl::Kernel(program, _kernel_name.c_str());
}

inline void COpenCLWrapper::MatMul(cl::Kernel& _kernel,
                            const float* _input_matx1, const float* _input_matx2, float* _output_matx,
                            std::size_t _rows1, std::size_t _cols1_rows2, std::size_t _cols2)
{
    std::cout << "..create buffers\n";

    // create memory buffers
    cl::Context& context = m_Inst->m_Context;
    cl::Buffer clmInputBuf1 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         _rows1 * _cols1_rows2 * sizeof(float), const_cast<float*>(_input_matx1));
    cl::Buffer clmInputBuf2 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         _cols1_rows2 * _cols2 * sizeof(float), const_cast<float*>(_input_matx2));
    cl::Buffer clmOutputBuf = cl::Buffer(context, CL_MEM_WRITE_ONLY, _rows1 * _cols2 * sizeof(float));

    std::cout << "..set args\n";

    // set arguments to kernel
    size_t arg_ind = 0;
    _kernel.setArg(arg_ind++, clmInputBuf1);
    _kernel.setArg(arg_ind++, clmInputBuf2);
    _kernel.setArg(arg_ind++, clmOutputBuf);
    _kernel.setArg(arg_ind++, _rows1);
    _kernel.setArg(arg_ind++, _cols1_rows2);
    _kernel.setArg(arg_ind++, _cols2);

    std::cout << "..add kernel to queue\n";

    cl::CommandQueue& queue = m_Inst->m_Queue;

    // calc output
    queue.enqueueNDRangeKernel(_kernel, cl::NullRange, cl::NDRange(_rows1 * _cols2));

    std::cout << "..launch kernel\n";

    queue.finish();

    std::cout << "..read output\n";

    // read output
    queue.enqueueReadBuffer(clmOutputBuf, CL_TRUE, 0, _rows1 * _cols2 * sizeof(float), _output_matx);
}

//
// other functions
//

const char* CLErrName(int err);

#endif // OPENCLWRAPPER_H
