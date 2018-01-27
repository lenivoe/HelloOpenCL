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
#include <cmath>

//
// other functions
//
const char* CLErrName(int err);
const char* CLDeviceType(int type);



class COpenCLWrapper {
public:
    ~COpenCLWrapper();

    static inline void Init(size_t _platform_ind = 0, size_t _device_type = CL_DEVICE_TYPE_GPU, size_t _device_ind = 0);
    static inline cl::Kernel BuildKernel(const cl::string& _file_name, const cl::string& _kernel_name, size_t tile_size);
    static inline void MatMul(cl::Kernel& _kernel,
                     const float* _input_matx1,
                     const float* _input_matx2,
                     float* _output_matx,
                     std::size_t _rows1,
                     std::size_t _cols1_rows2,
                     std::size_t _cols2);
    static inline size_t CalcTileSize(size_t _rows, size_t _cols);
private:
    inline COpenCLWrapper(const cl::Device& _device) :
        m_ContextDevices({ _device }), m_Context(m_ContextDevices), m_Queue(m_Context, _device) { }

    static COpenCLWrapper* m_Inst;

    cl::vector<cl::Device> m_ContextDevices;
    cl::Context m_Context;
    cl::CommandQueue m_Queue;
    cl::NDRange m_WorkgroupSize = cl::NullRange;
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

    // device info
    cl::Device& cur_device = devices[_device_ind];
    std::cout << "....device name: " << cur_device.getInfo<CL_DEVICE_NAME>() << std::endl <<
                 "....device type: " << CLDeviceType(cur_device.getInfo<CL_DEVICE_TYPE>()) << std::endl <<
                 "....device max work group size: " << cur_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;

    // create context and queue for selected device
    std::cout << "..create context and queue" << std::endl;
    m_Inst = new COpenCLWrapper(devices[_device_ind]);
}

inline cl::Kernel COpenCLWrapper::BuildKernel(const cl::string& _file_name, const cl::string& _kernel_name, size_t tile_size) {
    m_Inst->m_WorkgroupSize = cl::NDRange(tile_size, tile_size);

    // load OpenCL source code
    std::cout << "..read sources" << std::endl;
    std::ifstream sourceFile(_file_name);
    std::string sourceStr(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

    // build OpenCL program
    std::cout << "..set sources" << std::endl;
    cl::Program::Sources source = { sourceStr };
    std::cout << "..create program" << std::endl;
    cl::Program program = cl::Program(m_Inst->m_Context, source);
    std::stringstream build_options_builder;
    build_options_builder << "-D TS=" << tile_size;// << " -D M=" << 4000 << " -D K=" << 4000 << " -D N=" << 4000;
    std::string build_options = build_options_builder.str();
    std::cout << "..build program with options: " << build_options << std::endl;
    program.build(m_Inst->m_ContextDevices, build_options.c_str());
    // make the kernel
    std::cout << "..create kernel" << std::endl;
    cl::Kernel kernel(program, _kernel_name.c_str());
    // kernel info
    size_t kernel_max_work_group_size = 0;
    kernel.getWorkGroupInfo(m_Inst->m_ContextDevices[0], CL_KERNEL_WORK_GROUP_SIZE, &kernel_max_work_group_size);
    std::cout << "....max work group size: " << kernel_max_work_group_size << std::endl;
    size_t pref_work_group_size = 0;
    kernel.getWorkGroupInfo(m_Inst->m_ContextDevices[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &pref_work_group_size);
    std::cout << "....preferred multiple of workgroup size: " << pref_work_group_size << std::endl;
    return std::move(kernel);
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

    // set arguments to kernel
    std::cout << "..set args\n";
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
    queue.enqueueNDRangeKernel(_kernel, cl::NullRange, cl::NDRange(_rows1, _cols2), m_Inst->m_WorkgroupSize);
    std::cout << "..launch kernel\n";
    queue.finish();

    // read output
    std::cout << "..read output\n";
    queue.enqueueReadBuffer(clmOutputBuf, CL_TRUE, 0, _rows1 * _cols2 * sizeof(float), _output_matx);
}

inline size_t COpenCLWrapper::CalcTileSize(size_t _rows, size_t _cols) {
    const size_t max_workgroup_size = m_Inst->m_ContextDevices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const size_t max_tile_size = static_cast<size_t>(sqrt(max_workgroup_size));

    auto calc_gcd = [](unsigned a, unsigned b) {
        while (b != 0) {
            unsigned t = b;
            b = a % b;
            a = t;
        }
        return a;
    };

    const size_t gcd = calc_gcd(_rows, _cols);
    // TODO
}



#endif // OPENCLWRAPPER_H
