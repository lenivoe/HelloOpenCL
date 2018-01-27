#ifndef OPENCLTASK_H
#define OPENCLTASK_H

#include <fstream>
#include <iostream>
#include <cmath>

#include "OpenCLUtility.h"
#include "OpenCLQueue.h"

namespace OpenCL {

class COpenCLTask
{
public:
    COpenCLTask() { }
    virtual inline void Build(const COpenCLQueue& _queue, const cl::string& _source, const cl::string& _kernel_name, size_t tile_size);
    inline void MatMul(const COpenCLQueue& _queue,
                       const float* _input_matx1, const float* _input_matx2, float* _output_matx,
                       std::size_t _rows1, std::size_t _cols1_rows2, std::size_t _cols2);
    static inline size_t CalcTileSize(const COpenCLQueue& _queue, size_t _rows, size_t _cols);
private:
    cl::Kernel m_Kernel;
    cl::NDRange m_WorkgroupSize;
};

inline void COpenCLTask::Build(const COpenCLQueue& _queue, const cl::string& _source, const cl::string& _kernel_name, size_t tile_size) {
    m_WorkgroupSize = cl::NDRange(tile_size, tile_size);

    // build OpenCL program
    std::cout << "..set sources" << std::endl;
    cl::Program::Sources source = { _source };

    std::cout << "..create program" << std::endl;
    cl::Program program = cl::Program(_queue.Context(), source);
    std::stringstream build_options_builder;
    build_options_builder << "-D TS=" << tile_size;// << " -D M=" << 4000 << " -D K=" << 4000 << " -D N=" << 4000;
    std::string build_options = build_options_builder.str();

    std::cout << "..build program with options: " << build_options << std::endl;
    program.build({ _queue.Device() }, build_options.c_str());

    // make the kernel
    std::cout << "..create kernel" << std::endl;
    m_Kernel = cl::Kernel(program, _kernel_name.c_str());

    // kernel info
    size_t kernel_max_work_group_size = 0;
    m_Kernel.getWorkGroupInfo(_queue.Device(), CL_KERNEL_WORK_GROUP_SIZE, &kernel_max_work_group_size);

    std::cout << "....max work group size: " << kernel_max_work_group_size << std::endl;
    size_t pref_work_group_size = 0;
    m_Kernel.getWorkGroupInfo(_queue.Device(), CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &pref_work_group_size);
    std::cout << "....preferred multiple of workgroup size: " << pref_work_group_size << std::endl;
}

inline void COpenCLTask::MatMul(const COpenCLQueue& _queue,
                                const float* _input_matx1, const float* _input_matx2, float* _output_matx,
                                std::size_t _rows1, std::size_t _cols1_rows2, std::size_t _cols2)
{
    std::cout << "..create buffers\n";

    // create memory buffers
    const cl::Context& context = _queue.Context();
    cl::Buffer clmInputBuf1 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         _rows1 * _cols1_rows2 * sizeof(float), const_cast<float*>(_input_matx1));
    cl::Buffer clmInputBuf2 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         _cols1_rows2 * _cols2 * sizeof(float), const_cast<float*>(_input_matx2));
    cl::Buffer clmOutputBuf = cl::Buffer(context, CL_MEM_WRITE_ONLY, _rows1 * _cols2 * sizeof(float));

    // set arguments to kernel
    std::cout << "..set args\n";
    size_t arg_ind = 0;
    m_Kernel.setArg(arg_ind++, clmInputBuf1);
    m_Kernel.setArg(arg_ind++, clmInputBuf2);
    m_Kernel.setArg(arg_ind++, clmOutputBuf);
    m_Kernel.setArg(arg_ind++, _rows1);
    m_Kernel.setArg(arg_ind++, _cols1_rows2);
    m_Kernel.setArg(arg_ind++, _cols2);

    std::cout << "..add kernel to queue\n";
    const cl::CommandQueue& queue = _queue.Queue();

    // calc output
    queue.enqueueNDRangeKernel(m_Kernel, cl::NullRange, cl::NDRange(_rows1, _cols2), m_WorkgroupSize);
    std::cout << "..launch kernel\n";
    queue.finish();

    // read output
    std::cout << "..read output\n";
    queue.enqueueReadBuffer(clmOutputBuf, CL_TRUE, 0, _rows1 * _cols2 * sizeof(float), _output_matx);
}

inline size_t COpenCLTask::CalcTileSize(const COpenCLQueue& _queue, size_t _rows, size_t _cols) {
    const size_t max_workgroup_size = _queue.Device().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const size_t max_tile_size = static_cast<size_t>(sqrt(max_workgroup_size));

    const auto calc_gcd = [](unsigned a, unsigned b) {
        while (b != 0) {
            unsigned t = b;
            b = a % b;
            a = t;
        }
        return a;
    };

    const size_t gcd = calc_gcd(_rows, _cols);
    size_t tile_size = max_tile_size < gcd ? max_tile_size : gcd;
    while(gcd % tile_size != 0)
        tile_size--;

    return tile_size;
}


}

#endif // OPENCLTASK_H
