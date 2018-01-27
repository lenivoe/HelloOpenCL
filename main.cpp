#define __MAIN_CPP
#ifdef __MAIN_CPP

#define NDEBUG
#include <assert.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

#include "Matx.h"
#include "OpenCLQueue.h"
#include "OpenCLTask.h"

int StrToInt(std::string str) {
    int ret_val;
    std::istringstream(str) >> ret_val;
    return ret_val;
}


int main(int argc, char* argv[]) {
    // data
    const size_t common_size = argc > 4 ? StrToInt(argv[4]) : 16 * (128 * 3 / 2);
    const size_t r1_size = common_size * 2;
    const size_t c1_r2_size = common_size;
    const size_t c2_size = common_size * 3;

    std::cout << "data init. in1: " << r1_size << " x " << c1_r2_size <<
                          ", in2: " << c1_r2_size << " x " << c2_size << std::endl;
    CMatx<false> in1_matx(r1_size, c1_r2_size);
    CMatx<false> in2_matx(in1_matx.Cols(), c2_size);
    CMatx<false> out_matx(in1_matx.Rows(), in2_matx.Cols());

    auto InputMatx = [](auto& matx) {
        //int i = 0;
        for(size_t row = 0; row < matx.Rows(); row++)
            for(size_t col = 0; col < matx.Cols(); col++)
                matx.At(row, col) = abs(row - col) % 7 - 3;
    };
    auto InputMatx2 = [](auto& matx) {
        //int i = -5;
        for(size_t row = 0; row < matx.Rows(); row++)
            for(size_t col = 0; col < matx.Cols(); col++)
                matx.At(row, col) = abs(row - col) % 3;
    };

    InputMatx(in1_matx);
    InputMatx2(in2_matx);

    try {
        using OpenCL::COpenCLQueue;
        using OpenCL::COpenCLTask;
        using OpenCL::Utility::CLLoadSource;

        const size_t platform_ind = argc > 1 ? StrToInt(argv[1]) : 0;
        std::string device_type_str = argc > 2 ? argv[2] : "gpu";
        int device_type = -1;
        if(device_type_str == "gpu")
            device_type = CL_DEVICE_TYPE_GPU;
        else if(device_type_str == "cpu")
            device_type = CL_DEVICE_TYPE_CPU;
        else if(device_type_str == "all")
            device_type = CL_DEVICE_TYPE_ALL;
        const size_t device_ind = argc > 3 ? StrToInt(argv[3]) : 0;

        std::cout << "opencl init" << std::endl;
        COpenCLQueue queue(platform_ind, device_type, device_ind);

        std::cout << "build kernel" << std::endl;
        const char* kernal_filename = "mat_mul_kernel.cl";
        COpenCLTask task;
        const size_t tile_size = COpenCLTask::CalcTileSize(queue, out_matx.Rows(), out_matx.Cols());
        task.Build(queue, CLLoadSource(kernal_filename), "main", tile_size);

        std::cout << "calculating, local group size: " << tile_size << " x " << tile_size << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        task.MatMul(queue,
                    &in1_matx.At(0, 0), &in2_matx.At(0, 0), &out_matx.At(0, 0),
                    in1_matx.Rows(), in1_matx.Cols(), in2_matx.Cols());

        std::chrono::duration<double> seconds(std::chrono::high_resolution_clock::now() - start_time);
        std::cout << "time: " << seconds.count() << " sec" << std::endl << std::endl;
    } catch(cl::Error error) {
        using OpenCL::Utility::CLErrName;
        std::cout << error.what() << " : " << CLErrName(error.err()) << std::endl << std::endl;
        return error.err();
    }

    auto OutputMatx = [](auto& matx) {
        std::ostream& out = std::cout;
        const int weight = 3;
        size_t rows = matx.Rows();
        const int max_count = 10;
        if(rows > max_count) rows = max_count;
        size_t cols = matx.Cols();
        if(cols > max_count) cols = max_count;

        for(size_t row = 0; row < rows; row++) {
            for(size_t col = 0; col < cols; col++)
                out << std::setw(weight) << matx.At(row, col) << ' ';
            out << std::endl;
        }
        out << std::endl;
    };

    auto OutputOf2Matx = [common_size](auto& matx1, auto& matx2) {
        std::ostream& out = std::cout;
        const size_t max_size = 10;
        size_t size = common_size;
        if(size > max_size) size = max_size;
        const int weight = 2;
        for(size_t row = 0; row < size; row++) {
            for(size_t col = 0; col < size; col++)
                out << std::setw(weight) << matx1.At(row, col) << ' ';
            out << '\t';
            for(size_t col = 0; col < size; col++)
                out << std::setw(weight) << matx2.At(row, col) << ' ';
            out << std::endl;
        }
        out << std::endl;
    };

    if(in1_matx.Rows() > 10 || in1_matx.Cols() > 10 ||
            in2_matx.Rows() > 10 || in2_matx.Cols() > 10)
    {
        std::cout << "in1, in2:" << std::endl;
        OutputOf2Matx(in1_matx, in2_matx);
    } else {
        std::cout << "in1:" << std::endl;
        OutputMatx(in1_matx);
        std::cout << "in2:" << std::endl;
        OutputMatx(in2_matx);
    }

    std::cout << "out:" << std::endl;
    OutputMatx(out_matx);
}


#endif // __MAIN_CPP

