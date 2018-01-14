#define __MAIN_CPP
#ifdef __MAIN_CPP

#define NDEBUG
#include <assert.h>
#include <sstream>
#include <fstream>
#include <chrono>

#include "Matx.h"
#include "OpenCLWrapper.h"


int StrToInt(std::string str) {
    int ret_val;
    std::istringstream(str) >> ret_val;
    return ret_val;
}


int main(int argc, char* argv[]) {
    std::cout << "data init" << std::endl;
    // data
    const size_t common_size = argc > 4 ? StrToInt(argv[4]) : 4000;
    const size_t r1_size = common_size;
    const size_t c1_r2_size = common_size;
    const size_t c2_size = common_size;
    CMatx<true> in1_matx(r1_size, c1_r2_size);
    CMatx<true> in2_matx(in1_matx.Cols(), c2_size);
    CMatx<true> out_matx(in1_matx.Rows(), in2_matx.Cols());

    auto InputMatx = [](auto& matx) {
        for(size_t row = 0; row < matx.Rows(); row++)
            for(size_t col = 0; col < matx.Cols(); col++)
                matx.At(row, col) = static_cast<int>(row - col);
    };
    auto InputMatx2 = [&InputMatx](auto& matx) {
        InputMatx(matx);
        for(size_t row = 0; row < matx.Rows(); row++)
            for(size_t col = 0; col < matx.Cols(); col++)
                matx.At(row, col) /= 2;
    };

    InputMatx(in1_matx);
    InputMatx2(in2_matx);


    try {
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
        const char* kernal_filename = "..\\HelloOpenCL\\kernel.cl";
        COpenCLWrapper::Init(platform_ind, device_type, device_ind);
        std::cout << "build kernel" << std::endl;
        cl::Kernel kernel = COpenCLWrapper::BuildKernel(kernal_filename);


        std::cout << "calculating" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        COpenCLWrapper::MatMul(kernel,
                               static_cast<float*>(&in1_matx.At(0, 0)),
                               static_cast<float*>(&in2_matx.At(0, 0)),
                               static_cast<float*>(&out_matx.At(0, 0)),
                               in1_matx.Rows(), in1_matx.Cols(), in2_matx.Cols());

        std::chrono::duration<double> seconds(std::chrono::high_resolution_clock::now() - start_time);
        std::cout << "time: " << seconds.count() << " sec" << std::endl << std::endl;
    } catch(cl::Error error) {
        std::cout << error.what() << " : " << CLErrName(error.err()) << std::endl << std::endl;
        return error.err();
    }

    auto OutputMatx = [](auto& matx) {
        size_t rows = matx.Rows();
        const int max_count = 5;
        if(rows > max_count) rows = max_count;
        size_t cols = matx.Cols();
        if(cols > max_count) cols = max_count;

        for(size_t row = 0; row < rows; row++) {
            for(size_t col = 0; col < cols; col++)
                std::cout << matx.At(row, col) << ' ';
            std::cout << std::endl;
        }
        std::cout << std::endl;
    };

    OutputMatx(in1_matx);
    OutputMatx(in2_matx);
    OutputMatx(out_matx);

    std::cout << std::endl << "done" << std::endl;
}


#endif // __MAIN_CPP

