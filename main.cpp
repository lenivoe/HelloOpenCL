#define __MAIN_CPP
#ifdef __MAIN_CPP

#include "OpenCLWrapper.h"

#include "Matx.h"


int main() {
    // data
    const size_t common_size = 4000;
    const size_t rs = common_size;
    const size_t csrs = common_size;
    const size_t cs = common_size;
    CMatx<true> in1_matx(rs, csrs);
    CMatx<false> in2_matx(in1_matx.Cols(), cs);
    CMatx<true> out_matx(in1_matx.Rows(), in2_matx.Cols());

    auto MatxInput = [](auto& matx) {
        for(size_t row = 0; row < matx.Rows(); row++)
            for(size_t col = 0; col < matx.Cols(); col++)
                matx.At(row, col) = static_cast<float>(row) / matx.Rows() +
                                    static_cast<float>(col) / matx.Cols();
    };

    MatxInput(in1_matx);
    MatxInput(in2_matx);


    try {
        COpenCLWrapper::Init(0, CL_DEVICE_TYPE_GPU, 1);
        cl::Kernel kernel = COpenCLWrapper::BuildKernel("..\\HelloOpenCL\\kernel.cl");
        COpenCLWrapper::MatMul(kernel,
                               static_cast<float*>(&in1_matx.At(0, 0)),
                               static_cast<float*>(&in2_matx.At(0, 0)),
                               static_cast<float*>(&out_matx.At(0, 0)),
                               in1_matx.Rows(), in1_matx.Cols(), in2_matx.Cols());
    } catch(cl::Error error) {
        std::cout << error.what() << " : " << CLErrName(error.err()) << std::endl << std::endl;
        return error.err();
    }

    auto MatxOutput = [](auto& matx) {
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

    MatxOutput(in1_matx);
    MatxOutput(in2_matx);
    MatxOutput(out_matx);

    std::cout << std::endl << std::endl << "done." << std::endl;
}


#endif // __MAIN_CPP

