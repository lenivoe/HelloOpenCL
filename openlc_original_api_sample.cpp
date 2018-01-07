//#define __OLD_SAMPLE
#ifdef __OLD_SAMPLE

#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl2.hpp>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>

#define SUCCESS 0
#define FAILURE 1


// convert the kernel file into a string
int convertToString(const char* filename, std::string& s) {
    std::ifstream file(filename, std::ios::binary);
    if(file.is_open()) {
        file.seekg(0, std::fstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::fstream::beg);
        s.resize(size);
        file.read(&s[0], size);
        file.close();
        return 0;
    }
    return FAILURE;
}

int main() {
    const size_t max_platforms_count = 4;
    const size_t max_devices_count = 8;

    // Step1: Getting platforms and choose an available one.
    cl_uint numPlatforms;	//the NO. of platforms
    cl_platform_id platforms[max_platforms_count];	//the chosen platform
    cl_int is_error = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (is_error)
        throw "Getting platforms failed";

    // For clarity, choose the first available platform.
    if(numPlatforms > 0) {
        is_error = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if (is_error)
            throw "Getting platforms IDs failed";
    }

    // Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.
    cl_uint numDevices = 0;
    cl_device_id devices[max_devices_count];
    is_error = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (numDevices == 0) {	//no GPU available.
        std::cout << "No GPU device available." << std::endl
                  << "Choose CPU as default device." << std::endl;
        is_error = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
        if (is_error) throw "clGetDeviceIDs for CPU error 1";
        is_error = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
        if (is_error) throw "clGetDeviceIDs for CPU error 2";
    }
    else {
        is_error = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
        if (is_error) throw "clGetDeviceIDs for GPU error";
    }

    // Step 3: Create context.
    cl_context context = clCreateContext(NULL,1, devices,NULL,NULL,NULL);

    // Step 4: Creating command queue associate with the context.
    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, devices[0], 0, NULL);

    // Step 5: Create program object
    const char* filename = "..\\HelloOpenCL\\HelloWorld_Kernel.cl";
    std::string sourceStr;
    is_error = convertToString(filename, sourceStr);
    if (is_error)
        throw "convertToString error";
    const char* source = sourceStr.c_str();
    size_t sourceSize[] = { sourceStr.length() };
    cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);

    // Step 6: Build program.
    is_error = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    if (is_error) throw "clBuildProgram error";

    // Step 7: Initial input,output for the host and create memory objects for the kernel
    const std::string input = "GdkknVnqkc";
    std::cout << "input string:" << std::endl
              << input << std::endl << std::endl;
    size_t str_size = input.length();

    std::string output(input.size(), '\0');

    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                        (str_size + 1) * sizeof(char), (void*)&input[0], NULL);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         (str_size + 1) * sizeof(char), NULL, NULL);

    // Step 8: Create kernel object
    cl_kernel kernel = clCreateKernel(program, "helloworld", NULL);

    // Step 9: Sets Kernel arguments.
    is_error = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputBuffer);
    if (is_error) throw "clSetKernelArg error 1";
    is_error = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outputBuffer);
    if (is_error) throw "clSetKernelArg error 2";

    // Step 10: Running the kernel.
    size_t global_work_size[] = { str_size };
    is_error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                      global_work_size, NULL, 0, NULL, NULL);
    if (is_error) throw "clEnqueueNDRangeKernel error";

    // Step 11: Read the cout put back to host memory.
    is_error = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0,
                                   str_size * sizeof(char), &output[0], 0, NULL, NULL);
    if (is_error) throw "clEnqueueReadBuffer error";
    std::cout << "output string:" << std::endl
              << output << std::endl << std::endl;

    // Step 12: Clean the resources.
    is_error += !!clReleaseKernel(kernel);				//Release kernel.
    is_error += !!clReleaseProgram(program);				//Release the program object.
    is_error += !!clReleaseMemObject(inputBuffer);		//Release mem object.
    is_error += !!clReleaseMemObject(outputBuffer);
    is_error += !!clReleaseCommandQueue(commandQueue);	//Release  Command queue.
    is_error += !!clReleaseContext(context);				//Release context.

    if(is_error)
        throw "Common error";

    std::cout << "Passed!\n";

    return SUCCESS;
}

#endif // __OLD_SAMPLE
