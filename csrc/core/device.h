#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <stdexcept>

namespace llamatelemetry {

struct DeviceProperties {
    int device_id;
    std::string name;
    size_t total_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    int warp_size;
};

class Device {
public:
    static int get_device_count();
    static DeviceProperties get_device_properties(int device_id);
    static void set_device(int device_id);
    static int get_device();
    static void synchronize(int device_id = -1);
    static size_t get_free_memory(int device_id = -1);
    static size_t get_total_memory(int device_id = -1);
    static void check_cuda_error(cudaError_t error, const char* file, int line);
};

#define CUDA_CHECK(call) Device::check_cuda_error(call, __FILE__, __LINE__)

} // namespace llamatelemetry
