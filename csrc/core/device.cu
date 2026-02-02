#include "device.h"
#include <sstream>

namespace llamatelemetry {

void Device::check_cuda_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA Error at " << file << ":" << line << " - "
           << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

int Device::get_device_count() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

DeviceProperties Device::get_device_properties(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    DeviceProperties props;
    props.device_id = device_id;
    props.name = prop.name;
    props.total_memory = prop.totalGlobalMem;
    props.compute_capability_major = prop.major;
    props.compute_capability_minor = prop.minor;
    props.multiprocessor_count = prop.multiProcessorCount;
    props.max_threads_per_block = prop.maxThreadsPerBlock;
    props.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    props.warp_size = prop.warpSize;

    return props;
}

void Device::set_device(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

int Device::get_device() {
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    return device_id;
}

void Device::synchronize(int device_id) {
    if (device_id >= 0) {
        int current_device = get_device();
        set_device(device_id);
        CUDA_CHECK(cudaDeviceSynchronize());
        set_device(current_device);
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

size_t Device::get_free_memory(int device_id) {
    int current_device = get_device();
    if (device_id >= 0) {
        set_device(device_id);
    }

    size_t free_memory, total_memory;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));

    if (device_id >= 0) {
        set_device(current_device);
    }

    return free_memory;
}

size_t Device::get_total_memory(int device_id) {
    int current_device = get_device();
    if (device_id >= 0) {
        set_device(device_id);
    }

    size_t free_memory, total_memory;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));

    if (device_id >= 0) {
        set_device(current_device);
    }

    return total_memory;
}

} // namespace llamatelemetry
