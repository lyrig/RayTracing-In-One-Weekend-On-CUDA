#ifndef INTERVAL_H
#define INTERVAL_H

#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <thrust/extrema.h>
#include <float.h>
class interval {
public:
    float min, max;

    __host__ __device__ interval() : min(+FLT_MIN), max(FLT_MAX) {} // Default interval is empty

    __host__ __device__ interval(float min, float max) : min(min), max(max) {}

    __host__ __device__ float size() const {
        return max - min;
    }

    __host__ __device__ bool contains(float x) const {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(float x) const {
        return min < x&& x < max;
    }

    __host__ __device__ float clamp(float x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    static const interval empty, universe;
};

const interval interval::empty = interval(+FLT_MAX, -FLT_MAX);
const interval interval::universe = interval(-FLT_MAX, +FLT_MAX);

#endif
