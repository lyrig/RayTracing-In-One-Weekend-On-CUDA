#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "interval.h"
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <thrust/extrema.h>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

using color = vec3;
using point3 = vec3;

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__device__ color get_color(const ray& r, hitable** world, curandState* local_rand_state) {
    ray cur_ray = r;
    color cur_attenuation = color(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                if (rec.hit_type == 0)
                {
                    return cur_attenuation * attenuation;
                }
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    vec3 unit_direction = unit_vector(cur_ray.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    return cur_attenuation * c;
}

__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__device__ inline float linear_to_gamma(float linear_component)
{
    if (linear_component > 0)
        return sqrt(linear_component);

    return 0;
}


__global__ void render(vec3* fb, int max_x, int max_y, int sample_per_pixel, camera** cam, hitable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    color col(0, 0, 0);
    for (int s = 0; s < sample_per_pixel; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += get_color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(sample_per_pixel);
    col[0] = linear_to_gamma(col[0]);
    col[1] = linear_to_gamma(col[1]);
    col[2] = linear_to_gamma(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable** objects, hitable** d_world, camera** d_camera, int image_width, int image_height, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        objects[0] = new sphere(point3(0, -1000.0, -1), 1000,
            new lambertian(color(0.5, 0.5, 0.5)));
        int i = 0;
        
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                point3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    objects[++i] = new sphere(center, 0.2,
                        new lambertian(color(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95f) {
                    objects[++i] = new sphere(center, 0.2,
                        new metal(color(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else {
                    objects[++i] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        objects[++i] = new sphere(point3(0, 1, 0), 1.0, new dielectric(1.5));
        objects[++i] = new sphere(point3(-4, 1, 0), 1.0, new lambertian(color(0.4, 0.2, 0.1)));
        // objects[++i] = new sphere(point3(4, 1, 0), 1.0, new metal(color(0.7, 0.6, 0.5), 0.0));
        objects[++i] = new sphere(point3(4, 1, 0), 1.0, new LightSource(color(0.7, 0.6, 0.5), 2.0));
        *rand_state = local_rand_state;
        int num = 22 * 22 + 1 + 3;
        // num = 4;
        *d_world = new hitable_list(objects, num);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0;
        // or dist_to_focus = (lookfrom - lookat).length();
        float aperture = 0.1;
        float vfov = 30.0f;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            vfov,
            float(image_width) / float(image_height),
            aperture,
            dist_to_focus);
    }
}


int main() {
    int image_width = 1200; // 
    int image_height = 675;
    int sample_per_pixel = 50; // Sample number of each pixel.
    int tx = 8; // threads number.
    int ty = 8; // threads number.

    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3* frame_buffer;
    checkCudaErrors(cudaMalloc((void**)&frame_buffer, fb_size));

    // allocate random state
    curandState* d_rand_state1;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state1, num_pixels * sizeof(curandState)));
    curandState* d_rand_state2;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

    rand_init << <1, 1 >> > (d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    hitable** object_list;
    int num_hitables = 22 * 22 + 1 + 3;
    // num_hitables = 4;
    checkCudaErrors(cudaMalloc((void**)&object_list, num_hitables * sizeof(hitable*)));
    hitable** world;
    checkCudaErrors(cudaMalloc((void**)&world, sizeof(hitable*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world << <1, 1 >> > (object_list, world, d_camera, image_width, image_height, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (image_width, image_height, d_rand_state1);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render << <blocks, threads >> > (frame_buffer, image_width, image_height, sample_per_pixel, d_camera, world, d_rand_state1);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    vec3* fb2 = (vec3*)malloc(fb_size);
    checkCudaErrors(cudaMemcpy(fb2, frame_buffer, fb_size, cudaMemcpyDeviceToHost));
    // Output PPM as Image
    FILE* f = fopen("image.ppm", "w");
    std::fprintf(f, "P3\n%d %d\n%d\n", image_width, image_height, 255);
    static const interval intensity(0.000, 0.999);
    for (int j = image_height - 1; j >= 0; j--) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;
            auto pixel_color = fb2[pixel_index];
            auto r = pixel_color.x();
            auto g = pixel_color.y();
            auto b = pixel_color.z();

            int rbyte = int(256 * intensity.clamp(r));
            int gbyte = int(256 * intensity.clamp(g));
            int bbyte = int(256 * intensity.clamp(b));

            std::fprintf(f, "%d %d %d ", int(rbyte), int(gbyte), int(bbyte));
        }
    }
    std::cout << "Finished";
}