#include <cublas_v2.h>
#include <stdio.h>
#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/partition.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/logical.h>
#include <cusolverDn.h>

struct is_less_than
{
    thrust::device_ptr<const float> a;
    thrust::device_ptr<const float> b;

    is_less_than(thrust::device_ptr<const float> a_, thrust::device_ptr<const float> b_)
        : a(a_), b(b_) {}

    __host__ __device__ bool operator()(int i) const
    {
        return a[i] > b[i];
    }
};

float get_merit(int N, thrust::device_vector<float> &z, thrust::device_vector<float> &w,cublasHandle_t &handle);

void subvector(int N, thrust::device_vector<float> &q, thrust::device_vector<int> &alpha, thrust::device_vector<float> &q_alpha);

void submatrix(int N, thrust::device_vector<float> &M, thrust::device_vector<int> &alpha, thrust::device_vector<float> &M_alpha);

void alpha_set(int N, thrust::device_vector<float> &z1, thrust::device_vector<float> &z2, thrust::device_vector<int> &alpha, thrust::device_vector<int> &gamma);

void elementwise_min(int N, thrust::device_vector<float> &z1, thrust::device_vector<float> &z2, thrust::device_vector<float> &res);

void eval_linear(int N, thrust::device_vector<float> &M, thrust::device_vector<float> &q, thrust::device_vector<float> &z, thrust::device_vector<float> &res, cublasHandle_t &handle);

bool norm_termination_test(int N, thrust::device_vector<float> &z, thrust::device_vector<float> &w, float epsilon, cublasHandle_t &handle);

void solve_linear_system(int N, thrust::device_vector<float> &A, thrust::device_vector<float> &b, thrust::device_vector<float> &res, cusolverDnHandle_t &handle, cusolverDnParams_t &params, void *host_buffer, size_t host_buffer_size, void *device_buffer, size_t device_buffer_size);

void setup_solver(int N, cusolverDnHandle_t &handle, void *&host_buffer, size_t &host_buffer_size, void *&device_buffer, size_t &device_buffer_size, cusolverDnParams_t &params);

void scatter_vector(int N, thrust::device_vector<float> &u_alpha, thrust::device_vector<int> &alpha, thrust::device_vector<float> &res);

bool solve_termination_test(int N, thrust::device_vector<float> &u, thrust::device_vector<float> &phi, float epsilon, cublasHandle_t &handle);

void get_rhos(int N, thrust::device_vector<float> &z, thrust::device_vector<float> &u, thrust::device_vector<float> &phi, thrust::device_vector<float> &w, thrust::device_vector<int> &gamma, thrust::device_vector<int> &alpha, thrust::device_vector<float> &rhos);

void get_next_iter(int N, thrust::device_vector<float> &z, thrust::device_vector<float> &M, thrust::device_vector<float> &q, thrust::device_vector<float> &u, thrust::device_vector<float> &rhos, cublasHandle_t &handle, float current_merit, float xi, float sigma1, float sigma2, thrust::device_vector<float> &res);

int LCP_Newton(int N, thrust::device_vector<float> &M, thrust::device_vector<float> &q, thrust::device_vector<float> &z0, float epsilon, float xi, float sigma0, float sigma1, thrust::device_vector<float> &res);
