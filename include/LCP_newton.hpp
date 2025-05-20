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

// Returns merit of point z given point w=Mz+q, merit is the norm of the elementwise-min of the two
float get_merit(int N, thrust::device_vector<float> &z, thrust::device_vector<float> &w,cublasHandle_t &handle);

// Gets all q_i where i in alpha
void subvector(int N, thrust::device_vector<float> &q, thrust::device_vector<int> &alpha, thrust::device_vector<float> &q_alpha);

// Gets all M_i,k where i and k are in alpha
void submatrix(int N, thrust::device_vector<float> &M, thrust::device_vector<int> &alpha, thrust::device_vector<float> &M_alpha);

// Finds alpha set which is elements where z1 > z2
void alpha_set(int N, thrust::device_vector<float> &z1, thrust::device_vector<float> &z2, thrust::device_vector<int> &alpha, thrust::device_vector<int> &gamma);

// Takes elementwise minimum of two vectors z1 and z2
void elementwise_min(int N, thrust::device_vector<float> &z1, thrust::device_vector<float> &z2, thrust::device_vector<float> &res);

// Evaluates the linear function Mz+q
void eval_linear(int N, thrust::device_vector<float> &M, thrust::device_vector<float> &q, thrust::device_vector<float> &z, thrust::device_vector<float> &res, cublasHandle_t &handle);

// Tests for whether the merit of a point z < epsilon
bool norm_termination_test(int N, thrust::device_vector<float> &z, thrust::device_vector<float> &w, float epsilon, cublasHandle_t &handle);

// Solves linear system Ax = b, returns B
void solve_linear_system(int N, thrust::device_vector<float> &A, thrust::device_vector<float> &b, thrust::device_vector<float> &res, cusolverDnHandle_t &handle, cusolverDnParams_t &params, void *host_buffer, size_t host_buffer_size, void *device_buffer, size_t device_buffer_size);

// Sets up elements for linear system solver
void setup_solver(int N, cusolverDnHandle_t &handle, void *&host_buffer, size_t &host_buffer_size, void *&device_buffer, size_t &device_buffer_size, cusolverDnParams_t &params);

// Scatters vector u_alpha according to alpha i.e creates vector q such that q_i = q_j where i is in alpha and j is the corresponding element of alpha
void scatter_vector(int N, thrust::device_vector<float> &u_alpha, thrust::device_vector<int> &alpha, thrust::device_vector<float> &res);

// Tests whether u is a valid solution to the LCP phi = Mu + q
bool solve_termination_test(int N, thrust::device_vector<float> &u, thrust::device_vector<float> &phi, float epsilon, cublasHandle_t &handle);

// Gets rhos according to equation found in Bai and Dong equations 3.11 and 3.12
void get_rhos(int N, thrust::device_vector<float> &z, thrust::device_vector<float> &u, thrust::device_vector<float> &phi, thrust::device_vector<float> &w, thrust::device_vector<int> &gamma, thrust::device_vector<int> &alpha, thrust::device_vector<float> &rhos);

// Gets the next iteration where ||H(z)|| < (1-eta*tau)||H((1-tau)z + tau* u)
void get_next_iter(int N, thrust::device_vector<float> &z, thrust::device_vector<float> &M, thrust::device_vector<float> &q, thrust::device_vector<float> &u, thrust::device_vector<float> &rhos, cublasHandle_t &handle, float current_merit, float xi, float sigma1, float sigma2, thrust::device_vector<float> &res);

// Solves LCP(M, q) using newtons method given a starting point z_0 which is nondegenerate (i.e z_0_i =/= (Mz_0 + q)_i for any i)
int LCP_Newton(int N, thrust::device_vector<float> &M, thrust::device_vector<float> &q, thrust::device_vector<float> &z0, float epsilon, float xi, float sigma0, float sigma1, thrust::device_vector<float> &res);
