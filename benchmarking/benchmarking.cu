#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/transform.h>

#include "LCP_newton.hpp"

struct add_diagonal_functor
{
    const float *diag;
    int n;

    add_diagonal_functor(const float *d, int n_) : diag(d), n(n_) {}

    __host__ __device__ float operator()(const float &a, const int &idx) const
    {
        int row = idx % n;
        int col = idx / n;
        if (row == col)
        {
            return a + diag[row];
        }
        else
        {
            return a;
        }
    }
};

void add_diagonal_to_matrix(thrust::device_vector<float> &A, const thrust::device_vector<float> &diag, int n)
{
    thrust::counting_iterator<int> idx_first(0);
    thrust::counting_iterator<int> idx_last = idx_first + n * n;

    const float *d_ptr = thrust::raw_pointer_cast(diag.data());
    thrust::transform(
        A.begin(), A.end(),
        idx_first,
        A.begin(),
        add_diagonal_functor(d_ptr, n));
}

struct RandomFunctor
{
    float a, b;
    unsigned int seed;

    __host__ __device__
    RandomFunctor(float a, float b, unsigned int seed) : a(a), b(b), seed(seed) {}

    __device__
    float operator()(const int& i) const
    {
        curandState state;
        curand_init(seed, i, 0, &state);

        float r = curand_uniform(&state);
        return a + (b - a) * r;
    }
};

void generate_random_vector(int n, float a, float b, unsigned int &seed, thrust::device_vector<float> &res) {
    res.resize(n);
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(n),
        res.begin(),
        RandomFunctor(a, b, seed)
    );
    seed++;
}

static void BM_cuLCP_SOLVER(benchmark::State &state) {
    int n = state.range(0);

    float epsilon = 0.00001;
    float sigma0 = 0.1;
    float sigma1 = 0.75;
    float xi = 0.25;

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);

    unsigned int seed = 0;
    for (auto _ : state) {
        float one = 1.0;
        state.PauseTiming();
        thrust::device_vector<float> M_unsymm;
        generate_random_vector(n*n, -5, 5, seed, M_unsymm);

        thrust::device_vector<float> q;
        generate_random_vector(n, -500, 500, seed, q);

        thrust::device_vector<float> etas;
        generate_random_vector(n, 0, 0.3, seed, etas);

        thrust::device_vector<float> M(n * n);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &one, thrust::raw_pointer_cast(M_unsymm.data()), n, thrust::raw_pointer_cast(M_unsymm.data()), n, &one, thrust::raw_pointer_cast(M.data()), n);

        add_diagonal_to_matrix(M, etas, n);

        thrust::device_vector<float> z(n);
        thrust::device_vector<float> res;
        state.ResumeTiming();

        int status = LCP_Newton(n, M, q, z, epsilon, xi, sigma0, sigma1, res);
        cudaDeviceSynchronize();
        if (status != 0) {
            printf("Uh oh, solve has failed!\n");
        }
    }
    return;
}

BENCHMARK(BM_cuLCP_SOLVER)->Arg(8)->Unit(benchmark::kSecond)->Iterations(1);

BENCHMARK_MAIN();