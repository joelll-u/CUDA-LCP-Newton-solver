#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/transform.h>
#include <vector>

#include "LCP_newton.hpp"
#include "benchmarking_utils.cu"
#include "path_utils.cpp"

static void BM_cuLCP_SOLVER(benchmark::State &state) {
    int n = state.range(0);

    double epsilon = 0.00001;
    double sigma= 0.75;
    double xi = 0.25;

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);

    unsigned int seed = 0;
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<double> M_host;
        create_random_P_matrix(n, -5, 5, seed, M_host);

        std::vector<double> q_host;
        create_random_vector(n, -500, 500, seed, q_host);
        thrust::device_vector<double> z_0(n);
        thrust::device_vector<double> res;
        std::vector<double> res_host(n);
        state.ResumeTiming();

        thrust::device_vector<double> M = M_host;
        thrust::device_vector<double> q = q_host;

        int status = LCP_Newton(n, M, q, z_0, epsilon, xi, sigma, res);
        thrust::copy(res.begin(), res.end(), res_host.begin());
        cudaDeviceSynchronize();
        if (status != 0) {
            printf("Uh oh, solve has failed!\n");
        }
    }
    return;
}

static void BM_pathSolver(benchmark::State &state) {
    int n = state.range(0);
    unsigned int seed = 0;
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<double> M_host;
        create_random_P_matrix(n, -5, 5, seed, M_host);
        initializeM(M_host, n);

        std::vector<double> q_host;
        create_random_vector(n, -500, 500, seed, q_host);
        initializeQ(q_host, n);

        std::vector<double> z(n);
        std::vector<double> f(n);

        int status = 0;
        state.ResumeTiming();

        status = path_solve(n, z.data(), f.data());

        if (status != 1)
        {
            printf("Uh oh, solve has failed with status %d!\n", status);
        }
    }
}

static void BM_LemkeMethod(benchmark::State &state) {
    int n = state.range(0);
    unsigned int seed = 0;
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<double> M_host;
        create_random_P_matrix(n, -5, 5, seed, M_host);
        initializeM(M_host, n);

        std::vector<double> q_host;
        create_random_vector(n, -500, 500, seed, q_host);
        initializeQ(q_host, n);

        std::vector<double> z(n);
        std::vector<double> f(n);

        int status = 0;
        state.ResumeTiming();

        status = path_solve(n, z.data(), f.data(), true);


        if (status != 1)
        {
            printf("Uh oh, solve has failed with status %d!\n", status);
        }
    }
}

static void BM_cuLCP_Porous(benchmark::State &state) {
    int n = state.range(0);
    std::vector<double> q;
    create_porous_q(n, q);

    host_matrix_sparse f_host = create_sparse_porous_matrix(n);

    double epsilon = 0.00000001;
    double sigma = 0.75;
    double xi = 0.25;

    std::vector<double> res_host(n*n);

    for (auto _ : state)
    {

        thrust::device_vector<double> res;
        thrust::device_vector<double> q_d = q;
        thrust::device_vector<double> z_0(n * n, 0);
        matrix_sparse x = {f_host.row_offsets, f_host.column_indices, f_host.values};

        int status = LCP_Newton(n*n, x, q_d, z_0, epsilon, xi, sigma, res);

        thrust::copy(res.begin(), res.end(), res_host.begin());
        cudaDeviceSynchronize();
        if (status != 0) {
            printf("Uh oh, solve has failed with status %d\n", status);
        }
    }
}

static void BM_pathSolverPorous(benchmark::State &state)
{
    int n = state.range(0);
    std::vector<double> M_host;
    host_matrix_sparse x = create_sparse_porous_matrix(n);
    initializeM_from_sparse_symmetric(x, n*n);

    std::cout << n << " " << x.values.size() << std::endl;

    std::vector<double> q_host;
    create_porous_q(n, q_host);
    initializeQ(q_host, n*n);

    std::vector<double> z(n*n);
    std::vector<double> f(n*n);

    int status = 0;

    for (auto _ : state)
    {
        status = path_solve(n*n, z.data(), f.data());

        if (status != 1)
        {
            printf("Uh oh, solve has failed with status %d!\n", status);
        }
    }
}

static void BM_LemkePorous(benchmark::State &state)
{
    int n = state.range(0);
    std::vector<double> M_host;
    host_matrix_sparse x = create_sparse_porous_matrix(n);
    initializeM_from_sparse_symmetric(x, n * n);

    std::vector<double> q_host;
    create_porous_q(n, q_host);
    initializeQ(q_host, n * n);

    std::vector<double> z(n * n);
    std::vector<double> f(n * n);

    int status = 0;

    for (auto _ : state)
    {
        status = path_solve(n * n, z.data(), f.data(), true);

        if (status != 1)
        {
            printf("Uh oh, solve has failed with status %d!\n", status);
        }
    }
}

static void BM_pathSolverBimatrix(benchmark::State &state)
{
    std::vector<double> M_host = {
        0, 0, 0, 4, 3,
        0, 0, 0, 3, 7,
        0, 0, 0, 4, 2,
        4, 3, 1, 0, 0,
        4, 6, 7, 0, 0
    };

    initializeM(M_host, 5);

    std::vector<double> q_host = {-1, -1, -1, -1, -1};
    initializeQ(q_host, 5);

    std::vector<double> z(5, 0.5);
    std::vector<double> f(5);

    int status = 0;

    for (auto _ : state)
    {
        status = path_solve(5, z.data(), f.data(), true);
        for (int i = 0; i < 5; i++) { printf("%f ", z[i]); } printf("\n");
        if (status != 1)
        {
            printf("Uh oh, solve has failed with status %d!\n", status);
        }
    }
}

static void BM_cuLCP_random_sparsity(benchmark::State &state) {
    int n = 10000;
    int divisor = state.range(1);
    int subtractor = 10 - state.range(0);
    double sparsity = ((double) (divisor - subtractor))/divisor;

    unsigned int seed = 1;

    double epsilon = 0.00001;
    double sigma = 0.75;
    double xi = 0.25;

    for (auto _ : state)
    {
        state.PauseTiming();
        std::vector<double> M_host;
        std::vector<double> q_host;
        create_random_sparse(n, seed, sparsity, M_host, q_host);
        std::vector<double> res_host(n);
        state.ResumeTiming();

        thrust::device_vector<double> q = q_host;

        thrust::device_vector<double> z_0(n);
        thrust::device_vector<double> res;
        thrust::device_vector<double> M = M_host;
        int status = LCP_Newton(n, M, q, z_0, epsilon, xi, sigma, res);

        thrust::copy(res.begin(), res.end(), res_host.begin());
        cudaDeviceSynchronize();
        if (status != 0)
        {
            printf("Uh oh, solve has failed with status %d\n", status);
        }
    }
}

static void BM_cuLCP_random_sparsity_with_sparse(benchmark::State &state) {
    int n = 10000;
    int divisor = state.range(1);
    int subtractor = 10 - state.range(0);
    double sparsity = ((double)(divisor - subtractor)) / divisor;

    
    unsigned int seed = 1;
    // set to one because 0 was causing an unknown error - tracked down to cudss analyse failing for a specific matrix
    // have submitted bug report to nvidia but for the mean time this will do 

    double epsilon = 0.00001;
    double sigma = 0.75;
    double xi = 0.25;

    for (auto _ : state)
    {
        state.PauseTiming();
        std::vector<double> M_host;
        std::vector<double> q_host;

        create_random_sparse(n, seed, sparsity, M_host, q_host);
        host_matrix_sparse f_host = matrix_to_csr(n, M_host);

        std::vector<double> res_host(n);
        state.ResumeTiming();

        thrust::device_vector<double> q = q_host;
        matrix_sparse f = {f_host.row_offsets, f_host.column_indices, f_host.values};

        thrust::device_vector<double> z_0(n);
        thrust::device_vector<double> res;
        int status = LCP_Newton(n, f, q, z_0, epsilon, xi, sigma, res);

        thrust::copy(res.begin(), res.end(), res_host.begin());
        cudaDeviceSynchronize();
        if (status != 0)
        {
            printf("Uh oh, solve has failed with status %d\n", status);
        }
    }
}

static void BM_path_random_sparsity(benchmark::State &state)
{
    int n = 10000;
    int divisor = state.range(1);
    int subtractor = 10 - state.range(0);
    double sparsity = ((double) (divisor - subtractor))/divisor;


    unsigned int seed = 1;

    for (auto _ : state)
    {
        state.PauseTiming();
        std::vector<double> M;
        std::vector<double> q;
        create_random_sparse(n, seed, sparsity, M, q);
        initializeM(M, n);
        initializeQ(q, n);
        std::vector<double> z(n);
        std::vector<double> f(n);
        state.ResumeTiming();

        int status = path_solve(n, z.data(), f.data());
        if (status != 1)
        {
            printf("Uh oh, solve has failed with status %d\n", status);
        }
    }
}

static void BM_path_random_sparsity_with_sparse(benchmark::State &state)
{
    int n = 10000;
    int divisor = state.range(1);
    int subtractor = 10 - state.range(0);
    double sparsity = ((double) (divisor - subtractor))/divisor;


    unsigned int seed = 1;

    for (auto _ : state)
    {
        state.PauseTiming();
        std::vector<double> M;
        std::vector<double> q;
        create_random_sparse(n, seed, sparsity, M, q);
        initializeM_sparse(M, n);
        initializeQ(q, n);
        std::vector<double> z(n);
        std::vector<double> f(n);
        state.ResumeTiming();

        int status = path_solve(n, z.data(), f.data());
        if (status != 1)
        {
            printf("Uh oh, solve has failed with status %d\n", status);
        }
    }
}


BENCHMARK(BM_cuLCP_SOLVER)->RangeMultiplier(2)->Range(8, 8 << 10)->Unit(benchmark::kSecond)->Iterations(10)->UseRealTime();
BENCHMARK(BM_pathSolver)->RangeMultiplier(2)->Range(8, 8 << 8)->Unit(benchmark::kSecond)->Iterations(10)->UseRealTime();
BENCHMARK(BM_pathSolver)->RangeMultiplier(2)->Range(8 << 9, 8 << 10)->Unit(benchmark::kSecond)->Iterations(2);
BENCHMARK(BM_LemkeMethod)->RangeMultiplier(2)->Range(8, 8 << 8)->Unit(benchmark::kSecond)->Iterations(10);
BENCHMARK(BM_LemkeMethod)->RangeMultiplier(2)->Range(8 << 9, 8 << 9)->Unit(benchmark::kSecond)->Iterations(2);
BENCHMARK(BM_cuLCP_Porous)->RangeMultiplier(2)->Range(128, 4096)->Unit(benchmark::kSecond)->Iterations(10)->UseRealTime();
BENCHMARK(BM_pathSolverPorous)->RangeMultiplier(2)->Range(128, 4096)->Unit(benchmark::kSecond)->Iterations(10);
BENCHMARK(BM_LemkePorous)->RangeMultiplier(2)->Range(128, 1024)->Unit(benchmark::kSecond)->Iterations(10);
BENCHMARK(BM_pathSolverBimatrix)->Unit(benchmark::kSecond)->Iterations(1);
BENCHMARK(BM_cuLCP_random_sparsity)->ArgsProduct({benchmark::CreateDenseRange(0,9,1), benchmark::CreateRange(1000, 10000, 10)})->Unit(benchmark::kSecond)->Iterations(1)->UseRealTime();
BENCHMARK(BM_cuLCP_random_sparsity_with_sparse)->ArgsProduct({benchmark::CreateDenseRange(0,9,1), benchmark::CreateRange(1000, 10000, 10)})->Unit(benchmark::kSecond)->Iterations(1)->UseRealTime();
BENCHMARK(BM_path_random_sparsity)->ArgsProduct({benchmark::CreateDenseRange(0,9,1), benchmark::CreateRange(1000, 10000, 10)})->Unit(benchmark::kSecond)->Iterations(10);
BENCHMARK(BM_path_random_sparsity_with_sparse)->ArgsProduct({benchmark::CreateDenseRange(0, 9, 1), benchmark::CreateRange(1000, 10000, 10)})->Unit(benchmark::kSecond)->Iterations(10);
BENCHMARK_MAIN();