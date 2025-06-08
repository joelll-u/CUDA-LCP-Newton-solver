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

        // pathMain(n, n * n, &status, z, f, lb, ub);
        // printf("M: \n");
        // for (int i = 0; i < n; i++) {
        //     for (int j = 0; j < n; j++) {
        //         printf("%f ", M_host[i*n + j]);
        //     }
        //     printf("\n");
        // }

        // printf("Q: \n");
        // for (int i = 0; i < n; i++) {
        //     printf("%f ", q_host[i]);
        // }
        // printf("\n");

        // printf("Mz+q: \n");
        // funcEval(n, z, f);
        // for (int i = 0; i < n; i++) {
        //     printf("%f ", z[i]);
        // }
        // printf("\n");

        // for (int i = 0; i < n; i++) {
        //     printf("%f ", f[i]);
        // }
        // printf("\n");

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

        // pathMain(n, n * n, &status, z, f, lb, ub);
        // printf("M: \n");
        // for (int i = 0; i < n; i++) {
        //     for (int j = 0; j < n; j++) {
        //         printf("%f ", M_host[i*n + j]);
        //     }
        //     printf("\n");
        // }

        // printf("Q: \n");
        // for (int i = 0; i < n; i++) {
        //     printf("%f ", q_host[i]);
        // }
        // printf("\n");

        // printf("Mz+q: \n");
        // funcEval(n, z, f);
        // for (int i = 0; i < n; i++) {
        //     printf("%f ", z[i]);
        // }
        // printf("\n");

        // for (int i = 0; i < n; i++) {
        //     printf("%f ", f[i]);
        // }
        // printf("\n");

        if (status != 1)
        {
            printf("Uh oh, solve has failed with status %d!\n", status);
        }
    }
}

static void BM_cuLCP_Porous(benchmark::State &state) {
    int n = state.range(0);
    std::vector<double> M;
    create_porous_matrix(n, M);
    std::vector<double> q;
    create_porous_q(n, q);

    matrix_sparse x = matrix_to_csr(n*n, M);
        
    // printf("M: \n");
    // for (int i = 0; i < n*n; i++) {
    //     for (int j = 0; j < n*n; j++) {
    //         printf("%f ", M[i*n*n + j]);
    //     }
    //     printf("\n");
    // }

    double epsilon = 0.00001;
    double sigma = 0.75;
    double xi = 0.25;

    // for (int i = 0; i < n*n; i++) {
    //     printf("%f ", q[i]);
    // }
    // printf("\n");

    std::vector<double> res_host(n*n);
    thrust::device_vector<double> M_d = M;
    thrust::device_vector<double> q_d = q;
    thrust::device_vector<double> z_0(n * n, 0);
    for (auto _ : state)
    {

        thrust::device_vector<double> res;

        int status = LCP_Newton(n*n, M_d, q_d, z_0, epsilon, xi, sigma, res, true, &x);

        // thrust::copy(res.begin(), res.end(), res_host.begin());
        // cudaDeviceSynchronize();
        if (status != 0) {
            printf("Uh oh, solve has failed with status %d\n", status);
        }
    }
}

static void BM_pathSolverPorous(benchmark::State &state)
{
    int n = state.range(0);
    std::vector<double> M_host;
    create_porous_matrix(n, M_host);
    initializeM(M_host, n*n);

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
    create_porous_matrix(n, M_host);
    initializeM(M_host, n * n);

    std::vector<double> q_host;
    create_porous_q(n, q_host);
    initializeQ(q_host, n * n);

    std::vector<double> z(n * n);
    std::vector<double> f(n * n);

    int status = 0;

    for (auto _ : state)
    {
        status = path_solve(n * n, z.data(), f.data());

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
    int n = 1000;
    double sparsity = state.range(0)/100.0;
    unsigned int seed = 0;

    double epsilon = 0.00001;
    double sigma = 0.75;
    double xi = 0.25;

    for (auto _ : state)
    {
        state.PauseTiming();
        std::vector<double> M_host;
        std::vector<double> q_host;
        create_random_sparse(n, seed, sparsity, M_host, q_host);

        
        state.ResumeTiming();
        thrust::device_vector<double> M = M_host;
        thrust::device_vector<double> q = q_host;

        thrust::device_vector<double> z_0(n);
        thrust::device_vector<double> res;

        int status = LCP_Newton(n, M, q, z_0, epsilon, xi, sigma, res);

        // thrust::copy(res.begin(), res.end(), res_host.begin());
        // cudaDeviceSynchronize();
        if (status != 0)
        {
            printf("Uh oh, solve has failed with status %d\n", status);
        }
    }
}

static void BM_path_random_sparsity(benchmark::State &state)
{
    int n = 1000;
    double sparsity = state.range(0) / 100.0;
    unsigned int seed = 0;

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
        // thrust::copy(res.begin(), res.end(), res_host.begin());
        // cudaDeviceSynchronize();
        if (status != 1)
        {
            printf("Uh oh, solve has failed with status %d\n", status);
        }
    }
}

// BENCHMARK(BM_cuLCP_SOLVER)->RangeMultiplier(2)->Range(8, 8 << 10)->Unit(benchmark::kSecond)->Iterations(10)->UseRealTime();
// BENCHMARK(BM_pathSolver)->RangeMultiplier(2)->Range(8, 8 << 8)->Unit(benchmark::kSecond)->Iterations(10)->UseRealTime();
// BENCHMARK(BM_pathSolver)->RangeMultiplier(2)->Range(8 << 9, 8 << 10)->Unit(benchmark::kSecond)->Iterations(2);
// BENCHMARK(BM_LemkeMethod)->RangeMultiplier(2)->Range(8, 8 << 8)->Unit(benchmark::kSecond)->Iterations(10);
// BENCHMARK(BM_LemkeMethod)->RangeMultiplier(2)->Range(8 << 9, 8 << 10)->Unit(benchmark::kSecond)->Iterations(2);
// BENCHMARK(BM_cuLCP_Porous)->RangeMultiplier(2)->Range(8, 64)->Unit(benchmark::kSecond)->Iterations(10);
// BENCHMARK(BM_pathSolverPorous)->RangeMultiplier(2)->Range(8, 64)->Unit(benchmark::kSecond)->Iterations(10);
// BENCHMARK(BM_LemkePorous)->RangeMultiplier(2)->Range(8, 64)->Unit(benchmark::kSecond)->Iterations(10);
// BENCHMARK(BM_pathSolverBimatrix)->Unit(benchmark::kSecond)->Iterations(1);
BENCHMARK(BM_cuLCP_random_sparsity)->DenseRange(90, 99, 1)->Unit(benchmark::kSecond)->Iterations(10)->UseRealTime();
BENCHMARK(BM_path_random_sparsity)->DenseRange(90, 99, 1)->Unit(benchmark::kSecond)->Iterations(10);
BENCHMARK_MAIN();