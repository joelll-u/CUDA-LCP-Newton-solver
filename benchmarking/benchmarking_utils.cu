#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>
#include <random>
#include <thrust/transform.h>
#include "LCP_newton.hpp"


void create_random_vector(int n, double a, double b, unsigned int &seed, std::vector<double> &res)
{
    res.reserve(n);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(a, b);

    for (int i = 0; i < n; ++i)
    {
        res.push_back(dist(rng));
    }
    seed++;
    return;
}

void create_random_P_matrix(int n, double a, double b, unsigned int &seed, std::vector<double> &res)
{
    std::vector<double> M_host;
    create_random_vector(n*n, a, b, seed, M_host);

    thrust::device_vector<double> M_unsymm = M_host;

    std::vector<double> B_host_unfull;
    create_random_vector((n*(n-1))/2, a, b, seed, B_host_unfull);

    // 00 10
    // 01 11
    // 02 12
    // 03 13
    //
    //

    std::vector<double> B_host(n*n);
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            M_host[i*n + j] += B_host_unfull[cnt];
            M_host[j*n + i] -= B_host_unfull[cnt];
            cnt++;
        }
    }


    thrust::device_vector<double> M(n * n);

    cublasHandle_t handle;
    cublasCreate(&handle);
    double one = 1.0;

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &one, thrust::raw_pointer_cast(M_unsymm.data()), n, thrust::raw_pointer_cast(M_unsymm.data()), n, &one, thrust::raw_pointer_cast(M.data()), n);
    cudaDeviceSynchronize();

    res.resize(n * n);
    thrust::copy(M.begin(), M.end(), res.begin());

    std::vector<double> etas;
    create_random_vector(n, 0, 0.3, seed, etas);

    cudaDeviceSynchronize();

    for (int i = 0; i < n; i++) {
        M[i*n + i] += etas[i];
    }
}

void create_porous_matrix(int n, std::vector<double> &res) {
    int N = n*n;
    res.clear();
    res.resize(N*N);
    for (int i = 0; i < N; i++) {
        res[i*N + i] = 4;
        if (i < N - 1 && i + n < N) {
            res[i*N + i + n] = -1;
        }
        if (i > 0 && i - n >= 0) {
            res[i*N + i - n] = -1;
        }
        if (i % n != 0) {
            res[i*N + i - 1] = -1;
        }
        if (i % n != n-1) {
            res[i*N + i + 1] = -1;
        }
    }
}

void create_porous_q(int n, std::vector<double> &res) {
    res.resize(n*n);
    for (int i = 0; i < n*n; i++) {
        if (i % 2) {
            res[i] = 1;
        } else {
            res[i] = -1;
        }
    }
}

sparse_format matrix_to_csr(int n, std::vector<double> &M) {

    std::vector<int> csr_row_ptr;
    std::vector<int> csr_values;
    std::vector<int> csr_col_indices;

    std::vector<int> row_nnz(n, 0);
    for (int col = 0; col < n; ++col)
    {
        for (int row = 0; row < n; ++row)
        {
            double val = M[col * n + row];
            if (val != 0.0)
            {
                row_nnz[row]++;
            }
        }
    }

    csr_row_ptr.resize(n + 1, 0);
    for (int i = 0; i < n; ++i)
    {
        csr_row_ptr[i + 1] = csr_row_ptr[i] + row_nnz[i];
    }

    int nnz = csr_row_ptr[n];
    csr_values.resize(nnz);
    csr_col_indices.resize(nnz);

    std::vector<int> row_offset = csr_row_ptr;

    for (int col = 0; col < n; ++col)
    {
        for (int row = 0; row < n; ++row)
        {
            double val = M[col * n + row];
            if (val != 0.0)
            {
                int idx = row_offset[row]++;
                csr_values[idx] = val;
                csr_col_indices[idx] = col;
            }
        }
    }

    return {csr_row_ptr, csr_col_indices, csr_values};
}