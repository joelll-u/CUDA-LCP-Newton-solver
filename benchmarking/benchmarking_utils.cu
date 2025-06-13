#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>
#include <random>
#include <unordered_set>
#include <thrust/transform.h>
#include "LCP_newton.hpp"


typedef struct
{
    std::vector<int> row_offsets;
    std::vector<int> column_indices;
    std::vector<double> values;
} host_matrix_sparse;

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

host_matrix_sparse create_sparse_porous_matrix(int N) {
    host_matrix_sparse crs;
    crs.row_offsets.push_back(0); // First row starts at index 0

    for (int blockRow = 0; blockRow < N; ++blockRow)
    {
        for (int localRow = 0; localRow < N; ++localRow)
        {
            int globalRow = blockRow * N + localRow;

            // For each row, build its non-zero entries
            // These may come from up to 3 blocks: prev (-I), current (S), next (-I)

            // 1. Previous block: -I
            if (blockRow > 0)
            {
                int globalCol = (blockRow - 1) * N + localRow;
                crs.values.push_back(-1.0);
                crs.column_indices.push_back(globalCol);
            }

            // 2. Current block: S (tridiagonal -1, 4, -1)
            for (int offset = -1; offset <= 1; ++offset)
            {
                int localCol = localRow + offset;
                if (localCol >= 0 && localCol < N)
                {
                    int globalCol = blockRow * N + localCol;
                    double val = (offset == 0) ? 4.0 : -1.0;
                    crs.values.push_back(val);
                    crs.column_indices.push_back(globalCol);
                }
            }

            // 3. Next block: -I
            if (blockRow < N - 1)
            {
                int globalCol = (blockRow + 1) * N + localRow;
                crs.values.push_back(-1.0);
                crs.column_indices.push_back(globalCol);
            }

            crs.row_offsets.push_back(static_cast<int>(crs.values.size()));
        }
    }

    return crs;
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

host_matrix_sparse matrix_to_csr(int n, std::vector<double> &M) {

    std::vector<int> csr_row_ptr;
    std::vector<double> csr_values;
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

void delete_random_non_diag(int n, int num_to_be_deleted, unsigned int &seed, std::vector<double> &M) {
    std::mt19937 rng(seed);

    const int N = n * n;
    const int D = n;
    const int ND = N - D;

    if (num_to_be_deleted <= ND / 2)
    {
        std::unordered_set<int> chosen;
        std::uniform_int_distribution<int> dist(0, N - 1);
        int deleted = 0;

        while (deleted < num_to_be_deleted)
        {
            int idx = dist(rng);
            int row = idx % n;
            int col = idx / n;

            if (row != col && chosen.insert(idx).second)
            {
                M[idx] = 0.0;
                ++deleted;
            }
        }
    }
    else
    {
        std::vector<int> non_diag_indices;
        non_diag_indices.reserve(ND);

        for (int col = 0; col < n; ++col)
        {
            for (int row = 0; row < n; ++row)
            {
                if (row != col)
                {
                    non_diag_indices.push_back(col * n + row);
                }
            }
        }

        std::shuffle(non_diag_indices.begin(), non_diag_indices.end(), rng);

        std::unordered_set<int> to_keep(
            non_diag_indices.begin(),
            non_diag_indices.begin() + (ND - num_to_be_deleted));

        for (int idx : non_diag_indices)
        {
            if (to_keep.find(idx) == to_keep.end())
            {
                M[idx] = 0.0;
            }
        }
    }
    seed++;
}

void delete_random_diag(int n, int num_to_be_deleted, unsigned int &seed, std::vector<double> &M) {
    if (num_to_be_deleted > n)
    {
        throw std::invalid_argument("Cannot delete more diagonal elements than exist.");
    }
    std::mt19937 rng(seed);

    std::vector<int> diag_positions(n);
    for (int i = 0; i < n; ++i)
    {
        diag_positions[i] = i;
    }

    std::shuffle(diag_positions.begin(), diag_positions.end(), rng);

    for (int i = 0; i < num_to_be_deleted; ++i)
    {
        int diag_idx = diag_positions[i] * (n + 1);
        M[diag_idx] = 0.0;
    }

    seed++;
}

void create_random_sparse(int n, unsigned int &seed, double sparsity, std::vector<double> &M, std::vector<double> &q)
{
    create_random_P_matrix(n, -5, 5, seed, M);
    create_random_vector(n, -500, 500, seed, q);

    int num_deleted = sparsity * (n * n) - n;

    delete_random_non_diag(n, num_deleted, seed, M);
}