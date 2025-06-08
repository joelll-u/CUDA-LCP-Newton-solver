#include "LCP_newton.hpp"
#include <cudss.h>
#include <thrust/host_vector.h>

void sparse_submatrix(int N, matrix_sparse &M, thrust::device_vector<int> &alpha, matrix_sparse &M_alpha) {
    // Copy alpha to host for quick lookup and index mapping
    thrust::host_vector<int> h_alpha = alpha;
    int k = h_alpha.size();

    // Create a map from global index -> new index (since alpha is sorted, direct index mapping)
    std::vector<int> index_map(N, -1);
    for (int i = 0; i < k; ++i)
        index_map[h_alpha[i]] = i;

    // Copy sparse matrix to host
    thrust::host_vector<int> h_row_offsets = M.row_offsets;
    thrust::host_vector<int> h_column_indices = M.column_indices;
    thrust::host_vector<double> h_values = M.values;

    // Output containers
    std::vector<int> new_row_offsets(k + 1, 0);
    std::vector<int> new_column_indices;
    std::vector<double> new_values;

    // Construct the submatrix
    for (int new_row = 0; new_row < k; ++new_row)
    {
        int old_row = h_alpha[new_row];
        int row_start = h_row_offsets[old_row];
        int row_end = h_row_offsets[old_row + 1];

        for (int j = row_start; j < row_end; ++j)
        {
            int col = h_column_indices[j];
            if (index_map[col] != -1)
            {
                new_column_indices.push_back(index_map[col]);
                new_values.push_back(h_values[j]);
            }
        }

        new_row_offsets[new_row + 1] = new_column_indices.size();
    }

    // Copy results back to device
    M_alpha.row_offsets = thrust::device_vector<int>(new_row_offsets.begin(), new_row_offsets.end());
    M_alpha.column_indices = thrust::device_vector<int>(new_column_indices.begin(), new_column_indices.end());
    M_alpha.values = thrust::device_vector<double>(new_values.begin(), new_values.end());
}

    int sparse_solve(int n, matrix_sparse &A_sparse, thrust::device_vector<double> &q, thrust::device_vector<double> &res)
{
    cudssHandle_t handle;
    cudssConfig_t config;
    cudssData_t data;

    cudssCreate(&handle);
    cudssConfigCreate(&config);
    cudssDataCreate(handle, &data);

    cudssMatrix_t A;
    cudssMatrix_t b;
    cudssMatrix_t x;

    cudssMatrixCreateCsr(
        &A,
        n,
        n,
        A_sparse.values.size(),
        A_sparse.row_offsets.data().get(),
        nullptr,
        A_sparse.column_indices.data().get(),
        A_sparse.values.data().get(),
        CUDA_R_32I,
        CUDA_R_64F,
        CUDSS_MTYPE_GENERAL,
        CUDSS_MVIEW_FULL,
        CUDSS_BASE_ZERO
    );

    cudssMatrixCreateDn(
        &b,
        n,
        1,
        n,
        q.data().get(),
        CUDA_R_64F,
        CUDSS_LAYOUT_COL_MAJOR
    );

    res.resize(n);
    cudssMatrixCreateDn(
        &x,
        n,
        n,
        n,
        res.data().get(),
        CUDA_R_64F,
        CUDSS_LAYOUT_COL_MAJOR
    );


    cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, A, x, b);
    cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, A, x, b);
    cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A, x, b);

    cudssConfigDestroy(config);
    cudssDataDestroy(handle, data);
    cudssMatrixDestroy(A);
    cudssMatrixDestroy(x);
    cudssMatrixDestroy(b);
    cudssDestroy(handle);
    return 0;
}
