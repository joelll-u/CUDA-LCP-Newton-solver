#include "LCP_newton.hpp"
#include <cudss.h>
#include <thrust/zip_function.h>
#include <thrust/host_vector.h>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/iterator/discard_iterator.h>

struct findIndex
{
    const int *row_offsets;
    int num_rows;

    __host__ __device__
    findIndex(const int *row_offsets_, int num_rows_) : row_offsets(row_offsets_), num_rows(num_rows_){}

    __host__ __device__ int operator()(int i) const
    {
        return thrust::upper_bound(thrust::seq, row_offsets, row_offsets + num_rows + 1, i) - row_offsets - 1;
    }
};

struct inAlpha
{
    const int *alpha;
    const int k;

    __host__ __device__
    inAlpha(const int *alpha_, const int k_) : alpha(alpha_), k(k_){}

    __host__ __device__ int operator()(thrust::tuple<int, int, double> i) const
    {
        int row = thrust::get<0>(i);
        int col = thrust::get<1>(i);

        bool row_in = thrust::binary_search(thrust::seq, alpha, alpha + k, row);
        bool col_in = thrust::binary_search(thrust::seq, alpha, alpha + k, col);

        return row_in && col_in;
    }
};

void sparse_submatrix(int N, matrix_sparse &M, thrust::device_vector<int> &alpha, matrix_sparse &M_alpha)
{
    thrust::device_vector<int> rows(M.column_indices.size());
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator((int) M.column_indices.size()),
        rows.begin(),
        findIndex(M.row_offsets.data().get(), N)
    );

    thrust::device_vector<int> new_rows(rows.size());
    M_alpha.column_indices.resize(rows.size());
    M_alpha.values.resize(rows.size());
    auto iterator = thrust::make_zip_iterator(new_rows.begin(), M_alpha.column_indices.begin(), M_alpha.values.begin());
    auto end_it = thrust::copy_if(
        thrust::make_zip_iterator(rows.begin(), M.column_indices.begin(), M.values.begin()),
        thrust::make_zip_iterator(rows.end(), M.column_indices.end(), M.values.end()),
        iterator,
        inAlpha(alpha.data().get(), alpha.size())
    );

    int new_size = end_it - iterator;

    new_rows.resize(new_size);
    M_alpha.column_indices.resize(new_size);
    M_alpha.values.resize(new_size);

    thrust::lower_bound(
        alpha.begin(), alpha.end(),
        M_alpha.column_indices.begin(), M_alpha.column_indices.end(),
        M_alpha.column_indices.begin()
    );

    M_alpha.row_offsets.resize(new_rows.size() + 1);

    auto counting_begin = thrust::make_counting_iterator(0);

    auto end = thrust::unique_by_key_copy(
        new_rows.begin(), new_rows.end(),
        counting_begin,
        thrust::make_discard_iterator(),
        M_alpha.row_offsets.begin()
    );

    int num_unique = end.second - M_alpha.row_offsets.begin();
    M_alpha.row_offsets[num_unique] = M_alpha.column_indices.size();
    M_alpha.row_offsets.resize(num_unique + 1);

}

int sparse_solve(int n, matrix_sparse &A_sparse, thrust::device_vector<double> &q, thrust::device_vector<double> &res)
{

    nvtxRangePushA("setup_solve");
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
        1,
        n,
        res.data().get(),
        CUDA_R_64F,
        CUDSS_LAYOUT_COL_MAJOR
    );

    nvtxRangePop();
    nvtxRangePushA("analysis");
    cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, A, x, b);
    nvtxRangePop();
    nvtxRangePushA("factorization");
    cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, A, x, b);
    nvtxRangePop();
    nvtxRangePushA("solving");
    cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A, x, b);
    nvtxRangePop();
    nvtxRangePushA("teardown");

    cudssConfigDestroy(config);
    cudssDataDestroy(handle, data);
    cudssMatrixDestroy(A);
    cudssMatrixDestroy(x);
    cudssMatrixDestroy(b);
    cudssDestroy(handle);
    nvtxRangePop();
    return 0;
}
