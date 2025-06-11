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
#include <cmath>

#include <cusolverDn.h>
#include "LCP_newton.hpp"

void row_matrix(int n, thrust::device_vector<double> &M, thrust::device_vector<int> &alpha, thrust::device_vector<double> &res) {
    int size = n * alpha.size();
    int num_rows_out = static_cast<int>(alpha.size());
    res.resize(size);

    const double* M_ptr = thrust::raw_pointer_cast(M.data());
    const int* alpha_ptr = thrust::raw_pointer_cast(alpha.data());
    double* res_ptr = thrust::raw_pointer_cast(res.data());

    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(size),
        [=] __device__(int idx)
        {
            int row_out = idx % num_rows_out;
            int col = idx / num_rows_out;

            int selected_row = alpha_ptr[row_out];
            res_ptr[idx] = M_ptr[selected_row + col * n];
        });
}

void gradient(int n, thrust::device_vector<double> &z, thrust::device_vector<double> &w, thrust::device_vector<int> &alpha, thrust::device_vector<int> &gamma, thrust::device_vector<double> &M, cublasHandle_t &handle, thrust::device_vector<double> &res)
{

    thrust::device_vector<double> M_alpha;
    row_matrix(n, M, alpha, M_alpha);

    res.clear();
    res.resize(n);
    thrust::scatter(
        thrust::make_permutation_iterator(z.begin(), gamma.begin()),
        thrust::make_permutation_iterator(z.begin(), gamma.end()),
        gamma.begin(),
        res.begin()
    );

    thrust::device_vector<double> w_alpha(alpha.size());
    thrust::gather(alpha.begin(), alpha.end(), w.begin(), w_alpha.begin());

    double a = 1.0;
    cublasDgemv(handle, CUBLAS_OP_T, alpha.size(), n, &a, M_alpha.data().get(), alpha.size(), w_alpha.data().get(), 1, &a, res.data().get(), 1);
}

int gradient_step(int n, double step_size, thrust::device_vector<double> &z, thrust::device_vector<double> &w, thrust::device_vector<int> &alpha, thrust::device_vector<int> &gamma, thrust::device_vector<double> &M, thrust::device_vector<double> &q, cublasHandle_t &handle) {
    thrust::device_vector<int> old_alpha(alpha.begin(), alpha.end());
    for (int i = 0; i < 100; i++) {
        thrust::device_vector<double> grad;
        gradient(n, z, w, alpha, gamma, M, handle, grad);
        if (thrust::all_of(
                grad.begin(),
                grad.end(),
                [] __device__(double x)
                {
                    return fabs(x) < 1e-5;
                })) {
                    return 1;
                }
        cublasDaxpy(handle, n, &step_size, grad.data().get(), 1, z.data().get(), 1);
        eval_linear(n, M, q, z, w, handle);
        alpha_set(n, z, w, alpha, gamma);

        if (!(alpha == old_alpha || !norm_termination_test(n, z, w, 1e5, handle))) {
            return 0;
        }
    }
    return 0;
}