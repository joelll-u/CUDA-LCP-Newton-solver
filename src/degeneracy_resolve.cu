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
#include "LCP_newton.hpp"

void gradient(int n, thrust::device_vector<double> &z, thrust::device_vector<double> &w, thrust::device_vector<int> &alpha, thrust::device_vector<int> &gamma, thrust::device_vector<double> &M, cublasHandle_t &handle, thrust::device_vector<double> &res)
{

    res.resize(n);
    thrust::device_vector<double> grad_gamma(gamma.size());
    thrust::gather(gamma.begin(), gamma.end(), z.begin(), grad_gamma.begin());

    thrust::device_vector<double> Mw(n);
    double a = 1.0;
    double b = 0.0;
    cublasDgemv(handle, CUBLAS_OP_T, n, n, &a, M.data().get(), n, w.data().get(), 1, &b, Mw.data().get(), 1);
    thrust::device_vector<double> Mw_alpha(alpha.size());
    thrust::gather(alpha.begin(), alpha.end(), Mw.begin(), Mw_alpha.begin());
    thrust::scatter(grad_gamma.begin(), grad_gamma.end(), gamma.begin(), res.begin());
    thrust::scatter(Mw_alpha.begin(), Mw_alpha.end(), alpha.begin(), res.begin());
}

void gradient_step(int n, double step_size, thrust::device_vector<double> &z, thrust::device_vector<double> &w, thrust::device_vector<int> &alpha, thrust::device_vector<int> &gamma, thrust::device_vector<double> &M, thrust::device_vector<double> &q, cublasHandle_t &handle) {
    thrust::device_vector<int> old_alpha(alpha.begin(), alpha.end());
    for (int i = 0; i < 10; i++) {
        thrust::device_vector<double> grad;
        gradient(n, z, w, alpha, gamma, M, handle, grad);
        cublasDaxpy(handle, n, &step_size, grad.data().get(), 1, z.data().get(), 1);
        eval_linear(n, M, q, z, w, handle);
        alpha_set(n, z, w, alpha, gamma);
        printf("z: "); for(int i = 0; i < n; i++) {printf("%f ", (double) z[i]);}; printf("\n");
        if (!(alpha == old_alpha || !norm_termination_test(n, z, w, 1e5, handle))) {
            return;
        }
    }
    assert(false);
}