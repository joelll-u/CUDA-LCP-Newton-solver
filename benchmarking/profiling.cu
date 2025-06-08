#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <vector>

#include "LCP_newton.hpp"
#include "benchmarking_utils.cu"

int main() {
    int n = 128;
    std::vector<double> M;
    create_porous_matrix(n, M);
    std::vector<double> q;
    create_porous_q(n, q);

    matrix_sparse x = matrix_to_csr(n * n, M);

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

    std::vector<double> res_host(n * n);


    thrust::device_vector<double> res;
    thrust::device_vector<double> q_d = q;
    thrust::device_vector<double> z_0(n * n, 0);

    int status = LCP_Newton(n * n, x, q_d, z_0, epsilon, xi, sigma, res);

    thrust::copy(res.begin(), res.end(), res_host.begin());
    cudaDeviceSynchronize();
    if (status != 0)
    {
        printf("Uh oh, solve has failed with status %d\n", status);
    }
}