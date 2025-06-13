#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>

#include "LCP_newton.hpp"
#include "benchmarking_utils.cu"

#if !defined(MAX)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#if !defined(MIN)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

int main(int argc, char *argv[])
{
    double epsilon = 0.00001;
    double sigma = 0.75;
    double xi = 0.25;

    unsigned int seed = 1;
    int status;


    // for this test we temporarily modified cuLCP to return the iteration number it was o 
    for (int n = 128; n <= 4092; n *= 2) {

        int dense_max_iters = 0;
        int dense_total_iters = 0;
        int dense_min_iters = 1e8;

        for (int i = 0; i < 100; i++) {
            std::vector<double> M;
            std::vector<double> q;
        
            create_random_P_matrix(n, -5, 5, seed, M);
            create_random_vector(n, -500, 500, seed, q);

            matrix_dense M_d = M;
            thrust::device_vector<double> res;
            thrust::device_vector<double> q_d = q;
            thrust::device_vector<double> z_0(n);
            status = LCP_Newton(n, M_d, q_d, z_0, epsilon, xi, sigma, res) + 1;

            dense_total_iters += status;
            dense_min_iters = MIN(dense_min_iters, status);
            dense_max_iters = MAX(dense_max_iters, status);
        }

        std::cout << "Size: " << n << " Avg: " << dense_total_iters/100.0 << " Min: " << dense_min_iters << " Max: " << dense_max_iters << std::endl;

        int sp_max_iters = 0;
        int sp_total_iters = 0;
        int sp_min_iters = 1e8;

        for (int i = 0; i < 100; i++)
        {
            std::vector<double> M;
            std::vector<double> q;

            double sparsity = (n==128) ? 0.99 : 0.999;
            create_random_sparse(n, seed, sparsity, M, q);

            thrust::device_vector<double> res;
            thrust::device_vector<double> q_d = q;
            thrust::fill(q_d.begin(), q_d.end(), -1.0);
            thrust::device_vector<double> z_0(n);

            host_matrix_sparse x_host = matrix_to_csr(n, M);
            matrix_sparse x = {x_host.row_offsets, x_host.column_indices, x_host.values};
            status = LCP_Newton(n, x, q_d, z_0, epsilon, xi, sigma, res) + 1;
            cudaDeviceSynchronize();

            sp_total_iters += status;
            sp_min_iters = MIN(sp_min_iters, status);
            sp_max_iters = MAX(sp_max_iters, status);
        }

        std::cout << "Size: " << n << " Avg: " << sp_total_iters / 100.0 << " Min: " << sp_min_iters << " Max: " << sp_max_iters << std::endl;
    }

    cudaDeviceSynchronize();
    if (status != 0)
    {
        printf("Uh oh, solve has failed with status %d\n", status);
    }
}