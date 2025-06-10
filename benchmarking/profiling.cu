#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>

#include "LCP_newton.hpp"
#include "benchmarking_utils.cu"

bool parseBool(const std::string &str)
{
    return (str == "true" || str == "1");
}

int main(int argc, char* argv[]) {
    int n = 1000;
    double sparsity = 1.0;
    bool use_sparse = false;
    if (argc > 1)
    {
        std::istringstream ss_n(argv[1]);
        if (!(ss_n >> n))
        {
            std::cerr << "Invalid integer for n. Using default: " << n << "\n";
        }
    }

    if (argc > 2)
    {
        std::istringstream ss_sparsity(argv[2]);
        if (!(ss_sparsity >> sparsity))
        {
            std::cerr << "Invalid double for sparsity. Using default: " << sparsity << "\n";
        }
    }

    if (argc > 3)
    {
        use_sparse = parseBool(argv[3]);
    }

    unsigned int speed;
    std::vector<double> M;
    std::vector<double> q;

    unsigned int seed = 0;
    create_random_sparse(n, seed, sparsity, M, q);

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