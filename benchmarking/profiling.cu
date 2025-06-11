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
    int n = 5000;
    double sparsity = 0.0;
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
    std::vector<double> M;
    std::vector<double> q;

    unsigned int seed = 1;
    if (sparsity > 0)
    {
        create_random_sparse(n, seed, sparsity, M, q);
    } else {
        create_random_P_matrix(n, -5, 5, seed, M);
        create_random_vector(n, -500, 500, seed, q);
    }
    host_matrix_sparse x_host = matrix_to_csr(n, M);
    matrix_sparse x = {x_host.row_offsets, x_host.column_indices, x_host.values};

    double epsilon = 0.00001;
    double sigma = 0.75;
    double xi = 0.25;

    std::vector<double> res_host(n);

    thrust::device_vector<double> res;
    thrust::device_vector<double> q_d = q;
    thrust::device_vector<double> z_0(n, 0);

    int status = LCP_Newton(n, x, q_d, z_0, epsilon, xi, sigma, res);

    thrust::copy(res.begin(), res.end(), res_host.begin());
    cudaDeviceSynchronize();
    if (status != 0)
    {
        printf("Uh oh, solve has failed with status %d\n", status);
    }
}