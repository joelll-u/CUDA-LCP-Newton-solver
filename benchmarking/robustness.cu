#include "benchmarking_utils.cu"
#include "LCP_newton.hpp"

int bimatrix_game(int n, unsigned int &seed) {
    std::vector<double> A;
    std::vector<double> B;
    create_random_vector(n*n, 1, 5, seed, A);
    create_random_vector(n*n, 1, 5, seed, B);

    std::vector<double> M_host(4 * n * n, 0);
    // printf("%d %d\n", A.size(), B.size());
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // printf("%d \n", (i*n + j));
            M_host[i * (2*n) + j + n] = A[i*n + j];
            M_host[(i + n) * (2*n) + j] = B[i*n + j];
        }
    }

    thrust::device_vector<double> M(M_host.begin(), M_host.end());
    thrust::device_vector<double> q(2 * n, -1);

    thrust::device_vector<double> z_0(2 * n, 2);
    thrust::device_vector<double> z;

    double epsilon = 0.00001;
    double sigma = 0.75;
    double xi = 0.25;

    SOLVER_RESULT status = LCP_Newton(2 * n, M, q, z_0, epsilon, xi, sigma, z);
    // printf("SOLVER_RESULT: %d\n", (int) status);
    cudaDeviceSynchronize();
    if(status == SOLVE_SUCCESSFUL) {
        return 0;
    } else {
        return 1;
    }

}

int main() {
    int MAX_N = 64;
    int ITERS_PER_N = 10;

    unsigned int seed = 0;
    for (int n = 4; n <= MAX_N; n = n*2) {
        printf("Testing with n = %d\n", n);
        int bmg_failures = 0;
        for (int j = 0; j < ITERS_PER_N; j++) {
            bmg_failures += bimatrix_game(n, seed);
        }
        printf("Bimatrix Game Failures: %d / %d\n----\n", bmg_failures, ITERS_PER_N);
    }

    return 0;
}