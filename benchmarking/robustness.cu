#include "benchmarking_utils.cu"
#include "LCP_newton.hpp"
#include "path_utils.cpp"

int bimatrix_game(int n, unsigned int &seed) {
    std::vector<double> A;
    std::vector<double> B;
    create_random_vector(n*n, 1, 5, seed, A);
    create_random_vector(n*n, 1, 5, seed, B);

    std::vector<double> M_host(4 * n * n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
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
    cudaDeviceSynchronize();
    if(status == SOLVE_SUCCESSFUL) {
        return 0;
    } else {
        return 1;
    }

}

void create_degen_matrix(int n, unsigned int &seed, double sparsity, std::vector<double> &M, std::vector<double> &q) {
    create_random_P_matrix(n, -5, 5, seed, M);
    create_random_vector(n, -500, 500, seed, q);

    int num_deleted = sparsity * (n*n);
    int diag_deleted = sparsity * n;

    delete_random_non_diag(n, num_deleted, seed, M);
    delete_random_diag(n, diag_deleted, seed, M);
}

int main() {
    int MAX_N = 64;
    int ITERS_PER_N = 10;
    int degen_n = 1000;

    unsigned int seed = 0;
    for (int n = 4; n <= MAX_N; n = n*2) {
        printf("Testing with n = %d\n", n);
        int bmg_failures = 0;
        for (int j = 0; j < ITERS_PER_N; j++) {
            bmg_failures += bimatrix_game(n, seed);
        }
        printf("Bimatrix Game Failures: %d / %d\n----\n", bmg_failures, ITERS_PER_N);
    }
    for (double i = 0; i < 0.0099; i+=0.001) {
        int feasible_count = 0;
        int solved_count = 0;
        for (int j = 0; j < ITERS_PER_N; j++) {
            std::vector<double> M;
            std::vector<double> q;
            create_degen_matrix(degen_n, seed, i, M, q);

            int path_solved = path_simple_solve(degen_n, M, q, true);
            feasible_count += path_solved;

            thrust::device_vector<double> M_d = M;
            thrust::device_vector<double> q_d = q;

            thrust::device_vector<double> z_0(degen_n);
            thrust::device_vector<double> z;

            double epsilon = 0.000001;
            double sigma = 0.75;
            double xi = 0.25;

            SOLVER_RESULT status = LCP_Newton(degen_n, M_d, q_d, z_0, epsilon, xi, sigma, z);
            
            cudaDeviceSynchronize();
            if (status == SOLVE_SUCCESSFUL && path_solved == 1) {
                solved_count++;
            } else if (status == SOLVE_SUCCESSFUL) {
                solved_count++;
                printf("Somehow better than path? \n");
            }

        }
        printf("Solved %d/%d with sparsity %f and n=%d\n", solved_count, feasible_count, i, degen_n);
    }
    return 0;
}