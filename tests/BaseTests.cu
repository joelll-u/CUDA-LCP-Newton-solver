#include <gtest/gtest.h>

#include <thrust/device_vector.h>
#include <vector>
#include <cublas_v2.h>

#include "LCP_newton.hpp"

using namespace std;

class BaseTest : public testing::Test {
    protected:
    BaseTest() {
        vector<double> M_host = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0};
        vector<double> q_host = {1.0, 2.0, 3.0};
        vector<double> z_host = {1.0, 1.0, 1.0};

        M = thrust::device_vector<double>(M_host.begin(), M_host.end());
        q = thrust::device_vector<double>(q_host.begin(), q_host.end());
        z = thrust::device_vector<double>(z_host.begin(), z_host.end());

        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            printf("CUBLAS initialization failed: %d\n", status);
        }
    }

    thrust::device_vector<double> M;
    thrust::device_vector<double> q;
    thrust::device_vector<double> z;

    int n = 3;

    cublasHandle_t handle;
};

TEST_F(BaseTest, evaluates_linear) {
    thrust::device_vector<double> expected = {2.0, 3.0, 5.0};
    thrust::device_vector<double> res(n);
    eval_linear(n, M, q, z, res, handle);
    EXPECT_EQ(res, expected);

    int n = 4;
    thrust::device_vector<double> M = {1, 0, 0, 0, 2, 1, 0, 0, 2, 2, 1, 0, 2, 2, 2, 1};
    thrust::device_vector<double> q = {-1, -1, -1, -1};
    thrust::device_vector<double> z_0 = {1, 1, 1, 1};
    thrust::device_vector<double> res_2;
    eval_linear(n, M, q, z_0, res_2, handle);
    
    thrust::device_vector<double> expected_2 = {6, 4, 2, 0};
    EXPECT_EQ(res_2, expected_2);
}

TEST_F(BaseTest, evaluates_linear_sparse)
{
    int n = 4;
    thrust::device_vector<int> row_offset = {0, 2, 4, 5, 7};
    thrust::device_vector<int> col_index = {0, 1, 0, 1, 2, 1, 3};
    thrust::device_vector<double> data = {1, 2, 2, 1, 2, 2, 3};
    matrix_sparse M = {row_offset, col_index, data};

    matrix_sparse old_M = M;

    thrust::device_vector<double> q_old = {-1,-1,-1,-1};
    thrust::device_vector<double> q = {-1, -1, -1, -1};
    thrust::device_vector<double> z_0 = {1, 1, 1, 1};
    thrust::device_vector<double> z_old = {1,1,1,1};
    thrust::device_vector<double> res;

    cusparseHandle_t handle;
    cusparseCreate(&handle);
    eval_linear_sparse(n, M, q, z_0, res, handle);
    thrust::device_vector<double> expected = {2, 2, 1, 4};
    EXPECT_EQ(res, expected);
    EXPECT_EQ(M.column_indices, old_M.column_indices);
    EXPECT_EQ(M.row_offsets, old_M.row_offsets);
    EXPECT_EQ(M.values, old_M.values);
    EXPECT_EQ(q, q_old);
    EXPECT_EQ(z_0, z_old);
    

}

TEST_F(BaseTest, terminates_correctly) {
    thrust::device_vector<double> terminates = {0, 0, 0.001};
    thrust::device_vector<double> terminates_w = {1.0, 2.0, 3.002};
    thrust::device_vector<double> w = {2.0, 3.0, 5.0};
    
    EXPECT_TRUE(norm_termination_test(n, terminates, terminates_w, 0.01, handle));
    EXPECT_FALSE(norm_termination_test(n, z, w, 0.01, handle));
}

TEST_F(BaseTest, elementwise_min) {
    int n = 4;
    thrust::device_vector<double> v1 = {1, 2, 3, 5};
    thrust::device_vector<double> v2 = {0, 3, -5, 5};
    thrust::device_vector<double> res;

    thrust::device_vector<double> expected = {0, 2, -5, 5};
    elementwise_min(n, v1, v2, res);
    EXPECT_EQ(res, expected);
    res = {};
    elementwise_min(n, v2, v1, res);
    EXPECT_EQ(res, expected);
}

TEST_F(BaseTest, gets_alpha_gamma) {
    int n = 5;
    thrust::device_vector<double> v1 = {1, 2, 3, 4, 5};
    thrust::device_vector<double> v2 = {0, 3, -5, 5, 8};
    thrust::device_vector<double> res;

    thrust::device_vector<int> expected_gamma = {1, 3, 4};
    thrust::device_vector<int> expected_alpha = {0, 2};

    thrust::device_vector<int> alpha;
    thrust::device_vector<int> gamma;

    alpha_set(n, v1, v2, alpha, gamma);
    EXPECT_EQ(alpha, expected_alpha);
    EXPECT_EQ(gamma, expected_gamma);
}

TEST_F(BaseTest, gets_submatrix) {
    int n = 5;
    thrust::device_vector<double> M(25);
    thrust::sequence(M.begin(), M.end(), 1);

    thrust::device_vector<int> alpha = {1, 3, 4};
    thrust::device_vector<double> M_alpha;

    thrust::device_vector<double> expected = {7, 9, 10, 17, 19, 20, 22, 24, 25};

    submatrix(n, M, alpha, M_alpha);
    EXPECT_EQ(M_alpha, expected);
}

TEST_F(BaseTest, gets_subvector) {
    int n = 5;
    thrust::device_vector<double> q = {1.0, 2.0, 3.0, 4.0, 5.0};
    thrust::device_vector<int> alpha = {1, 2, 4};
    thrust::device_vector<double> q_alpha;

    thrust::device_vector<double> expected = {2.0, 3.0, 5.0};

    subvector(n, q, alpha, q_alpha);
    EXPECT_EQ(q_alpha, expected);
}

TEST_F(BaseTest, solves_linear_system) {
    int n = 3;
    thrust::device_vector<double> A = {1, 6, 5, 1, -4, 2, 1, 5, 2};
    thrust::device_vector<double> b = {2, 31, 13};

    thrust::device_vector<double> expected = {3, -2, 1};

    dn_solver_params params;

    setup_solver(n, params);

    ASSERT_GT(params.device_buffer_size, 0);
    ASSERT_NE(params.device_buffer, nullptr);


    thrust::device_vector<double> res;

    int status = solve_dense_linear_system(
        n,
        A,
        b,
        res,
        params
    );
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(res[i], expected[i], 1e-5);
    }
    EXPECT_EQ(status, 0);
}

TEST_F(BaseTest, solves_when_singular) {
    n = 3;
    thrust::device_vector<double> A_2 = {1, 0, 0, 0, 0, 0, 0, 0, 1};
    thrust::device_vector<double> b_2 = {1, 1, 1};

    dn_solver_params params;

    setup_solver(n, params);

    thrust::device_vector<double> res(n);
    int status = solve_dense_linear_system(
        n,
        A_2,
        b_2,
        res,
        params);

    thrust::device_vector<double> expected = {0.9999, 1e4, 0.9999};
    cudaDeviceSynchronize();
    EXPECT_EQ(status, 0);
    for (int i = 0; i < n; i++)
    {
        EXPECT_NEAR(res[i], expected[i], 1e-6);
    }

}

TEST_F(BaseTest, allocates_u) {
    n = 6;
    thrust::device_vector<double> q_alpha = {1, 2, 3};
    thrust::device_vector<int> alpha = {1, 3, 4};

    thrust::device_vector<double> res;

    thrust::device_vector<double> expected = {0, 1, 0, 2, 3, 0};

    scatter_vector(n, q_alpha, alpha, res);

    EXPECT_EQ(res, expected);
}

TEST_F(BaseTest, termination_2) {
    int n = 3;
    thrust::device_vector<double> u_pos = {1,2,0};
    thrust::device_vector<double> phi_pos = {0, 0, 1};

    EXPECT_TRUE(solve_termination_test(n, u_pos, phi_pos, 0.01, handle));
    
    thrust::device_vector<double> u_neg = {-1, 2, 0};
    EXPECT_FALSE(solve_termination_test(n, u_neg, phi_pos, 0.01, handle));

    thrust::device_vector<double> phi_neg = {0, 0, -1};
    EXPECT_FALSE(solve_termination_test(n, u_pos, phi_neg, 0.01, handle));

    thrust::device_vector<double> incorrect_phi = {1, 0, 1};
    EXPECT_FALSE(solve_termination_test(n, u_pos, incorrect_phi, 0.01, handle));
}

TEST_F(BaseTest, get_rho) {
    int n = 7;
    thrust::device_vector<int> alpha = {0, 1, 2, 4, 6};
    thrust::device_vector<int> gamma = {3, 5};

    thrust::device_vector<double> u = {0, 1, -2, 3, -4, -5, 6};
    thrust::device_vector<double> z = {1, 2, 3, 4, 5, 6, 7};
    thrust::device_vector<double> w = {10, 20, 30, 40, 50, 60, 70};
    thrust::device_vector<double> phi = {5, 10, 15, 20, 25, 31, 35};

    thrust::device_vector<double> rhos;

    get_rhos(
        n,
        z,
        u,
        phi,
        w,
        gamma,
        alpha,
        rhos
    );

    thrust::device_vector<double> expected = {27.0/25, 45.0/41, 36.0/16, 54.0/23};

    EXPECT_EQ(rhos, expected);
}

// TEST_F(BaseTest, findsIter) {
//     int n = 4;
//     thrust::device_vector<double> M = {1, 0, 0, 0, 2, 1, 0, 0, 2, 2, 1, 0, 2, 2, 2, 1};
//     thrust::device_vector<double> q = {-1, -1, -1, -1};
//     thrust::device_vector<double> u = {-1, 1, 0, 0};
//     thrust::device_vector<double> z = {-0.125000, 0.507812, 0.437500, 0.191406}; 
//     double current_merit = 0.979597;
//     double sigma0 = 0.1;
//     double sigma1 = 0.75;
//     double xi = 0.25;

//     thrust::device_vector<double> rhos = {4.657143, -1.612245};
//     thrust::device_vector<double> res;
//     get_next_iter(n, z, M, q, u, rhos, handle, current_merit, xi, sigma0, sigma1, res);

//     EXPECT_NE(res, z);
//     printf("res: "); for(int i = 0; i < n; i++) {printf("%f ", (double) res[i]);}; printf("\n");
//     FAIL();
// }

TEST_F(BaseTest, solvesLCP) {
    int n = 4;
    thrust::device_vector<double> M = {1, 0, 0, 0, 2, 1, 0, 0, 2, 2, 1, 0, 2, 2, 2, 1};
    thrust::device_vector<double> q = {-1, -1, -1, -1};
    thrust::device_vector<double> z_0(n);
    
    thrust::device_vector<double> z(n);
    double epsilon = 0.00001;
    double sigma = 0.75;
    double xi = 0.25;
    int status = LCP_Newton(n, M, q, z_0, epsilon, xi, sigma, z);
    ASSERT_EQ(status, 0);

    thrust::device_vector<double> expected = {0.0, 0.0, 0.0, 1.0};
    cudaDeviceSynchronize();
    EXPECT_EQ(z, expected);
    // for (int i = 0; i < n; i++) {
    //     EXPECT_double_EQ(z[i], expected[i]);
    // }
}

TEST_F(BaseTest, gets_columns) {
    int n = 5;
    thrust::device_vector<double> M = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    thrust::device_vector<int> alpha = {1, 3, 4};
    thrust::device_vector<double> res;
    row_matrix(n, M, alpha, res);

    thrust::device_vector<double> expected = {
        2, 4, 5,
        7, 9, 10,
        12, 14, 15,
        17, 19, 20,
        22, 24, 25
    };
    EXPECT_EQ(res, expected);
}

TEST_F(BaseTest, gets_gradient) {
    int n = 4;

    thrust::device_vector<double> M = {
        1, 0, 0, 0,
        2, 1, 0, 0,
        2, 2, 1, 0,
        2, 2, 2, 1,
    };
    thrust::device_vector<double> z = {0, 2, 1, 0};
    thrust::device_vector<double> w = {5, 3, 0, -1};
    thrust::device_vector<int> alpha = {2, 3};
    thrust::device_vector<int> gamma = {0, 1};

    thrust::device_vector<double> res;

    gradient(n, z, w, alpha, gamma, M, handle, res);

    thrust::device_vector<double> expected = {0, 2, 0, -1};
    EXPECT_EQ(res, expected);
}

TEST_F(BaseTest, gets_gradient2) {
    int n = 5;

    thrust::device_vector<double> M = {
        0, 0, 0, 4, 3,
        0, 0, 0, 3, 7,
        0, 0, 0, 4, 2,
        4, 3, 1, 0, 0,
        4, 6, 7, 0, 0};
    thrust::device_vector<double> z = {4.0/19, 1.0/19, 0, 0.080000, 0.170000};
    thrust::device_vector<double> w = {0.000000, 0.260000, 0.270000, 0, 0};
    thrust::device_vector<int> alpha = {0, 3, 4};
    thrust::device_vector<int> gamma = {1, 2};

    thrust::device_vector<double> res;

    gradient(n, z, w, alpha, gamma, M, handle, res);

    thrust::device_vector<double> expected = {0,1.0/19,0,0,0};
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++)
    {
        EXPECT_NEAR(res[i], expected[i], 1e-5);
    }
}

TEST_F(BaseTest, solves_bimatrix_game) {
    int n = 5;

    thrust::device_vector<double> M = {
                        0, 0, 0, 4, 3,
                        0, 0, 0, 3, 7,
                        0, 0, 0, 4, 2,
                        4, 3, 1, 0, 0,
                        4, 6, 7, 0, 0};
    thrust::device_vector<double> q = {-1, -1, -1, -1, -1};
    thrust::device_vector<double> z_0(5, 0.5);
    // thrust::device_vector<double> z_0 = {0.333333, 0, 0, 0, 0.25};
    thrust::device_vector<double> z(5);

    double epsilon = 0.00001;
    double sigma = 0.75;
    double xi = 0.25;
    int status = LCP_Newton(n, M, q, z_0, epsilon, xi, sigma, z);

    ASSERT_EQ(status, 0);
    thrust::device_vector<double> expected = {0.333333, 0, 0, 0, 0.25};
    for (int i = 0; i < n; i++)
    {
        EXPECT_NEAR(z[i], expected[i], 1e-5);
    }
}

TEST_F(BaseTest, gets_sparse_submatrix) {
    int n = 4;
    thrust::device_vector<int> rows = {0, 2, 3, 4, 6};
    thrust::device_vector<int> col_offsets = {0, 3, 1, 2, 0, 3};
    thrust::device_vector<double> values = {1.2, 5, 2, 3, 5, 4};
    matrix_sparse x = {rows, col_offsets, values};

    thrust::device_vector<int> alpha = {0, 1, 3};
    matrix_sparse res;
    sparse_submatrix(n, x, alpha, res);

    thrust::device_vector<int> expected_rows = {0, 2, 3, 5};
    thrust::device_vector<int> expected_col_offsets = {0, 2, 1, 0, 2};
    thrust::device_vector<double> expected_values = {1.2, 5, 2, 5, 4};

    EXPECT_EQ(res.row_offsets, expected_rows);
    EXPECT_EQ(res.column_indices, expected_col_offsets);
    EXPECT_EQ(res.values, expected_values);
}

TEST_F(BaseTest, solves_sparse_linear_system) {
    int n = 4;
    thrust::device_vector<int> rows = {0, 2, 3, 4, 6};
    thrust::device_vector<int> col_offsets = {0, 3, 1, 2, 0, 3};
    thrust::device_vector<double> values = {1, 5, 2, 3, 5, 4};
    matrix_sparse x = {rows, col_offsets, values};

    thrust::device_vector<double> b = {21, 4, 6, 42};

    thrust::device_vector<double> res;

    sparse_solve(n, x, b, res);
    cudaDeviceSynchronize();
    thrust::device_vector<double> expected = {6, 2, 2, 3};

    EXPECT_EQ(res, expected);
}

TEST_F(BaseTest, solves_sparse_LCP) {
    int n = 4;
    thrust::device_vector<double> q = {-1, -1, -1, 1};
    thrust::device_vector<double> z_0(n);
    thrust::device_vector<double> z(5);

    thrust::device_vector<int> rows = {0, 2, 5, 8, 11};
    thrust::device_vector<int> col_offset = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    thrust::device_vector<double> values = {2, 1, 1, 2, 1, 1, 2, 1, 1, 2};

    matrix_sparse f = {rows, col_offset, values};

    double epsilon = 0.00001;
    double sigma = 0.75;
    double xi = 0.25;
    int status = LCP_Newton(n, f, q, z_0, epsilon, xi, sigma, z);
    ASSERT_EQ(status, 0);

    thrust::device_vector<double> expected = {0.5, 0, 0.5, 0};

    cudaDeviceSynchronize();
    EXPECT_EQ(z, expected);
}