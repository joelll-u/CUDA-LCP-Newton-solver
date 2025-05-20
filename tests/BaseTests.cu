#include <gtest/gtest.h>

#include <thrust/device_vector.h>
#include <vector>
#include <cublas_v2.h>

#include "LCP_newton.hpp"

using namespace std;

class BaseTest : public testing::Test {
    protected:
    BaseTest() {
        vector<float> M_host = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0};
        vector<float> q_host = {1.0, 2.0, 3.0};
        vector<float> z_host = {1.0, 1.0, 1.0};

        M = thrust::device_vector<float>(M_host.begin(), M_host.end());
        q = thrust::device_vector<float>(q_host.begin(), q_host.end());
        z = thrust::device_vector<float>(z_host.begin(), z_host.end());

        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            printf("CUBLAS initialization failed: %d\n", status);
        }
    }

    thrust::device_vector<float> M;
    thrust::device_vector<float> q;
    thrust::device_vector<float> z;

    int n = 3;

    cublasHandle_t handle;
};

TEST_F(BaseTest, evaluates_linear) {
    thrust::device_vector<float> expected = {2.0, 3.0, 5.0};
    thrust::device_vector<float> res(n);
    eval_linear(n, M, q, z, res, handle);
    EXPECT_EQ(res, expected);

    int n = 4;
    thrust::device_vector<float> M = {1, 0, 0, 0, 2, 1, 0, 0, 2, 2, 1, 0, 2, 2, 2, 1};
    thrust::device_vector<float> q = {-1, -1, -1, -1};
    thrust::device_vector<float> z_0 = {1, 1, 1, 1};
    thrust::device_vector<float> res_2;
    eval_linear(n, M, q, z_0, res_2, handle);
    
    thrust::device_vector<float> expected_2 = {6, 4, 2, 0};
    EXPECT_EQ(res_2, expected_2);
}

TEST_F(BaseTest, terminates_correctly) {
    thrust::device_vector<float> terminates = {0, 0, 0.001};
    thrust::device_vector<float> terminates_w = {1.0, 2.0, 3.002};
    thrust::device_vector<float> w = {2.0, 3.0, 5.0};
    
    EXPECT_TRUE(norm_termination_test(n, terminates, terminates_w, 0.01, handle));
    EXPECT_FALSE(norm_termination_test(n, z, w, 0.01, handle));
}

TEST_F(BaseTest, elementwise_min) {
    int n = 4;
    thrust::device_vector<float> v1 = {1, 2, 3, 5};
    thrust::device_vector<float> v2 = {0, 3, -5, 5};
    thrust::device_vector<float> res;

    thrust::device_vector<float> expected = {0, 2, -5, 5};
    elementwise_min(n, v1, v2, res);
    EXPECT_EQ(res, expected);
    res = {};
    elementwise_min(n, v2, v1, res);
    EXPECT_EQ(res, expected);
}

TEST_F(BaseTest, gets_alpha_gamma) {
    int n = 5;
    thrust::device_vector<float> v1 = {1, 2, 3, 4, 5};
    thrust::device_vector<float> v2 = {0, 3, -5, 5, 8};
    thrust::device_vector<float> res;

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
    thrust::device_vector<float> M(25);
    thrust::sequence(M.begin(), M.end(), 1);

    thrust::device_vector<int> alpha = {1, 3, 4};
    thrust::device_vector<float> M_alpha;

    thrust::device_vector<float> expected = {7, 9, 10, 17, 19, 20, 22, 24, 25};

    submatrix(n, M, alpha, M_alpha);
    EXPECT_EQ(M_alpha, expected);
}

TEST_F(BaseTest, gets_subvector) {
    int n = 5;
    thrust::device_vector<float> q = {1.0, 2.0, 3.0, 4.0, 5.0};
    thrust::device_vector<int> alpha = {1, 2, 4};
    thrust::device_vector<float> q_alpha;

    thrust::device_vector<float> expected = {2.0, 3.0, 5.0};

    subvector(n, q, alpha, q_alpha);
    EXPECT_EQ(q_alpha, expected);
}

TEST_F(BaseTest, solves_linear_system) {
    int n = 3;
    thrust::device_vector<float> A = {1, 6, 5, 1, -4, 2, 1, 5, 2};
    thrust::device_vector<float> b = {2, 31, 13};

    thrust::device_vector<float> expected = {3, -2, 1};

    void* host_buffer = nullptr;
    void* device_buffer = nullptr;
    cusolverDnHandle_t handle;
    cusolverDnParams_t params;
    size_t host_size;
    size_t device_size;

    setup_solver(
        n,
        handle,
        host_buffer,
        host_size,
        device_buffer,
        device_size,
        params
    );

    ASSERT_GT(device_size, 0);
    ASSERT_NE(device_buffer, nullptr);

    thrust::device_vector<float> res;

    solve_linear_system(
        n,
        A,
        b,
        res,
        handle,
        params,
        host_buffer,
        host_size,
        device_buffer,
        device_size
    );
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) {
        EXPECT_FLOAT_EQ(res[i], expected[i]);
    }

    n = 2;
    thrust::device_vector<float> A_2 = {2, 1, -1, 1};
    thrust::device_vector<float> b_2 = {1, 5};

    solve_linear_system(
        n,
        A_2,
        b_2,
        res,
        handle,
        params,
        host_buffer,
        host_size,
        device_buffer,
        device_size);
    thrust::device_vector<float> expected_2 = {2, 3};
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++)
    {
        EXPECT_FLOAT_EQ(res[i], expected_2[i]);
    }
}

TEST_F(BaseTest, allocates_u) {
    n = 6;
    thrust::device_vector<float> q_alpha = {1, 2, 3};
    thrust::device_vector<int> alpha = {1, 3, 4};

    thrust::device_vector<float> res;

    thrust::device_vector<float> expected = {0, 1, 0, 2, 3, 0};

    scatter_vector(n, q_alpha, alpha, res);

    EXPECT_EQ(res, expected);
}

TEST_F(BaseTest, termination_2) {
    int n = 3;
    thrust::device_vector<float> u_pos = {1,2,0};
    thrust::device_vector<float> phi_pos = {0, 0, 1};

    EXPECT_TRUE(solve_termination_test(n, u_pos, phi_pos, 0.01, handle));
    
    thrust::device_vector<float> u_neg = {-1, 2, 0};
    EXPECT_FALSE(solve_termination_test(n, u_neg, phi_pos, 0.01, handle));

    thrust::device_vector<float> phi_neg = {0, 0, -1};
    EXPECT_FALSE(solve_termination_test(n, u_pos, phi_neg, 0.01, handle));

    thrust::device_vector<float> incorrect_phi = {1, 0, 1};
    EXPECT_FALSE(solve_termination_test(n, u_pos, incorrect_phi, 0.01, handle));
}

TEST_F(BaseTest, get_rho) {
    int n = 7;
    thrust::device_vector<int> alpha = {0, 1, 2, 4, 6};
    thrust::device_vector<int> gamma = {3, 5};

    thrust::device_vector<float> u = {0, 1, -2, 3, -4, -5, 6};
    thrust::device_vector<float> z = {1, 2, 3, 4, 5, 6, 7};
    thrust::device_vector<float> w = {10, 20, 30, 40, 50, 60, 70};
    thrust::device_vector<float> phi = {5, 10, 15, 20, 25, 31, 35};

    thrust::device_vector<float> rhos;

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

    thrust::device_vector<float> expected = {27.0/25, 45.0/41, 36.0/16, 54.0/23};

    EXPECT_EQ(rhos, expected);
}

// TEST_F(BaseTest, findsIter) {
//     int n = 4;
//     thrust::device_vector<float> M = {1, 0, 0, 0, 2, 1, 0, 0, 2, 2, 1, 0, 2, 2, 2, 1};
//     thrust::device_vector<float> q = {-1, -1, -1, -1};
//     thrust::device_vector<float> u = {-1, 1, 0, 0};
//     thrust::device_vector<float> z = {-0.125000, 0.507812, 0.437500, 0.191406}; 
//     float current_merit = 0.979597;
//     float sigma0 = 0.1;
//     float sigma1 = 0.75;
//     float xi = 0.25;

//     thrust::device_vector<float> rhos = {4.657143, -1.612245};
//     thrust::device_vector<float> res;
//     get_next_iter(n, z, M, q, u, rhos, handle, current_merit, xi, sigma0, sigma1, res);

//     EXPECT_NE(res, z);
//     printf("res: "); for(int i = 0; i < n; i++) {printf("%f ", (float) res[i]);}; printf("\n");
//     FAIL();
// }

TEST_F(BaseTest, solvesLCP) {
    int n = 4;
    thrust::device_vector<float> M = {1, 0, 0, 0, 2, 1, 0, 0, 2, 2, 1, 0, 2, 2, 2, 1};
    thrust::device_vector<float> q = {-1, -1, -1, -1};
    thrust::device_vector<float> z_0 = {1, 1, 1, 1};
    
    thrust::device_vector<float> z(n);
    float epsilon = 0.00001;
    float sigma0 = 0.1;
    float sigma1 = 0.75;
    float xi = 0.25;
    int status = LCP_Newton(n, M, q, z_0, epsilon, xi, sigma0, sigma1, z);
    ASSERT_EQ(status, 0);

    thrust::device_vector<float> expected = {0.0, 0.0, 0.0, 1.0};
    cudaDeviceSynchronize();
    EXPECT_EQ(z, expected);
    // for (int i = 0; i < n; i++) {
    //     EXPECT_FLOAT_EQ(z[i], expected[i]);
    // }
}