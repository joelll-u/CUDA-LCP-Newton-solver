
#include "LCP_newton.hpp"
#include "degeneracy_resolve.cu"
#include "sparse.cu"
#include <nvtx3/nvToolsExt.h>

using namespace std;

struct is_negative
{
    const double *u;
    is_negative(const double *_u) : u(_u) {}

    __device__ bool operator()(int i)
    {
        return u[i] < 0;
    }
};

struct is_less_than
{
    thrust::device_ptr<const double> a;
    thrust::device_ptr<const double> b;

    is_less_than(thrust::device_ptr<const double> a_, thrust::device_ptr<const double> b_)
        : a(a_), b(b_) {}

    __host__ __device__ bool operator()(int i) const
    {
        return a[i] > b[i];
    }
};

struct rho_i
{
    const double *z;
    const double *w;
    const double *u;

    rho_i(const double *_u, const double *_w, const double *_z) : u(_u), w(_w), z(_z) {}

    __device__ double operator()(int i)
    {
        return (z[i] - w[i]) / ((z[i] - w[i]) - u[i]);
    }
};

struct rho_j
{
    const double *z;
    const double *w;
    const double *phi;

    rho_j(const double *_z, const double *_w, const double *_phi) : z(_z), w(_w), phi(_phi) {}

    __device__ double operator()(int i)
    {
        return (w[i] - z[i]) / (w[i] - z[i] - phi[i]);
    }
};

double get_merit(int N, thrust::device_vector<double> &z, thrust::device_vector<double> &w, cublasHandle_t &handle)
{
    thrust::device_vector<double> res(N);

    elementwise_min(N, z, w, res);

    double nrm = 0;
    cublasDnrm2(handle, N, res.data().get(), 1, &nrm);
    return nrm;
}

void subvector(int N, thrust::device_vector<double> &q, thrust::device_vector<int> &alpha, thrust::device_vector<double> &q_alpha)
{
    int k = alpha.size();
    q_alpha.resize(k);

    thrust::gather(alpha.begin(), alpha.end(), q.begin(), q_alpha.begin());
}

void submatrix(int N, thrust::device_vector<double> &M, thrust::device_vector<int> &alpha, thrust::device_vector<double> &M_alpha)
{
    int k = alpha.size();
    M_alpha.resize(k * k);

    // Allocate gather indices and intermediate vectors
    thrust::device_vector<int> gather_indices(k * k);

    const int *alpha_ptr = thrust::raw_pointer_cast(alpha.data());

    auto ff = [=] __host__ __device__ (int idx) {
            int col = alpha_ptr[idx / k];  
            int row = alpha_ptr[idx % k];
            return col * N + row;
        };
    // Lambda to generate gather indices: row = a[i], col = a[j], index = row * N + col
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(k * k),
        gather_indices.begin(),
        ff
    );

    thrust::gather(gather_indices.begin(), gather_indices.end(), M.begin(), M_alpha.begin());
}

void alpha_set(int N, thrust::device_vector<double> &z1, thrust::device_vector<double> &z2, thrust::device_vector<int> &alpha, thrust::device_vector<int> &gamma)
{
    alpha.resize(N);
    gamma.resize(N);
    auto ends = thrust::stable_partition_copy(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(N),
        alpha.begin(),
        gamma.begin(),
        is_less_than(z1.data(),z2.data())
    );
    alpha.resize(ends.first - alpha.begin());
    gamma.resize(ends.second - gamma.begin());
}

void elementwise_min(int N, thrust::device_vector<double> &z1, thrust::device_vector<double> &z2, thrust::device_vector<double> &res)
{
    res.resize(N);
    thrust::transform(z1.begin(), z1.end(), z2.begin(), res.begin(), thrust::minimum<double>());
}

void eval_linear(int N, thrust::device_vector<double> &M, thrust::device_vector<double> &q, thrust::device_vector<double> &z, thrust::device_vector<double> &res, cublasHandle_t &handle)
{
        res.resize(N);
        double alpha = 1.0;
        double beta = 1.0;
        thrust::copy(q.begin(), q.end(), res.begin());
        // printf("res: "); for(int i = 0; i < N; i++) {printf("%f ", (double) res[i]);}; printf("\n");
        // printf("z: "); for(int i = 0; i < N; i++) {printf("%f ", (double) z[i]);}; printf("\n");
        cublasDgemv(handle, CUBLAS_OP_N, N, N, &alpha, thrust::raw_pointer_cast(M.data()), N, thrust::raw_pointer_cast(z.data()), 1, &beta, thrust::raw_pointer_cast(res.data()), 1);

    return;
}

void eval_linear_sparse(int N, matrix_sparse &M, thrust::device_vector<double> &q, thrust::device_vector<double> &z, thrust::device_vector<double> &res, cusparseHandle_t &handle) 
{
    // Step 1: Copy q into res
    res.resize(N);
    thrust::copy(q.begin(), q.end(), res.begin());

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecZ, vecRes;

    cusparseCreateCsr(&matA, N, N, M.values.size(),
                      thrust::raw_pointer_cast(M.row_offsets.data()),
                      thrust::raw_pointer_cast(M.column_indices.data()),
                      thrust::raw_pointer_cast(M.values.data()),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateDnVec(&vecZ, N, thrust::raw_pointer_cast(z.data()), CUDA_R_64F);
    cusparseCreateDnVec(&vecRes, N, thrust::raw_pointer_cast(res.data()), CUDA_R_64F);

    // Step 3: Perform SpMV: res = M*z + res
    const double alpha = 1.0;
    const double beta = 1.0;

    size_t bufferSize = 0;
    void *dBuffer = nullptr;

    cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        vecZ,
        &beta,
        vecRes,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize);

    cudaMalloc(&dBuffer, bufferSize);

    cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        vecZ,
        &beta,
        vecRes,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        dBuffer);

    // Cleanup
    cudaFree(dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecZ);
    cusparseDestroyDnVec(vecRes);
}

bool norm_termination_test(int N, thrust::device_vector<double> &z, thrust::device_vector<double> &w, double epsilon, cublasHandle_t &handle)
{
    double nrm = get_merit(N, z, w, handle);
    return nrm <= epsilon;
}

int solve_dense_linear_system(int N, thrust::device_vector<double> &A, thrust::device_vector<double> &b, thrust::device_vector<double> &res, dn_solver_params params)
{
    res.resize(N);
    cudaDataType_t type = CUDA_R_64F;

    double *A_cpy;
    cudaMalloc(&A_cpy, sizeof(double) * A.size());
    cudaMemcpy(A_cpy, A.data().get(), sizeof(double) * A.size(), cudaMemcpyDeviceToDevice);

    int64_t *d_ipiv;
    cudaMalloc(&d_ipiv, sizeof(int64_t) * N);

    int *d_info;
    cudaMalloc(&d_info, sizeof(int));

    thrust::copy(b.begin(), b.end(), res.begin());

    cusolverDnXgetrf(
        params.handle,
        params.params,
        N,
        N,
        type,
        A_cpy,
        N,
        d_ipiv,
        type,
        params.device_buffer,
        params.device_buffer_size,
        params.host_buffer,
        params.host_buffer_size,
        d_info
    );

    int info;
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("%d\n", info);
    if (info > 0) {
        // return 1;
        for (int i = 0; i < N; i++) {
            A[i*N + i] += 1e-4;
        }
        return solve_dense_linear_system(N, A, b, res, params);
    }

    cusolverDnXgetrs(
        params.handle,
        params.params,
        CUBLAS_OP_N,
        N,
        1,
        type,
        A_cpy,
        N,
        d_ipiv,
        type,
        (void *) res.data().get(),
        N,
        d_info
    );

    cudaFree(A_cpy);
    cudaFree(d_ipiv);
    cudaFree(d_info);
    return 0;
}

void setup_solver(int N, dn_solver_params &params_s)
{
    cusolverDnCreate(&params_s.handle);
    cusolverDnCreateParams(&params_s.params);
    cusolverDnSetAdvOptions(params_s.params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0);

    cudaDataType_t type = CUDA_R_64F;

    cusolverDnXgetrf_bufferSize(
        params_s.handle,
        params_s.params,
        N,
        N,
        type,
        nullptr,
        N,
        type,
        &params_s.device_buffer_size,
        &params_s.host_buffer_size
    );

    if (params_s.host_buffer_size > 0) {
        params_s.host_buffer = malloc(params_s.host_buffer_size);
    } else {
        params_s.host_buffer = nullptr;
    }

    cudaMalloc(&params_s.device_buffer, params_s.device_buffer_size);

    return;
}

void scatter_vector(int N, thrust::device_vector<double> &u_alpha, thrust::device_vector<int> &alpha, thrust::device_vector<double> &res) {
    res.resize(N);
    thrust::fill(res.begin(), res.end(), 0);
    thrust::scatter(u_alpha.begin(), u_alpha.end(), alpha.begin(), res.begin());

    return;
}

bool solve_termination_test(int N, thrust::device_vector<double> &u, thrust::device_vector<double> &phi, double epsilon, cublasHandle_t &handle) {
    auto negative = [] __device__ (double x) {return x < 0;};

    bool u_negative = thrust::any_of(
        u.begin(),
        u.end(),
        negative
    );

    bool phi_negative = thrust::any_of(
        phi.begin(),
        phi.end(),
        negative
    );

    if (u_negative || phi_negative) return false;

    return norm_termination_test(N, u, phi, epsilon, handle);
}

void get_rhos(int N, thrust::device_vector<double> &z, thrust::device_vector<double> &u, thrust::device_vector<double> &phi, thrust::device_vector<double> &w, thrust::device_vector<int> &gamma, thrust::device_vector<int> &alpha, thrust::device_vector<double> &rhos)
{
    thrust::device_vector<int> is(alpha.size());
    auto final = thrust::copy_if(
        alpha.begin(),
        alpha.end(),
        is.begin(),
        is_negative(thrust::raw_pointer_cast(u.data()))
    );
    is.resize(final - is.begin());

    rhos.resize(is.size() + gamma.size());

    thrust::transform(
        is.begin(),
        is.end(),
        rhos.begin(),
        rho_i(
            thrust::raw_pointer_cast(u.data()),
            thrust::raw_pointer_cast(w.data()),
            thrust::raw_pointer_cast(z.data())
        )
    );

    thrust::transform(
        gamma.begin(),
        gamma.end(),
        rhos.begin() + is.size(),
        rho_j(
            thrust::raw_pointer_cast(z.data()),
            thrust::raw_pointer_cast(w.data()),
            thrust::raw_pointer_cast(phi.data())
        )
    );
}

int get_next_iter(int N, thrust::device_vector<double> &z, thrust::device_vector<double> &w, thrust::device_vector<double> &q, thrust::device_vector<double> &u, thrust::device_vector<double> &phi, thrust::device_vector<double> &rhos, cublasHandle_t &handle, double current_merit, double xi, double sigma, thrust::device_vector<double> &res, thrust::device_vector<double> &wv)
{
    res.resize(N);
    wv.resize(N);
    double tau = 1.0;
    for (int i = 0; i < 100; i++){
        // printf("%d\n", i);
        thrust::copy(z.begin(), z.end(), res.begin());
        thrust::copy(w.begin(), w.end(), wv.begin());
        auto idx = thrust::find(rhos.begin(), rhos.end(), tau);
        if (idx != rhos.end()) {
            tau *= sigma;
            continue;
        
        }
        double x = 1 - tau;
        cublasDscal(handle, N, &x, res.data().get(), 1);
        cublasDaxpy(handle, N, &tau, u.data().get(), 1, res.data().get(), 1);

        cublasDscal(handle, N, &x, wv.data().get(), 1);
        cublasDaxpy(handle, N, &tau, phi.data().get(), 1, wv.data().get(), 1);
        double merit_zv = get_merit(N, res, wv, handle);
        // printf("new merit: %f\n", merit_zv);
        if (current_merit - (tau*xi*current_merit) > merit_zv) {
            // printf("tau: %f\n", tau);
            return 0;
        }
        tau *= sigma;
    }
    return 3;
    // printf("xxxx\n");
}

SOLVER_RESULT LCP_Newton(int N, bool sparse, matrix_sparse &f, matrix_dense &M, thrust::device_vector<double> &q, thrust::device_vector<double> &z0, double epsilon, double xi, double sigma, thrust::device_vector<double> &res) 
{
    int max_iters = 100;
    res.resize(N);
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS initialization failed: %d\n", status);
        return UNKNOWN_ERROR;
    }

    cusparseHandle_t sparse_handle;
    if (sparse) {
        cusparseCreate(&sparse_handle);
    }
    // printf("M:\n");
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%f ", (double) M[j * N + i]);
    //     }
    //     printf("\n");
    // }
    // printf("q:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%f ", (double) q[i]);
    // }
    // printf("\n");

    dn_solver_params dn_params;
    if (!sparse) {
        setup_solver(N, dn_params);
    }


    thrust::device_vector<double> z_v(N);
    thrust::copy(z0.begin(), z0.end(), z_v.begin());

    thrust::device_vector<double> w;
    w.reserve(N);

    thrust::device_vector<double> u;
    u.reserve(N);

    thrust::device_vector<double> phi;
    phi.reserve(N);

    thrust::device_vector<double> old_z;
    old_z.reserve(N);

    thrust::device_vector<int> alpha;
    alpha.reserve(N);

    thrust::device_vector<int> gamma;
    gamma.reserve(N);
    
    thrust::device_vector<double> M_alpha;
    if (!sparse) {
        M_alpha.reserve(N*N);
    }

    thrust::device_vector<double> q_alpha;
    q_alpha.reserve(N);

    nvtxRangePushA("eval_linear");
    if(!sparse)
    {
        eval_linear(N, M, q, z_v, w, handle);
    } else {
        eval_linear_sparse(N, f, q, z_v, w, sparse_handle);
    }
    nvtxRangePop();
    for (int v = 0; v < max_iters; v++)
    {
        // printf("z: "); for(int i = 0; i < N; i++) {printf("%f ", (double) z_v[i]);}; printf("\n");
        // 1.check for termination

        // printf("w: "); for(int i = 0; i < N; i++) {printf("%f ", (double) w[i]);}; printf("\n");
        nvtxRangePushA("get_merit");
        double merit = get_merit(N, z_v, w, handle);
        nvtxRangePop();

        printf("merit: %f\n", merit);
        if (merit < epsilon)
        {
            thrust::copy(z_v.begin(), z_v.end(), res.begin());
            return SOLVE_SUCCESSFUL;
        }

        //2.1 compute alpha and gamma

        nvtxRangePushA("alpha_set");
        alpha_set(N, z_v, w, alpha, gamma);
        nvtxRangePop();
        // 2.2 compute u and phi

        matrix_sparse M_alpha_f;
        nvtxRangePushA("submatrix");
        if (sparse) {
            sparse_submatrix(N, f, alpha, M_alpha_f);
        } else {
            submatrix(N, M, alpha, M_alpha);
        }
        nvtxRangePop();

        nvtxRangePushA("subvector");
        subvector(N, q, alpha, q_alpha);
        nvtxRangePop();

        // printf("alpha: "); for (int i = 0; i < alpha.size(); i++) {printf("%d ", (int) alpha[i]);}; printf("\n");
        // printf("gamma: "); for (int i = 0; i < gamma.size(); i++) {printf("%d ", (int) gamma[i]);}; printf("\n");
        printf("M_alpha:\n");
        printf("%d %d %d %d\n", M_alpha_f.row_offsets.size(), M_alpha_f.column_indices.size(), M_alpha_f.values.size(), q_alpha.size());
        for (int i = 0; i < M_alpha_f.row_offsets.size(); i++) {
            printf("%d ", (int) M_alpha_f.row_offsets[i]);
        }
        cout <<endl << endl;
        for (int i = 0; i < M_alpha_f.column_indices.size(); i++)
        {
            printf("%d ", (int)M_alpha_f.column_indices[i]);
        }
        cout << endl << endl;
        for (int i = 0; i < M_alpha_f.values.size(); i++) {
            cout << (double) M_alpha_f.values[i] << " ";
        }
        cout << endl << endl;
        // printf("q:\n");
        // for (int i = 0; i < alpha.size(); i++)
        // {
        //     printf("%f ", (double)q_alpha[i]);
        // }
        printf("\n");

        nvtxRangePushA("solve");
        thrust::device_vector<double> u_alpha;
        if (!sparse) {
            int solve_status = solve_dense_linear_system(
            alpha.size(),
            M_alpha, 
            q_alpha, 
            u_alpha, 
            dn_params);

            if (solve_status != 0) {
                // printf("HERE!\n");
                // printf("grad: ");
                thrust::device_vector<double> grad;
                gradient(N, z_v, w, alpha, gamma, M, handle, grad);
                // for (int i = 0; i < alpha.size(); i++) {
                //     printf("%f ", (double) u_alpha[i]);
                // }
                // printf("\n");
                gradient_step(N, -1, z_v, w, alpha, gamma, M, q, handle);
                // printf("z: "); for(int i = 0; i < N; i++) {printf("%f ", (double) z_v[i]);}; printf("\n");
                continue;
                
                // return DEGENERACY_ENCOUNTERED;
            }
        } else {
            int solve_status = sparse_solve(alpha.size(), M_alpha_f, q_alpha, u_alpha);
        }
        nvtxRangePop();
        nvtxRangePushA("get_u");
        ::cuda::std::negate<double> minus;
        printf("here! %d %d\n", alpha.size(), u_alpha.size());
        printf("u_alpha0: %f\n", (double) u_alpha[0]);
        // printf("u_alpha: "); for(int i = 0; i < 1; i++) {printf("%f ", (double) u_alpha[i]);}; printf("\n");
        thrust::transform(u_alpha.begin(), u_alpha.end(), u_alpha.begin(), minus);
        scatter_vector(N, u_alpha, alpha, u);
        nvtxRangePop();

        printf("u: "); for(int i = 0; i < N; i++) {printf("%f ", (double) u[i]);}; printf("\n");
        nvtxRangePushA("eval_linear");
        if (!sparse) {
            eval_linear(N, M, q, u, phi, handle);
        } else {
            eval_linear_sparse(N, f, q, u, phi, sparse_handle);
        }
        nvtxRangePop();
        // printf("phi: "); for(int i = 0; i < N; i++) {printf("%f ", (double) phi[i]);}; printf("\n");
         // 3. check for termination
        if (solve_termination_test(N, u, phi, epsilon, handle))
        {
            // printf("HERE! %d\n", (int) u.size());
            thrust::copy(u.begin(), u.end(), res.begin());
            return SOLVE_SUCCESSFUL;
        }
        // 4.1 find rhos
        nvtxRangePushA("get_rhos");
        thrust::device_vector<double> rhos;
        get_rhos(N, z_v, u, phi, w, gamma, alpha, rhos);
        nvtxRangePop();
        // printf("rho: "); for(int i = 0; i < rhos.size(); i++) {printf("%f ", (double) rhos[i]);}; printf("\n");
        // 4.2 calculate z+1
        nvtxRangePushA("get_next_iter");
        thrust::device_vector<double> new_z(N);
        thrust::device_vector<double> new_w(N);

        int status = get_next_iter(N, z_v, w, q, u, phi, rhos, handle, merit, xi, sigma, new_z, new_w);
        if (status != 0)
        {
            if (sparse)
                throw std::invalid_argument("Sparse non degenerate matrices are not allowed");
            // thrust::device_vector<double> grad;
            // gradient(N, z_v, w, alpha, gamma, M, handle, grad);
            // for (int i = 0; i < alpha.size(); i++) {
            //     printf("%f ", (double) u_alpha[i]);
            // }
            // printf("\n");
            int status = gradient_step(N, -1e-2, z_v, w, alpha, gamma, M, q, handle);
            if (status == 1)
            {
                return STATIONARY_POINT_FOUND;
            }
             // printf("z: "); for(int i = 0; i < N; i++) {printf("%f ", (double) z_v[i]);}; printf("\n");
            continue;

            // return DEGENERACY_ENCOUNTERED;
            return (SOLVER_RESULT)status;
         }
        // printf("-----------\n");

        thrust::copy(new_z.begin(), new_z.end(), z_v.begin());
        thrust::copy(new_w.begin(), new_w.end(), w.begin());
        nvtxRangePop();
        // z_v = new_z;
        // w = new_w;
    }
    return TERMINATION_LIMIT;
}

SOLVER_RESULT LCP_Newton(int N, matrix_dense &M, thrust::device_vector<double> &q, thrust::device_vector<double> &z0, double epsilon, double xi, double sigma, thrust::device_vector<double> &res) {
    matrix_sparse m_empty;
    return LCP_Newton(N, false, m_empty, M, q, z0, epsilon, xi, sigma, res);
}

SOLVER_RESULT LCP_Newton(int N, matrix_sparse &M, thrust::device_vector<double> &q, thrust::device_vector<double> &z0, double epsilon, double xi, double sigma, thrust::device_vector<double> &res) {
    matrix_dense m_empty;
    return LCP_Newton(N, true, M, m_empty, q, z0, epsilon, xi, sigma, res);
}