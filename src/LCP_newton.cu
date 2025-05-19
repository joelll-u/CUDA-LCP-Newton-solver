
#include "LCP_newton.hpp"

using namespace std;

float get_merit(int N, thrust::device_vector<float> &z, thrust::device_vector<float> &w, cublasHandle_t &handle)
{
    thrust::device_vector<float> res(N);

    elementwise_min(N, z, w, res);

    float nrm = 0;
    cublasSnrm2(handle, N, res.data().get(), 1, &nrm);
    return nrm;
}

void subvector(int N, thrust::device_vector<float> &q, thrust::device_vector<int> &alpha, thrust::device_vector<float> &q_alpha)
{
    int k = alpha.size();
    q_alpha.resize(k);

    thrust::gather(alpha.begin(), alpha.end(), q.begin(), q_alpha.begin());
}

void submatrix(int N, thrust::device_vector<float> &M, thrust::device_vector<int> &alpha, thrust::device_vector<float> &M_alpha)
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

void alpha_set(int N, thrust::device_vector<float> &z1, thrust::device_vector<float> &z2, thrust::device_vector<int> &alpha, thrust::device_vector<int> &gamma)
{
    thrust::stable_partition_copy(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(N),
        std::back_inserter(alpha),
        std::back_inserter(gamma),
        is_less_than(z1.data(),z2.data())
    );
}

void elementwise_min(int N, thrust::device_vector<float> &z1, thrust::device_vector<float> &z2, thrust::device_vector<float> &res)
{
    res.resize(N);
    thrust::transform(z1.begin(), z1.end(), z2.begin(), res.begin(), thrust::minimum<float>());
}

void eval_linear(int N, thrust::device_vector<float> &M, thrust::device_vector<float> &q, thrust::device_vector<float> &z, thrust::device_vector<float> &res, cublasHandle_t &handle)
{
    res.resize(N);
    float alpha = 1.0;
    float beta = 1.0;
    thrust::copy(q.begin(), q.end(), res.begin());
    // printf("res: "); for(int i = 0; i < N; i++) {printf("%f ", (float) res[i]);}; printf("\n");
    // printf("z: "); for(int i = 0; i < N; i++) {printf("%f ", (float) z[i]);}; printf("\n");
    cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, thrust::raw_pointer_cast(M.data()), N, thrust::raw_pointer_cast(z.data()), 1, &beta, thrust::raw_pointer_cast(res.data()), 1);

    return;
}

bool norm_termination_test(int N, thrust::device_vector<float> &z, thrust::device_vector<float> &w, float epsilon, cublasHandle_t &handle)
{
    float nrm = get_merit(N, z, w, handle);
    return nrm <= epsilon;
}

void solve_linear_system(int N, thrust::device_vector<float> &A, thrust::device_vector<float> &b, thrust::device_vector<float> &res, cusolverDnHandle_t &handle, cusolverDnParams_t &params, void *host_buffer, size_t host_buffer_size, void *device_buffer, size_t device_buffer_size)
{
    res.resize(N);
    cudaDataType_t type = CUDA_R_32F;

    float *A_cpy;
    cudaMalloc(&A_cpy, sizeof(float) * A.size());
    cudaMemcpy(A_cpy, A.data().get(), sizeof(float) * A.size(), cudaMemcpyDeviceToDevice);

    int64_t *d_ipiv;
    cudaMalloc(&d_ipiv, sizeof(int64_t) * N);

    int *d_info;
    cudaMalloc(&d_info, sizeof(int));

    thrust::copy(b.begin(), b.end(), res.begin());

    cusolverDnXgetrf(
        handle,
        params,
        N,
        N,
        type,
        A_cpy,
        N,
        d_ipiv,
        type,
        device_buffer,
        device_buffer_size,
        host_buffer,
        host_buffer_size,
        d_info
    );

    cusolverDnXgetrs(
        handle,
        params,
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
}

void setup_solver(int N, cusolverDnHandle_t &handle, void* &host_buffer, size_t &host_buffer_size, void* &device_buffer, size_t &device_buffer_size, cusolverDnParams_t &params)
{
    cusolverDnCreate(&handle);
    cusolverDnCreateParams(&params);
    cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0);

    cudaDataType_t type = CUDA_R_32F;

    cusolverDnXgetrf_bufferSize(
        handle,
        params,
        N,
        N,
        type,
        nullptr,
        N,
        type,
        &device_buffer_size,
        &host_buffer_size
    );

    if (host_buffer_size > 0) {
        host_buffer = malloc(host_buffer_size);
    } else {
        host_buffer = nullptr;
    }

    cudaMalloc(&device_buffer, device_buffer_size);

    return;
}

void scatter_vector(int N, thrust::device_vector<float> &u_alpha, thrust::device_vector<int> &alpha, thrust::device_vector<float> &res) {
    res.resize(N);
    thrust::scatter(u_alpha.begin(), u_alpha.end(), alpha.begin(), res.begin());

    return;
}

bool solve_termination_test(int N, thrust::device_vector<float> &u, thrust::device_vector<float> &phi, float epsilon, cublasHandle_t &handle) {
    auto negative = [] __device__ (float x) {return x < 0;};

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

struct is_negative {
    const float* u;
    is_negative(const float* _u): u(_u){}

    __device__ bool operator()(int i)
    {
        return u[i] < 0;
    }
};

struct rho_i {
    const float* z;
    const float* w;
    const float* u;

    rho_i(const float* _u, const float* _w, const float* _z): u(_u), w(_w), z(_z) {}

    __device__ float operator()(int i)
    {
        return (z[i]- w[i])/ ((z[i] - w[i]) - u[i]);
    }
};

struct rho_j {
    const float* z;
    const float* w;
    const float* phi;

    rho_j(const float* _z, const float* _w, const float* _phi): z(_z), w(_w), phi(_phi) {}

    __device__ float operator()(int i)
    {
        return (w[i] - z[i])/ (w[i] - z[i]- phi[i]);
    }
};

void get_rhos(int N, thrust::device_vector<float> &z, thrust::device_vector<float> &u, thrust::device_vector<float> &phi, thrust::device_vector<float> &w, thrust::device_vector<int> &gamma, thrust::device_vector<int> &alpha, thrust::device_vector<float> &rhos)
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

// bool valid_dir(int N, thrust::device_vector<float> &z, thrust::device_vector<float> &M, thrust::device_vector<float> &q, thrust::device_vector<float> &d, thrust::device_vector<float> rhos, float xi) 
// {
//     thrust::device_vector<float> w_tau;
//     eval_linear(N, M, q, z, w_tau, handle);


// }

void get_next_iter(int N, thrust::device_vector<float> &z, thrust::device_vector<float> &M, thrust::device_vector<float> &q, thrust::device_vector<float> &u, thrust::device_vector<float> &rhos, cublasHandle_t &handle, float current_merit, float xi, float sigma1, float sigma2, thrust::device_vector<float> &res)
{
    res.resize(N);
    float tau = 1.0;
    for (int i = 0; i < 100; i++){
        thrust::copy(z.begin(), z.end(), res.begin());
        auto idx = thrust::find(rhos.begin(), rhos.end(), tau);
        if (idx != rhos.end()) {
            tau *= sigma2;
            continue;
        
        }
        float x = 1 - tau;
        cublasSscal(handle, N, &x, res.data().get(), 1);
        cublasSaxpy(handle, N, &tau, u.data().get(), 1, res.data().get(), 1);
        thrust::device_vector<float> wv;
        eval_linear(N, M, q, res, wv, handle);
        float merit_zv = get_merit(N, res, wv, handle);
        // printf("new merit: %f\n", merit_zv);
        if ((1-tau*xi)*current_merit > merit_zv) {
            printf("tau: %f\n", tau);
            return;
        }
        tau *= sigma2;
    }
    printf("xxxx\n");
}

int LCP_Newton(int N, thrust::device_vector<float> &M, thrust::device_vector<float> &q, thrust::device_vector<float> &z0, float epsilon, float xi, float sigma0, float sigma1, thrust::device_vector<float> &res) 
{
    int max_iters = 100;
    res.resize(N);
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS initialization failed: %d\n", status);
        return 1;
    }

    cusolverDnHandle_t solver_handle;
    cusolverDnParams_t solver_params;
    size_t host_buffer_size;
    void* host_buffer;
    size_t device_buffer_size;
    void* device_buffer;

    setup_solver(N, solver_handle, host_buffer, host_buffer_size, device_buffer, device_buffer_size, solver_params);


    thrust::device_vector<float> z_v(N);
    thrust::copy(z0.begin(), z0.end(), z_v.begin());

    thrust::device_vector<float> w;
    w.reserve(N);

    thrust::device_vector<float> u;
    u.reserve(N);

    thrust::device_vector<float> phi;
    phi.reserve(N);

    thrust::device_vector<float> old_z;
    old_z.reserve(N);

    thrust::device_vector<int> alpha;
    alpha.reserve(N);

    thrust::device_vector<int> gamma;
    gamma.reserve(N);
    
    thrust::device_vector<float> M_alpha;
    M_alpha.reserve(N*N);

    thrust::device_vector<float> q_alpha;
    q_alpha.reserve(N);

    for (int v = 0; v < max_iters; v++)
    {
        // printf("z: "); for(int i = 0; i < N; i++) {printf("%f ", (float) z_v[i]);}; printf("\n");
        // 1.check for termination

        eval_linear(N, M, q, z_v, w, handle);
        // printf("w: "); for(int i = 0; i < N; i++) {printf("%f ", (float) w[i]);}; printf("\n");
        float merit = get_merit(N, z_v, w, handle);
        // printf("merit: %f\n", merit);
        if (merit < epsilon)
        {
            thrust::copy(z_v.begin(), z_v.end(), res.begin());
            return 0;
        }

        //2.1 compute alpha and gamma

        alpha_set(N, z_v, w, alpha, gamma);
        //2.2 compute u and phi

        submatrix(N, M, alpha, M_alpha);
        subvector(N, q, alpha, q_alpha);

        thrust::device_vector<float> u_alpha;
        solve_linear_system(
            alpha.size(),
            M_alpha, 
            q_alpha, 
            u_alpha, 
            solver_handle, 
            solver_params, 
            host_buffer, 
            host_buffer_size, 
            device_buffer, 
            device_buffer_size);
        
        ::cuda::std::negate<float> minus;
        thrust::transform(u_alpha.begin(), u_alpha.end(), u_alpha.begin(), minus);

        thrust::device_vector<float> u;
        scatter_vector(N, u_alpha, alpha, u);
        // printf("u: "); for(int i = 0; i < N; i++) {printf("%f ", (float) u[i]);}; printf("\n");
        thrust::device_vector<float> phi;
        eval_linear(N, M, q, u, phi, handle);
        // printf("phi: "); for(int i = 0; i < N; i++) {printf("%f ", (float) phi[i]);}; printf("\n");
        //3. check for termination
        if (solve_termination_test(N, u, phi, epsilon, handle))
        {
            // printf("HERE! %d\n", (int) u.size());
            thrust::copy(u.begin(), u.end(), res.begin());
            return 0;
        }
        // 4.1 find rhos
        thrust::device_vector<float> rhos;
        get_rhos(N, z_v, u, phi, w, gamma, alpha, rhos);
        // printf("rho: "); for(int i = 0; i < rhos.size(); i++) {printf("%f ", (float) rhos[i]);}; printf("\n");
        //4.2 calculate z+1
        thrust::device_vector<float> old_z(N);
        thrust::copy(z_v.begin(), z_v.end(), old_z.begin());
        get_next_iter(N, old_z, M, q, u, rhos, handle, merit, xi, sigma0, sigma1, z_v);
        // printf("-----------\n");
    }
    return 1;
}