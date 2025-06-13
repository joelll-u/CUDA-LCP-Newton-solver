#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

#include "MCP_Interface.h"
#include "Path.h"
#include "PathOptions.h"

#include "Macros.h"
#include "Output_Interface.h"
#include "Options.h"
#include "Types.h"

static std::vector<int> col_start_;
static std::vector<int> col_len_;
static std::vector<int> row_;
static std::vector<double> mdata;

static bool inputted = false;
static bool sparse = false;

static Eigen::MatrixXd m_eigen;
static Eigen::SparseMatrix<double> m_eigen_sparse;

static Eigen::VectorXd q_eigen;

static int nnz_;
static int size_;

static double* z_0;

void problem_size(void* id, int* size, int* nnz){
    *nnz = nnz_;
    *size = size_;
}

void bounds(void *id, int size, double *x, double *l, double *u)
{
    for (int i = 0; i < size; i++) {
        x[i] = z_0[i];
        l[i] = 0;
    }
}

int funcEval(void* id, int n, double *z, double *f)
{
    Eigen::Map<const Eigen::VectorXd> z_vec(z, n);     // Wrap input z
    Eigen::Map<Eigen::VectorXd> result_vec(f, n); // Wrap output result

    if (!sparse ) {
        result_vec = m_eigen * z_vec + q_eigen; // Perform the operation
    } else {
        result_vec = (m_eigen_sparse * z_vec + q_eigen).eval();
        // std:: cout << n << std::endl;
        // for (int i = 0; i < n; i++)
        // {
        //     std::cout << z[i] << " ";
        // }
        // std::cout << std::endl;
        // for (int i = 0; i < n; i++) {
        //     std::cout << f[i] << " ";
        // }
        // std::cout << std::endl;
        // std::cout<< result_vec << std::endl;
        
        // throw std::invalid_argument("a");
    }
    return 0;
}

int jacEval(void *id, int n, double *x,
            int wantf, double *f,
            int *nnz, int *col, int *len,
            int *row, double *data)
{
    if (wantf) {
        funcEval(id, n, x, f);
    }
    if (!inputted) { 
        std::memcpy(col, col_start_.data(), col_start_.size() * sizeof(int));
        std::memcpy(len, col_len_.data(), col_len_.size() * sizeof(int));
        std::memcpy(row, row_.data(), row_.size() * sizeof(int));
        inputted = true;
    }
    std::memcpy(data, mdata.data(), mdata.size() * sizeof(double));
    *nnz = nnz_;

    return 0;
}

void matrixToCCF(int n, const std::vector<double> &m)
{
    col_start_.clear();
    mdata.clear();
    row_.clear();
    col_len_.clear();

    int cnt = 0;
    int nnz_col = 0;
    nnz_ = 0;
    for (int i = 0; i < n; i++)
    {
        col_start_.push_back((i == 0) ? 1 : col_start_[i - 1] + nnz_col);
        nnz_col = 0;
        for (int j = 0; j < n; j++)
        {
            double x = m[i*n + j];
            if (x == 0)
                continue;
            nnz_col++;
            mdata.push_back(x);
            row_.push_back(j + 1);
            cnt++;
        }
        col_len_.push_back(nnz_col);
        nnz_ += nnz_col;
    }
    // printf("Col_start: \n");
    // for (int i = 0; i < col_start_.size(); i++) {
    //     printf("%d ", col_start_[i]);
    // }
    // printf("\n");

    // printf("Col_len: \n");
    // for (int i = 0; i < col_len_.size(); i++)
    // {
    //     printf("%d ", col_len_[i]);
    // }
    // printf("\n");


    // printf("Row: \n");
    // for (int i = 0; i < row_.size(); i++)
    // {
    //     printf("%d ", row_[i]);
    // }
    // printf("\n");

    // printf("Data: \n");
    // for (int i = 0; i < mdata.size(); i++)
    // {
    //     printf("%f ", mdata[i]);
    // }
    // printf("\n");
}

void initializeM(const std::vector<double> &m, int n)
{
    sparse = false;
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mapped(m.data(), n, n);

    m_eigen = mapped;
    matrixToCCF(n, m);
}

void initializeM_sparse(const std::vector<double> &m, int n)
{
    // m_eigen = mapped;
    sparse = true;
    matrixToCCF(n, m);

    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < n; i++)
    {
        int colStart = col_start_[i] - 1;   // Convert to 0-based
        int colEnd = col_len_[i] + colStart; // Convert to 0-based

        for (int j = colStart; j < colEnd; j++)
        {
            int row = row_[j] - 1; // Convert to 0-based
            triplets.emplace_back(row, i, mdata[j]);
        }
    }

    m_eigen_sparse.resize(n, n);
    m_eigen_sparse.setFromTriplets(triplets.begin(), triplets.end());
    m_eigen_sparse.makeCompressed();
    // std::cout << Eigen::MatrixXd(m_eigen_sparse) << std::endl;
}

void initializeM_from_sparse_symmetric(host_matrix_sparse &x, int n) {

    col_start_.clear();
    mdata.clear();
    row_.clear();
    col_len_.clear();

    for (int i = 0; i < x.row_offsets.size() - 1; i++) {
        col_start_.push_back(x.row_offsets[i] + 1);
        col_len_.push_back(x.row_offsets[i+1] - x.row_offsets[i]);
    }
    
    for(int i = 0; i < x.column_indices.size(); i++ ){
        row_.push_back(x.column_indices[i] + 1);
        mdata.push_back(x.values[i]);
    }
    nnz_ = x.values.size();
    sparse = true;
    size_ = x.row_offsets.size() - 1;

    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < n; i++)
    {
        int colStart = col_start_[i] - 1;    // Convert to 0-based
        int colEnd = col_len_[i] + colStart; // Convert to 0-based

        for (int j = colStart; j < colEnd; j++)
        {
            int row = row_[j] - 1; // Convert to 0-based
            triplets.emplace_back(row, i, mdata[j]);
        }
    }

    m_eigen_sparse.resize(n, n);
    m_eigen_sparse.setFromTriplets(triplets.begin(), triplets.end());
    m_eigen_sparse.makeCompressed();
}

void initializeQ(const std::vector<double> q, int n)
{
    q_eigen = Eigen::Map<const Eigen::VectorXd>(q.data(), q.size());
}


int path_solve(int n, double* z, double* w, bool lemke = false)
{
    Options_Interface* o;
    MCP *m;
    MCP_Termination t = MCP_Error;
    Information info;

    z_0 = z;
    size_ = n;
    inputted = false;

    o = Options_Create();
    Path_AddOptions(o);
    Options_Default(o);

    if (lemke) {
        Options_Set(o, "crash_method none");
        Options_Set(o, "lemke_start first");
        Options_SetInt(o, "major_iteration_limit", 1);
        Options_Set(o, "nms no");
    }

    MCP_Interface mcp_interface =
        {
            NULL,
            problem_size, bounds,
            funcEval, jacEval,
            NULL, /* hessian evaluation */
            NULL, NULL,
            NULL, NULL,
            NULL}
            ;

    m = MCP_Create(n, nnz_);
    MCP_SetInterface(m, &mcp_interface);

    info.generate_output = 0;
    info.use_start = True;
    info.use_basics = True;

    t = Path_Solve(m, &info);

    double* tempZ = MCP_GetX(m);
    double* tempF = MCP_GetF(m);

    for (int i = 0; i < n; i++)
    {
        z[i] = tempZ[i];
        w[i] = tempF[i];
    }

    MCP_Destroy(m);
    Options_Destroy(o);

    // col_start_.clear();
    // col_len_.clear();
    // row_.clear();
    // mdata.clear();
    // m_eigen.resize(0,0);
    // q_eigen.resize(0,0);

    return t;
}


int path_simple_solve(int n, std::vector<double> M, std::vector<double> q, bool lemke = false)
{
    initializeM(M, n);
    initializeQ(q, n);

    std::vector<double> z(n,1);
    std::vector<double> f(n);

    MCP_Termination status = (MCP_Termination) path_solve(n, z.data(), f.data());
    // for (int i = 0; i < n; i++) {
    //     printf("%f ", z[i]);
    // }
    // printf("\n");

    if (status == MCP_Solved) {
        return 1;
    } else if (status == MCP_Infeasible) {
        return 0;
    } else {
        // printf("Uh oh, unknown path error occured with status %d\n", status);
        return 0;
    }
}