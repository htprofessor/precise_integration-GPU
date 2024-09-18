#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>


#define MATRIX_SIZE 512


#define CUDA_CHECK(call)                                                    \
do {                                                                        \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__; \
        std::cerr << ": " << cudaGetErrorString(err) << std::endl;          \
        exit(1);                                                            \
    }                                                                       \
} while (0)

#define CUBLAS_CHECK(call)                                                  \
do {                                                                        \
    cublasStatus_t status = call;                                           \
    if (status != CUBLAS_STATUS_SUCCESS) {                                  \
        std::cerr << "CuBLAS error in " << __FILE__ << " at line " << __LINE__; \
        std::cerr << ": " << status << std::endl;                           \
        exit(1);                                                            \
    }                                                                       \
} while (0)

void matrix_exponential(const double* d_A, double* d_expAt, int n, double delt, int N, cublasHandle_t handle);
bool read_matrix_from_file(const std::string& filename, std::vector<double>& matrix, int& n);
void matrix_multiply(const double* d_A, const double* d_B, double* d_C, int n, cublasHandle_t handle);
void matrix_add(double* d_A, const double* d_B, int n, cublasHandle_t handle);
void matrix_square(const double* d_A, double* d_C, int n, cublasHandle_t handle);

int main() {
  
    auto program_start = std::chrono::high_resolution_clock::now();

    
    std::string matrixAFile = "matrixA_512.txt";
    std::vector<double> A;
    int nA;

    if (!read_matrix_from_file(matrixAFile, A, nA)) {
        std::cerr << "读取矩阵时出错" << std::endl;
        return -1;
    }

    if (nA != MATRIX_SIZE) {
        std::cerr << "矩阵的维度不是512阶" << std::endl;
        return -1;
    }

    int n = nA;

   
    double *d_A, *d_expAt;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_expAt, n * n * sizeof(double)));

    
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

  
    double delt = 0.0000001;  // 时间步长
    int N = 20;               // 精细积分法迭代次数

  
    auto start = std::chrono::high_resolution_clock::now();

    
    matrix_exponential(d_A, d_expAt, n, delt, N, handle);

    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    
    std::vector<double> expAt(n * n);
    CUDA_CHECK(cudaMemcpy(expAt.data(), d_expAt, n * n * sizeof(double), cudaMemcpyDeviceToHost));

    
    std::cout << "矩阵指数计算时间: " << elapsed.count() << " 秒" << std::endl;

    
    std::cout << "exp(A * delt) 的前4x4块：\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << expAt[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_expAt));

    
    CUBLAS_CHECK(cublasDestroy(handle));

    auto program_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> program_elapsed = program_end - program_start;
    std::cout << "程序总运行时间: " << program_elapsed.count() << " 秒" << std::endl;

    return 0;
}


void matrix_exponential(const double* d_A, double* d_expAt, int n, double delt, int N, cublasHandle_t handle) {
    double *d_T_a, *d_temp, *d_I;
    CUDA_CHECK(cudaMalloc(&d_T_a, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_temp, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_I, n * n * sizeof(double)));

  
    const double alpha = delt;
    CUBLAS_CHECK(cublasDcopy(handle, n * n, d_A, 1, d_T_a, 1));  
    CUBLAS_CHECK(cublasDscal(handle, n * n, &alpha, d_T_a, 1));   
    CUBLAS_CHECK(cublasDcopy(handle, n * n, d_T_a, 1, d_temp, 1));  
    const double beta = 0.5;
    CUBLAS_CHECK(cublasDscal(handle, n * n, &beta, d_temp, 1));    
    const double one = 1.0;
    CUBLAS_CHECK(cublasDaxpy(handle, n * n, &one, d_temp, 1, d_T_a, 1));  
    // 递推计算 T_a
    for (int iter = 0; iter < N; iter++) {
        matrix_square(d_T_a, d_temp, n, handle);
        const double two = 2.0;
        CUBLAS_CHECK(cublasDscal(handle, n * n, &two, d_T_a, 1));  // T_a = 2 * T_a
        CUBLAS_CHECK(cublasDaxpy(handle, n * n, &one, d_temp, 1, d_T_a, 1));
    }

    CUBLAS_CHECK(cublasDcopy(handle, n * n, d_T_a, 1, d_expAt, 1));  // expAt = T_a

    std::vector<double> I(n * n, 0.0);
    for (int i = 0; i < n; i++) {
        I[i * n + i] = 1.0;
    }
    CUDA_CHECK(cudaMemcpy(d_I, I.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    matrix_add(d_expAt, d_I, n, handle);

    CUDA_CHECK(cudaFree(d_T_a));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_I));
}

bool read_matrix_from_file(const std::string& filename, std::vector<double>& matrix, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    std::string line;
    std::vector<double> temp_matrix;
    int row_count = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double value;
        int col_count = 0;
        while (iss >> value) {
            temp_matrix.push_back(value);
            col_count++;
        }
        if (row_count == 0) {
            n = col_count;
        } else if (col_count != n) {
            std::cerr << "文件格式不正确，列数不匹配" << std::endl;
            return false;
        }
        row_count++;
    }

    matrix = std::move(temp_matrix); 
    return true;
}

void matrix_multiply(const double* d_A, const double* d_B, double* d_C, int n, cublasHandle_t handle) {
    const double alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n));
}

void matrix_add(double* d_A, const double* d_B, int n, cublasHandle_t handle) {
    const double alpha = 1.0;
    CUBLAS_CHECK(cublasDaxpy(handle, n * n, &alpha, d_B, 1, d_A, 1));
}

void matrix_square(const double* d_A, double* d_C, int n, cublasHandle_t handle) {
    matrix_multiply(d_A, d_A, d_C, n, handle);
}
