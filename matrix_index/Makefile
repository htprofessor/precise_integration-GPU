# 使用 nvcc 作为编译器
NVCC = nvcc
CXXFLAGS = -O3 -std=c++11

# 链接 CuBLAS 和 CUDA runtime 库
LDFLAGS = -lcublas -lcudart

# 目标可执行文件
TARGET = matrix_exponential_program

# 源文件
SRCS = matrix_exponential.cu

# 编译规则
all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
