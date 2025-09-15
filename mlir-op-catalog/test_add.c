#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 定义tensor结构体
typedef struct {
    float* data;
    float* aligned_data;
    long long size;
    long long shape[4];
    long long strides[4];
} tensor_t;

// 声明MLIR生成的函数
extern void* mlir_add(void* x, void* y, long long x_size, long long x_shape0, long long x_stride0, 
                  long long x_shape1, long long x_stride1, long long x_shape2, long long x_stride2, 
                  long long x_shape3, long long x_stride3, void* y_data, void* y_aligned, 
                  long long y_size, long long y_shape0, long long y_stride0, long long y_shape1, 
                  long long y_stride1, long long y_shape2, long long y_stride2, long long y_shape3, 
                  long long y_stride3);

int main() {
    printf("🧮 测试MLIR生成的add操作\n");
    
    // 创建测试数据
    const int N=1, H=8, W=8, C=8;
    const int total_size = N * H * W * C;
    
    // 分配内存
    float* x_data = (float*)malloc(total_size * sizeof(float));
    float* y_data = (float*)malloc(total_size * sizeof(float));
    float* result_data = (float*)malloc(total_size * sizeof(float));
    
    // 初始化数据
    for (int i = 0; i < total_size; i++) {
        x_data[i] = (float)(i % 10);
        y_data[i] = (float)((i + 1) % 10);
    }
    
    printf("输入数据 x[0:5] = ");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", x_data[i]);
    }
    printf("\n");
    
    printf("输入数据 y[0:5] = ");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", y_data[i]);
    }
    printf("\n");
    
    // 调用MLIR生成的函数
    void* result = mlir_add(x_data, x_data, total_size, N, N*H*W*C, H, H*W*C, W, W*C, C, 1,
                        y_data, y_data, total_size, N, N*H*W*C, H, H*W*C, W, W*C, C, 1);
    
    printf("✅ MLIR add操作执行完成！\n");
    printf("结果指针: %p\n", result);
    
    // 清理内存
    free(x_data);
    free(y_data);
    free(result_data);
    
    return 0;
}
