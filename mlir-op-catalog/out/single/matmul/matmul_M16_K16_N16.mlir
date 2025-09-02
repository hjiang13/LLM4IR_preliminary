module {
  func.func @main(
    %A: tensor<16x16xf32>,
    %B: tensor<16x16xf32>
  ) -> tensor<16x16xf32> {
    %init = tensor.empty() : tensor<16x16xf32>
    %C = linalg.matmul ins(%A, %B : tensor<16x16xf32>, tensor<16x16xf32>)
                      outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %C : tensor<16x16xf32>
  }
}