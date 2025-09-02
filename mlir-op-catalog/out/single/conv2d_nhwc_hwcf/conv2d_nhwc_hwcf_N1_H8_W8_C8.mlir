module {
  func.func @main(
    %input: tensor<1x8x8x8xf32>,
    %kernel: tensor<3x3x8x8xf32>
  ) -> tensor<1x6x6x8xf32> {
    %init = tensor.empty() : tensor<1x6x6x8xf32>
    %output = linalg.conv_2d_nhwc_hwcf
      ins(%input, %kernel : tensor<1x8x8x8xf32>, tensor<3x3x8x8xf32>)
      outs(%init : tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>
    return %output : tensor<1x6x6x8xf32>
  }
}