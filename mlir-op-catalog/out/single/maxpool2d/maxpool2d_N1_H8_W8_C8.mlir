module {
  func.func @main(
    %input: tensor<1x8x8x8xf32>
  ) -> tensor<1x4x4x8xf32> {
    %init = tensor.empty() : tensor<1x4x4x8xf32>
    
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k, l) -> (i, j*2, k*2, l)>,
        affine_map<(i, j, k, l) -> (i, j, k, l)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input : tensor<1x8x8x8xf32>) 
        outs(%init : tensor<1x4x4x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    } -> tensor<1x4x4x8xf32>
    
    return %output : tensor<1x4x4x8xf32>
  }
}