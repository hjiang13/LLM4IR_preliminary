module {
  func.func @main(
    %input: tensor<1x8x8x8xf32>
  ) -> tensor<1x8xf32> {
    %init = tensor.empty() : tensor<1x8xf32>
    
    %output = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k, l) -> (i, j, k, l)>,
        affine_map<(i, j, k, l) -> (i, l)>
      ],
      iterator_types = ["parallel", "reduction", "reduction", "parallel"]
    } ins(%input : tensor<1x8x8x8xf32>) 
        outs(%init : tensor<1x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %result = arith.addf %in, %out : f32
        linalg.yield %result : f32
    } -> tensor<1x8xf32>
    
    return %output : tensor<1x8xf32>
  }
}