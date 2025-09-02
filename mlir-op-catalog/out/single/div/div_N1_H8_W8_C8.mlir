module {
  func.func @main(%x: tensor<1x8x8x8xf32>, %y: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32> {
    %init = tensor.empty() : tensor<1x8x8x8xf32>
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k, l) -> (i, j, k, l)>,
        affine_map<(i, j, k, l) -> (i, j, k, l)>,
        affine_map<(i, j, k, l) -> (i, j, k, l)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%x, %y : tensor<1x8x8x8xf32>, tensor<1x8x8x8xf32>) 
        outs(%init : tensor<1x8x8x8xf32>) {
      ^bb0(%x_val: f32, %y_val: f32, %out: f32):
        %res = arith.divf %x_val, %y_val : f32
        linalg.yield %res : f32
    } -> tensor<1x8x8x8xf32>
    return %result : tensor<1x8x8x8xf32>
  }
}