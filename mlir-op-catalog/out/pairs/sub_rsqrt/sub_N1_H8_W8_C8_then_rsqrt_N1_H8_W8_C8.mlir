module {
  func.func @main(
    %input: tensor<1x8x8x8xf32>
  ) -> tensor<1x8x8x8xf32> {
    %result1 = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k, l) -> (i, j, k, l)>,
        affine_map<(i, j, k, l) -> (i, j, k, l)>,
        affine_map<(i, j, k, l) -> (i, j, k, l)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %input : tensor<1x8x8x8xf32>, tensor<1x8x8x8xf32>) 
        outs(%input : tensor<1x8x8x8xf32>) {
      ^bb0(%x: f32, %y: f32, %o: f32):
        %result = arith.subf %x, %y : f32
        linalg.yield %result : f32
    } -> tensor<1x8x8x8xf32>

    %result2 = linalg.generic {
      indexing_maps = [
        affine_map<(i, j, k, l) -> (i, j, k, l)>,
        affine_map<(i, j, k, l) -> (i, j, k, l)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%result1 : tensor<1x8x8x8xf32>) 
        outs(%result1 : tensor<1x8x8x8xf32>) {
      ^bb0(%x: f32, %o: f32):
        linalg.yield %x : f32
    } -> tensor<1x8x8x8xf32>

    return %result2 : tensor<1x8x8x8xf32>
  }
}