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
        %result = arith.mulf %x, %y : f32
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
        %cst_neg1 = arith.constant -1.0 : f32
        %cst1 = arith.constant 1.0 : f32
        %neg_x = arith.mulf %x, %cst_neg1 : f32
        %exp_neg_x = math.exp %neg_x : f32
        %denom = arith.addf %cst1, %exp_neg_x : f32
        %result = arith.divf %cst1, %denom : f32
        linalg.yield %result : f32
    } -> tensor<1x8x8x8xf32>

    return %result2 : tensor<1x8x8x8xf32>
  }
}