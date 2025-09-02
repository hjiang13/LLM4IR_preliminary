module {
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64) -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %5 = llvm.insertvalue %arg7, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %7 = llvm.insertvalue %arg8, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %8 = llvm.insertvalue %arg5, %7[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %9 = llvm.insertvalue %arg9, %8[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %10 = llvm.insertvalue %arg6, %9[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %11 = llvm.insertvalue %arg10, %10[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %12 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %13 = llvm.mlir.constant(-1.000000e+00 : f32) : f32
    %14 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.mlir.constant(8 : index) : i64
    llvm.br ^bb1(%15 : i64)
  ^bb1(%18: i64):  // 2 preds: ^bb0, ^bb11
    %19 = llvm.icmp "slt" %18, %16 : i64
    llvm.cond_br %19, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%15 : i64)
  ^bb3(%20: i64):  // 2 preds: ^bb2, ^bb10
    %21 = llvm.icmp "slt" %20, %17 : i64
    llvm.cond_br %21, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%15 : i64)
  ^bb5(%22: i64):  // 2 preds: ^bb4, ^bb9
    %23 = llvm.icmp "slt" %22, %17 : i64
    llvm.cond_br %23, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%15 : i64)
  ^bb7(%24: i64):  // 2 preds: ^bb6, ^bb8
    %25 = llvm.icmp "slt" %24, %17 : i64
    llvm.cond_br %25, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %26 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %27 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %28 = llvm.getelementptr %26[%27] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %29 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %30 = llvm.mul %18, %29 : i64
    %31 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %32 = llvm.mul %20, %31 : i64
    %33 = llvm.add %30, %32 : i64
    %34 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %35 = llvm.mul %22, %34 : i64
    %36 = llvm.add %33, %35 : i64
    %37 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %38 = llvm.mul %24, %37 : i64
    %39 = llvm.add %36, %38 : i64
    %40 = llvm.getelementptr %28[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %41 = llvm.load %40 : !llvm.ptr -> f32
    %42 = llvm.intr.maximum(%41, %14) : (f32, f32) -> f32
    %43 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %44 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %45 = llvm.getelementptr %43[%44] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %46 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %47 = llvm.mul %18, %46 : i64
    %48 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %49 = llvm.mul %20, %48 : i64
    %50 = llvm.add %47, %49 : i64
    %51 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %52 = llvm.mul %22, %51 : i64
    %53 = llvm.add %50, %52 : i64
    %54 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %55 = llvm.mul %24, %54 : i64
    %56 = llvm.add %53, %55 : i64
    %57 = llvm.getelementptr %45[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %42, %57 : f32, !llvm.ptr
    %58 = llvm.add %24, %16 : i64
    llvm.br ^bb7(%58 : i64)
  ^bb9:  // pred: ^bb7
    %59 = llvm.add %22, %16 : i64
    llvm.br ^bb5(%59 : i64)
  ^bb10:  // pred: ^bb5
    %60 = llvm.add %20, %16 : i64
    llvm.br ^bb3(%60 : i64)
  ^bb11:  // pred: ^bb3
    %61 = llvm.add %18, %16 : i64
    llvm.br ^bb1(%61 : i64)
  ^bb12:  // pred: ^bb1
    llvm.br ^bb13(%15 : i64)
  ^bb13(%62: i64):  // 2 preds: ^bb12, ^bb23
    %63 = llvm.icmp "slt" %62, %16 : i64
    llvm.cond_br %63, ^bb14, ^bb24
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%15 : i64)
  ^bb15(%64: i64):  // 2 preds: ^bb14, ^bb22
    %65 = llvm.icmp "slt" %64, %17 : i64
    llvm.cond_br %65, ^bb16, ^bb23
  ^bb16:  // pred: ^bb15
    llvm.br ^bb17(%15 : i64)
  ^bb17(%66: i64):  // 2 preds: ^bb16, ^bb21
    %67 = llvm.icmp "slt" %66, %17 : i64
    llvm.cond_br %67, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    llvm.br ^bb19(%15 : i64)
  ^bb19(%68: i64):  // 2 preds: ^bb18, ^bb20
    %69 = llvm.icmp "slt" %68, %17 : i64
    llvm.cond_br %69, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %70 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %71 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %72 = llvm.getelementptr %70[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %73 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %74 = llvm.mul %62, %73 : i64
    %75 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %76 = llvm.mul %64, %75 : i64
    %77 = llvm.add %74, %76 : i64
    %78 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %79 = llvm.mul %66, %78 : i64
    %80 = llvm.add %77, %79 : i64
    %81 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %82 = llvm.mul %68, %81 : i64
    %83 = llvm.add %80, %82 : i64
    %84 = llvm.getelementptr %72[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %85 = llvm.load %84 : !llvm.ptr -> f32
    %86 = llvm.fmul %85, %13 : f32
    %87 = llvm.intr.exp(%86) : (f32) -> f32
    %88 = llvm.fadd %87, %12 : f32
    %89 = llvm.fdiv %12, %88 : f32
    %90 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %91 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %92 = llvm.getelementptr %90[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %93 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %94 = llvm.mul %62, %93 : i64
    %95 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %96 = llvm.mul %64, %95 : i64
    %97 = llvm.add %94, %96 : i64
    %98 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %99 = llvm.mul %66, %98 : i64
    %100 = llvm.add %97, %99 : i64
    %101 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %102 = llvm.mul %68, %101 : i64
    %103 = llvm.add %100, %102 : i64
    %104 = llvm.getelementptr %92[%103] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %89, %104 : f32, !llvm.ptr
    %105 = llvm.add %68, %16 : i64
    llvm.br ^bb19(%105 : i64)
  ^bb21:  // pred: ^bb19
    %106 = llvm.add %66, %16 : i64
    llvm.br ^bb17(%106 : i64)
  ^bb22:  // pred: ^bb17
    %107 = llvm.add %64, %16 : i64
    llvm.br ^bb15(%107 : i64)
  ^bb23:  // pred: ^bb15
    %108 = llvm.add %62, %16 : i64
    llvm.br ^bb13(%108 : i64)
  ^bb24:  // pred: ^bb13
    llvm.return %11 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  }
}

