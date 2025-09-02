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
    %12 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(8 : index) : i64
    llvm.br ^bb1(%13 : i64)
  ^bb1(%16: i64):  // 2 preds: ^bb0, ^bb11
    %17 = llvm.icmp "slt" %16, %14 : i64
    llvm.cond_br %17, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%13 : i64)
  ^bb3(%18: i64):  // 2 preds: ^bb2, ^bb10
    %19 = llvm.icmp "slt" %18, %15 : i64
    llvm.cond_br %19, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%13 : i64)
  ^bb5(%20: i64):  // 2 preds: ^bb4, ^bb9
    %21 = llvm.icmp "slt" %20, %15 : i64
    llvm.cond_br %21, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%13 : i64)
  ^bb7(%22: i64):  // 2 preds: ^bb6, ^bb8
    %23 = llvm.icmp "slt" %22, %15 : i64
    llvm.cond_br %23, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %24 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %25 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %26 = llvm.getelementptr %24[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %27 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %28 = llvm.mul %16, %27 : i64
    %29 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %30 = llvm.mul %18, %29 : i64
    %31 = llvm.add %28, %30 : i64
    %32 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %33 = llvm.mul %20, %32 : i64
    %34 = llvm.add %31, %33 : i64
    %35 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %36 = llvm.mul %22, %35 : i64
    %37 = llvm.add %34, %36 : i64
    %38 = llvm.getelementptr %26[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %39 = llvm.load %38 : !llvm.ptr -> f32
    %40 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %41 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %42 = llvm.getelementptr %40[%41] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %43 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %44 = llvm.mul %16, %43 : i64
    %45 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %46 = llvm.mul %18, %45 : i64
    %47 = llvm.add %44, %46 : i64
    %48 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %49 = llvm.mul %20, %48 : i64
    %50 = llvm.add %47, %49 : i64
    %51 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %52 = llvm.mul %22, %51 : i64
    %53 = llvm.add %50, %52 : i64
    %54 = llvm.getelementptr %42[%53] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %39, %54 : f32, !llvm.ptr
    %55 = llvm.add %22, %14 : i64
    llvm.br ^bb7(%55 : i64)
  ^bb9:  // pred: ^bb7
    %56 = llvm.add %20, %14 : i64
    llvm.br ^bb5(%56 : i64)
  ^bb10:  // pred: ^bb5
    %57 = llvm.add %18, %14 : i64
    llvm.br ^bb3(%57 : i64)
  ^bb11:  // pred: ^bb3
    %58 = llvm.add %16, %14 : i64
    llvm.br ^bb1(%58 : i64)
  ^bb12:  // pred: ^bb1
    llvm.br ^bb13(%13 : i64)
  ^bb13(%59: i64):  // 2 preds: ^bb12, ^bb23
    %60 = llvm.icmp "slt" %59, %14 : i64
    llvm.cond_br %60, ^bb14, ^bb24
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%13 : i64)
  ^bb15(%61: i64):  // 2 preds: ^bb14, ^bb22
    %62 = llvm.icmp "slt" %61, %15 : i64
    llvm.cond_br %62, ^bb16, ^bb23
  ^bb16:  // pred: ^bb15
    llvm.br ^bb17(%13 : i64)
  ^bb17(%63: i64):  // 2 preds: ^bb16, ^bb21
    %64 = llvm.icmp "slt" %63, %15 : i64
    llvm.cond_br %64, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    llvm.br ^bb19(%13 : i64)
  ^bb19(%65: i64):  // 2 preds: ^bb18, ^bb20
    %66 = llvm.icmp "slt" %65, %15 : i64
    llvm.cond_br %66, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %67 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %68 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %69 = llvm.getelementptr %67[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %70 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %71 = llvm.mul %59, %70 : i64
    %72 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %73 = llvm.mul %61, %72 : i64
    %74 = llvm.add %71, %73 : i64
    %75 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %76 = llvm.mul %63, %75 : i64
    %77 = llvm.add %74, %76 : i64
    %78 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %79 = llvm.mul %65, %78 : i64
    %80 = llvm.add %77, %79 : i64
    %81 = llvm.getelementptr %69[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %82 = llvm.load %81 : !llvm.ptr -> f32
    %83 = llvm.intr.maximum(%82, %12) : (f32, f32) -> f32
    %84 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %85 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %86 = llvm.getelementptr %84[%85] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %87 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %88 = llvm.mul %59, %87 : i64
    %89 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %90 = llvm.mul %61, %89 : i64
    %91 = llvm.add %88, %90 : i64
    %92 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %93 = llvm.mul %63, %92 : i64
    %94 = llvm.add %91, %93 : i64
    %95 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %96 = llvm.mul %65, %95 : i64
    %97 = llvm.add %94, %96 : i64
    %98 = llvm.getelementptr %86[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %83, %98 : f32, !llvm.ptr
    %99 = llvm.add %65, %14 : i64
    llvm.br ^bb19(%99 : i64)
  ^bb21:  // pred: ^bb19
    %100 = llvm.add %63, %14 : i64
    llvm.br ^bb17(%100 : i64)
  ^bb22:  // pred: ^bb17
    %101 = llvm.add %61, %14 : i64
    llvm.br ^bb15(%101 : i64)
  ^bb23:  // pred: ^bb15
    %102 = llvm.add %59, %14 : i64
    llvm.br ^bb13(%102 : i64)
  ^bb24:  // pred: ^bb13
    llvm.return %11 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  }
}

