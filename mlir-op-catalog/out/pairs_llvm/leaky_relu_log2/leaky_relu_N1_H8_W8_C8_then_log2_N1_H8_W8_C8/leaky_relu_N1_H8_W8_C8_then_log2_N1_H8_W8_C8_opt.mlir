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
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.mlir.constant(8 : index) : i64
    llvm.br ^bb1(%12 : i64)
  ^bb1(%15: i64):  // 2 preds: ^bb0, ^bb11
    %16 = llvm.icmp "slt" %15, %13 : i64
    llvm.cond_br %16, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%12 : i64)
  ^bb3(%17: i64):  // 2 preds: ^bb2, ^bb10
    %18 = llvm.icmp "slt" %17, %14 : i64
    llvm.cond_br %18, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%12 : i64)
  ^bb5(%19: i64):  // 2 preds: ^bb4, ^bb9
    %20 = llvm.icmp "slt" %19, %14 : i64
    llvm.cond_br %20, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%12 : i64)
  ^bb7(%21: i64):  // 2 preds: ^bb6, ^bb8
    %22 = llvm.icmp "slt" %21, %14 : i64
    llvm.cond_br %22, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %23 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %25 = llvm.getelementptr %23[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %26 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %27 = llvm.mul %15, %26 : i64
    %28 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %29 = llvm.mul %17, %28 : i64
    %30 = llvm.add %27, %29 : i64
    %31 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %32 = llvm.mul %19, %31 : i64
    %33 = llvm.add %30, %32 : i64
    %34 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %35 = llvm.mul %21, %34 : i64
    %36 = llvm.add %33, %35 : i64
    %37 = llvm.getelementptr %25[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.load %37 : !llvm.ptr -> f32
    %39 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %40 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %41 = llvm.getelementptr %39[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %42 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %43 = llvm.mul %15, %42 : i64
    %44 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %45 = llvm.mul %17, %44 : i64
    %46 = llvm.add %43, %45 : i64
    %47 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %48 = llvm.mul %19, %47 : i64
    %49 = llvm.add %46, %48 : i64
    %50 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %51 = llvm.mul %21, %50 : i64
    %52 = llvm.add %49, %51 : i64
    %53 = llvm.getelementptr %41[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %38, %53 : f32, !llvm.ptr
    %54 = llvm.add %21, %13 : i64
    llvm.br ^bb7(%54 : i64)
  ^bb9:  // pred: ^bb7
    %55 = llvm.add %19, %13 : i64
    llvm.br ^bb5(%55 : i64)
  ^bb10:  // pred: ^bb5
    %56 = llvm.add %17, %13 : i64
    llvm.br ^bb3(%56 : i64)
  ^bb11:  // pred: ^bb3
    %57 = llvm.add %15, %13 : i64
    llvm.br ^bb1(%57 : i64)
  ^bb12:  // pred: ^bb1
    llvm.br ^bb13(%12 : i64)
  ^bb13(%58: i64):  // 2 preds: ^bb12, ^bb23
    %59 = llvm.icmp "slt" %58, %13 : i64
    llvm.cond_br %59, ^bb14, ^bb24
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%12 : i64)
  ^bb15(%60: i64):  // 2 preds: ^bb14, ^bb22
    %61 = llvm.icmp "slt" %60, %14 : i64
    llvm.cond_br %61, ^bb16, ^bb23
  ^bb16:  // pred: ^bb15
    llvm.br ^bb17(%12 : i64)
  ^bb17(%62: i64):  // 2 preds: ^bb16, ^bb21
    %63 = llvm.icmp "slt" %62, %14 : i64
    llvm.cond_br %63, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    llvm.br ^bb19(%12 : i64)
  ^bb19(%64: i64):  // 2 preds: ^bb18, ^bb20
    %65 = llvm.icmp "slt" %64, %14 : i64
    llvm.cond_br %65, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %66 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %67 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %68 = llvm.getelementptr %66[%67] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %69 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %70 = llvm.mul %58, %69 : i64
    %71 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %72 = llvm.mul %60, %71 : i64
    %73 = llvm.add %70, %72 : i64
    %74 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %75 = llvm.mul %62, %74 : i64
    %76 = llvm.add %73, %75 : i64
    %77 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %78 = llvm.mul %64, %77 : i64
    %79 = llvm.add %76, %78 : i64
    %80 = llvm.getelementptr %68[%79] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %81 = llvm.load %80 : !llvm.ptr -> f32
    %82 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %83 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %84 = llvm.getelementptr %82[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %85 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %86 = llvm.mul %58, %85 : i64
    %87 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %88 = llvm.mul %60, %87 : i64
    %89 = llvm.add %86, %88 : i64
    %90 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %91 = llvm.mul %62, %90 : i64
    %92 = llvm.add %89, %91 : i64
    %93 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %94 = llvm.mul %64, %93 : i64
    %95 = llvm.add %92, %94 : i64
    %96 = llvm.getelementptr %84[%95] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %81, %96 : f32, !llvm.ptr
    %97 = llvm.add %64, %13 : i64
    llvm.br ^bb19(%97 : i64)
  ^bb21:  // pred: ^bb19
    %98 = llvm.add %62, %13 : i64
    llvm.br ^bb17(%98 : i64)
  ^bb22:  // pred: ^bb17
    %99 = llvm.add %60, %13 : i64
    llvm.br ^bb15(%99 : i64)
  ^bb23:  // pred: ^bb15
    %100 = llvm.add %58, %13 : i64
    llvm.br ^bb13(%100 : i64)
  ^bb24:  // pred: ^bb13
    llvm.return %11 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  }
}

