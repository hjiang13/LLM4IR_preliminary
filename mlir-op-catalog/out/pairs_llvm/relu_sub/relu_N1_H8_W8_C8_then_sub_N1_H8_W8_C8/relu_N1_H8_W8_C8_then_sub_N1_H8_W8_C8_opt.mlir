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
    %40 = llvm.intr.maximum(%39, %12) : (f32, f32) -> f32
    %41 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %42 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %43 = llvm.getelementptr %41[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %44 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %45 = llvm.mul %16, %44 : i64
    %46 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %47 = llvm.mul %18, %46 : i64
    %48 = llvm.add %45, %47 : i64
    %49 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %50 = llvm.mul %20, %49 : i64
    %51 = llvm.add %48, %50 : i64
    %52 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %53 = llvm.mul %22, %52 : i64
    %54 = llvm.add %51, %53 : i64
    %55 = llvm.getelementptr %43[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %40, %55 : f32, !llvm.ptr
    %56 = llvm.add %22, %14 : i64
    llvm.br ^bb7(%56 : i64)
  ^bb9:  // pred: ^bb7
    %57 = llvm.add %20, %14 : i64
    llvm.br ^bb5(%57 : i64)
  ^bb10:  // pred: ^bb5
    %58 = llvm.add %18, %14 : i64
    llvm.br ^bb3(%58 : i64)
  ^bb11:  // pred: ^bb3
    %59 = llvm.add %16, %14 : i64
    llvm.br ^bb1(%59 : i64)
  ^bb12:  // pred: ^bb1
    llvm.br ^bb13(%13 : i64)
  ^bb13(%60: i64):  // 2 preds: ^bb12, ^bb23
    %61 = llvm.icmp "slt" %60, %14 : i64
    llvm.cond_br %61, ^bb14, ^bb24
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%13 : i64)
  ^bb15(%62: i64):  // 2 preds: ^bb14, ^bb22
    %63 = llvm.icmp "slt" %62, %15 : i64
    llvm.cond_br %63, ^bb16, ^bb23
  ^bb16:  // pred: ^bb15
    llvm.br ^bb17(%13 : i64)
  ^bb17(%64: i64):  // 2 preds: ^bb16, ^bb21
    %65 = llvm.icmp "slt" %64, %15 : i64
    llvm.cond_br %65, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    llvm.br ^bb19(%13 : i64)
  ^bb19(%66: i64):  // 2 preds: ^bb18, ^bb20
    %67 = llvm.icmp "slt" %66, %15 : i64
    llvm.cond_br %67, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %68 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %69 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %70 = llvm.getelementptr %68[%69] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %71 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %72 = llvm.mul %60, %71 : i64
    %73 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %74 = llvm.mul %62, %73 : i64
    %75 = llvm.add %72, %74 : i64
    %76 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %77 = llvm.mul %64, %76 : i64
    %78 = llvm.add %75, %77 : i64
    %79 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %80 = llvm.mul %66, %79 : i64
    %81 = llvm.add %78, %80 : i64
    %82 = llvm.getelementptr %70[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %83 = llvm.load %82 : !llvm.ptr -> f32
    %84 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %85 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %86 = llvm.getelementptr %84[%85] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %87 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %88 = llvm.mul %60, %87 : i64
    %89 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %90 = llvm.mul %62, %89 : i64
    %91 = llvm.add %88, %90 : i64
    %92 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %93 = llvm.mul %64, %92 : i64
    %94 = llvm.add %91, %93 : i64
    %95 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %96 = llvm.mul %66, %95 : i64
    %97 = llvm.add %94, %96 : i64
    %98 = llvm.getelementptr %86[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %99 = llvm.load %98 : !llvm.ptr -> f32
    %100 = llvm.fsub %83, %99 : f32
    %101 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %102 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %103 = llvm.getelementptr %101[%102] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %104 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %105 = llvm.mul %60, %104 : i64
    %106 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %107 = llvm.mul %62, %106 : i64
    %108 = llvm.add %105, %107 : i64
    %109 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %110 = llvm.mul %64, %109 : i64
    %111 = llvm.add %108, %110 : i64
    %112 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %113 = llvm.mul %66, %112 : i64
    %114 = llvm.add %111, %113 : i64
    %115 = llvm.getelementptr %103[%114] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %100, %115 : f32, !llvm.ptr
    %116 = llvm.add %66, %14 : i64
    llvm.br ^bb19(%116 : i64)
  ^bb21:  // pred: ^bb19
    %117 = llvm.add %64, %14 : i64
    llvm.br ^bb17(%117 : i64)
  ^bb22:  // pred: ^bb17
    %118 = llvm.add %62, %14 : i64
    llvm.br ^bb15(%118 : i64)
  ^bb23:  // pred: ^bb15
    %119 = llvm.add %60, %14 : i64
    llvm.br ^bb13(%119 : i64)
  ^bb24:  // pred: ^bb13
    llvm.return %11 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  }
}

