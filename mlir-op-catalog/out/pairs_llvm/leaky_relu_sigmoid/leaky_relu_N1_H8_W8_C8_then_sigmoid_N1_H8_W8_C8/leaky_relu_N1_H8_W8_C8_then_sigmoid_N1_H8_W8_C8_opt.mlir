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
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.mlir.constant(8 : index) : i64
    llvm.br ^bb1(%14 : i64)
  ^bb1(%17: i64):  // 2 preds: ^bb0, ^bb11
    %18 = llvm.icmp "slt" %17, %15 : i64
    llvm.cond_br %18, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%14 : i64)
  ^bb3(%19: i64):  // 2 preds: ^bb2, ^bb10
    %20 = llvm.icmp "slt" %19, %16 : i64
    llvm.cond_br %20, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%14 : i64)
  ^bb5(%21: i64):  // 2 preds: ^bb4, ^bb9
    %22 = llvm.icmp "slt" %21, %16 : i64
    llvm.cond_br %22, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%14 : i64)
  ^bb7(%23: i64):  // 2 preds: ^bb6, ^bb8
    %24 = llvm.icmp "slt" %23, %16 : i64
    llvm.cond_br %24, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %25 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %26 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %27 = llvm.getelementptr %25[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %28 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %29 = llvm.mul %17, %28 : i64
    %30 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %31 = llvm.mul %19, %30 : i64
    %32 = llvm.add %29, %31 : i64
    %33 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %34 = llvm.mul %21, %33 : i64
    %35 = llvm.add %32, %34 : i64
    %36 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %37 = llvm.mul %23, %36 : i64
    %38 = llvm.add %35, %37 : i64
    %39 = llvm.getelementptr %27[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %40 = llvm.load %39 : !llvm.ptr -> f32
    %41 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %42 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %43 = llvm.getelementptr %41[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %44 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %45 = llvm.mul %17, %44 : i64
    %46 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %47 = llvm.mul %19, %46 : i64
    %48 = llvm.add %45, %47 : i64
    %49 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %50 = llvm.mul %21, %49 : i64
    %51 = llvm.add %48, %50 : i64
    %52 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %53 = llvm.mul %23, %52 : i64
    %54 = llvm.add %51, %53 : i64
    %55 = llvm.getelementptr %43[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %40, %55 : f32, !llvm.ptr
    %56 = llvm.add %23, %15 : i64
    llvm.br ^bb7(%56 : i64)
  ^bb9:  // pred: ^bb7
    %57 = llvm.add %21, %15 : i64
    llvm.br ^bb5(%57 : i64)
  ^bb10:  // pred: ^bb5
    %58 = llvm.add %19, %15 : i64
    llvm.br ^bb3(%58 : i64)
  ^bb11:  // pred: ^bb3
    %59 = llvm.add %17, %15 : i64
    llvm.br ^bb1(%59 : i64)
  ^bb12:  // pred: ^bb1
    llvm.br ^bb13(%14 : i64)
  ^bb13(%60: i64):  // 2 preds: ^bb12, ^bb23
    %61 = llvm.icmp "slt" %60, %15 : i64
    llvm.cond_br %61, ^bb14, ^bb24
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%14 : i64)
  ^bb15(%62: i64):  // 2 preds: ^bb14, ^bb22
    %63 = llvm.icmp "slt" %62, %16 : i64
    llvm.cond_br %63, ^bb16, ^bb23
  ^bb16:  // pred: ^bb15
    llvm.br ^bb17(%14 : i64)
  ^bb17(%64: i64):  // 2 preds: ^bb16, ^bb21
    %65 = llvm.icmp "slt" %64, %16 : i64
    llvm.cond_br %65, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    llvm.br ^bb19(%14 : i64)
  ^bb19(%66: i64):  // 2 preds: ^bb18, ^bb20
    %67 = llvm.icmp "slt" %66, %16 : i64
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
    %84 = llvm.fmul %83, %13 : f32
    %85 = llvm.intr.exp(%84) : (f32) -> f32
    %86 = llvm.fadd %85, %12 : f32
    %87 = llvm.fdiv %12, %86 : f32
    %88 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %89 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %90 = llvm.getelementptr %88[%89] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %91 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %92 = llvm.mul %60, %91 : i64
    %93 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %94 = llvm.mul %62, %93 : i64
    %95 = llvm.add %92, %94 : i64
    %96 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %97 = llvm.mul %64, %96 : i64
    %98 = llvm.add %95, %97 : i64
    %99 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %100 = llvm.mul %66, %99 : i64
    %101 = llvm.add %98, %100 : i64
    %102 = llvm.getelementptr %90[%101] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %87, %102 : f32, !llvm.ptr
    %103 = llvm.add %66, %15 : i64
    llvm.br ^bb19(%103 : i64)
  ^bb21:  // pred: ^bb19
    %104 = llvm.add %64, %15 : i64
    llvm.br ^bb17(%104 : i64)
  ^bb22:  // pred: ^bb17
    %105 = llvm.add %62, %15 : i64
    llvm.br ^bb15(%105 : i64)
  ^bb23:  // pred: ^bb15
    %106 = llvm.add %60, %15 : i64
    llvm.br ^bb13(%106 : i64)
  ^bb24:  // pred: ^bb13
    llvm.return %11 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  }
}

