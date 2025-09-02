module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64) -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %1 = llvm.insertvalue %arg11, %0[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %2 = llvm.insertvalue %arg12, %1[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %3 = llvm.insertvalue %arg13, %2[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %4 = llvm.insertvalue %arg14, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %5 = llvm.insertvalue %arg18, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %6 = llvm.insertvalue %arg15, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %7 = llvm.insertvalue %arg19, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %8 = llvm.insertvalue %arg16, %7[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %9 = llvm.insertvalue %arg20, %8[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %10 = llvm.insertvalue %arg17, %9[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %11 = llvm.insertvalue %arg21, %10[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %12 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %13 = llvm.insertvalue %arg0, %12[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %14 = llvm.insertvalue %arg1, %13[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %15 = llvm.insertvalue %arg2, %14[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %16 = llvm.insertvalue %arg3, %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %17 = llvm.insertvalue %arg7, %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %18 = llvm.insertvalue %arg4, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %19 = llvm.insertvalue %arg8, %18[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %20 = llvm.insertvalue %arg5, %19[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %21 = llvm.insertvalue %arg9, %20[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %22 = llvm.insertvalue %arg6, %21[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %23 = llvm.insertvalue %arg10, %22[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.mlir.constant(8 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.mlir.constant(0 : index) : i64
    %27 = llvm.mlir.constant(1 : index) : i64
    %28 = llvm.mlir.constant(8 : index) : i64
    %29 = llvm.mlir.constant(8 : index) : i64
    %30 = llvm.mlir.constant(8 : index) : i64
    %31 = llvm.mlir.constant(1 : index) : i64
    %32 = llvm.mlir.constant(64 : index) : i64
    %33 = llvm.mlir.constant(512 : index) : i64
    %34 = llvm.mlir.constant(512 : index) : i64
    %35 = llvm.mlir.zero : !llvm.ptr
    %36 = llvm.getelementptr %35[%34] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %37 = llvm.ptrtoint %36 : !llvm.ptr to i64
    %38 = llvm.mlir.constant(64 : index) : i64
    %39 = llvm.add %37, %38 : i64
    %40 = llvm.call @malloc(%39) : (i64) -> !llvm.ptr
    %41 = llvm.ptrtoint %40 : !llvm.ptr to i64
    %42 = llvm.mlir.constant(1 : index) : i64
    %43 = llvm.sub %38, %42 : i64
    %44 = llvm.add %41, %43 : i64
    %45 = llvm.urem %44, %38 : i64
    %46 = llvm.sub %44, %45 : i64
    %47 = llvm.inttoptr %46 : i64 to !llvm.ptr
    %48 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %49 = llvm.insertvalue %40, %48[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %50 = llvm.insertvalue %47, %49[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %51 = llvm.mlir.constant(0 : index) : i64
    %52 = llvm.insertvalue %51, %50[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %53 = llvm.insertvalue %27, %52[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %54 = llvm.insertvalue %28, %53[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %55 = llvm.insertvalue %29, %54[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %56 = llvm.insertvalue %30, %55[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %57 = llvm.insertvalue %33, %56[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %58 = llvm.insertvalue %32, %57[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %59 = llvm.insertvalue %30, %58[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %60 = llvm.insertvalue %31, %59[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.br ^bb1(%26 : i64)
  ^bb1(%61: i64):  // 2 preds: ^bb0, ^bb11
    %62 = llvm.icmp "slt" %61, %25 : i64
    llvm.cond_br %62, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%26 : i64)
  ^bb3(%63: i64):  // 2 preds: ^bb2, ^bb10
    %64 = llvm.icmp "slt" %63, %24 : i64
    llvm.cond_br %64, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%26 : i64)
  ^bb5(%65: i64):  // 2 preds: ^bb4, ^bb9
    %66 = llvm.icmp "slt" %65, %24 : i64
    llvm.cond_br %66, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%26 : i64)
  ^bb7(%67: i64):  // 2 preds: ^bb6, ^bb8
    %68 = llvm.icmp "slt" %67, %24 : i64
    llvm.cond_br %68, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %69 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %70 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %71 = llvm.getelementptr %69[%70] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %72 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %73 = llvm.mul %61, %72 : i64
    %74 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %75 = llvm.mul %63, %74 : i64
    %76 = llvm.add %73, %75 : i64
    %77 = llvm.extractvalue %23[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %78 = llvm.mul %65, %77 : i64
    %79 = llvm.add %76, %78 : i64
    %80 = llvm.extractvalue %23[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %81 = llvm.mul %67, %80 : i64
    %82 = llvm.add %79, %81 : i64
    %83 = llvm.getelementptr %71[%82] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %84 = llvm.load %83 : !llvm.ptr -> f32
    %85 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %86 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %87 = llvm.getelementptr %85[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %88 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %89 = llvm.mul %61, %88 : i64
    %90 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %91 = llvm.mul %63, %90 : i64
    %92 = llvm.add %89, %91 : i64
    %93 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %94 = llvm.mul %65, %93 : i64
    %95 = llvm.add %92, %94 : i64
    %96 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %97 = llvm.mul %67, %96 : i64
    %98 = llvm.add %95, %97 : i64
    %99 = llvm.getelementptr %87[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %100 = llvm.load %99 : !llvm.ptr -> f32
    %101 = llvm.fsub %84, %100 : f32
    %102 = llvm.extractvalue %60[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %103 = llvm.mlir.constant(512 : index) : i64
    %104 = llvm.mul %61, %103 : i64
    %105 = llvm.mlir.constant(64 : index) : i64
    %106 = llvm.mul %63, %105 : i64
    %107 = llvm.add %104, %106 : i64
    %108 = llvm.mlir.constant(8 : index) : i64
    %109 = llvm.mul %65, %108 : i64
    %110 = llvm.add %107, %109 : i64
    %111 = llvm.add %110, %67 : i64
    %112 = llvm.getelementptr %102[%111] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %101, %112 : f32, !llvm.ptr
    %113 = llvm.add %67, %25 : i64
    llvm.br ^bb7(%113 : i64)
  ^bb9:  // pred: ^bb7
    %114 = llvm.add %65, %25 : i64
    llvm.br ^bb5(%114 : i64)
  ^bb10:  // pred: ^bb5
    %115 = llvm.add %63, %25 : i64
    llvm.br ^bb3(%115 : i64)
  ^bb11:  // pred: ^bb3
    %116 = llvm.add %61, %25 : i64
    llvm.br ^bb1(%116 : i64)
  ^bb12:  // pred: ^bb1
    llvm.return %60 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  }
}

