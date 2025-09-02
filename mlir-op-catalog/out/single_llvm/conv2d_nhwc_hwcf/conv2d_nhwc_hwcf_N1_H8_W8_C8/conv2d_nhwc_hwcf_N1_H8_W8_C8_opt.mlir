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
    %24 = llvm.mlir.constant(3 : index) : i64
    %25 = llvm.mlir.constant(8 : index) : i64
    %26 = llvm.mlir.constant(6 : index) : i64
    %27 = llvm.mlir.constant(1 : index) : i64
    %28 = llvm.mlir.constant(0 : index) : i64
    %29 = llvm.mlir.constant(1 : index) : i64
    %30 = llvm.mlir.constant(6 : index) : i64
    %31 = llvm.mlir.constant(6 : index) : i64
    %32 = llvm.mlir.constant(8 : index) : i64
    %33 = llvm.mlir.constant(1 : index) : i64
    %34 = llvm.mlir.constant(48 : index) : i64
    %35 = llvm.mlir.constant(288 : index) : i64
    %36 = llvm.mlir.constant(288 : index) : i64
    %37 = llvm.mlir.zero : !llvm.ptr
    %38 = llvm.getelementptr %37[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %39 = llvm.ptrtoint %38 : !llvm.ptr to i64
    %40 = llvm.mlir.constant(64 : index) : i64
    %41 = llvm.add %39, %40 : i64
    %42 = llvm.call @malloc(%41) : (i64) -> !llvm.ptr
    %43 = llvm.ptrtoint %42 : !llvm.ptr to i64
    %44 = llvm.mlir.constant(1 : index) : i64
    %45 = llvm.sub %40, %44 : i64
    %46 = llvm.add %43, %45 : i64
    %47 = llvm.urem %46, %40 : i64
    %48 = llvm.sub %46, %47 : i64
    %49 = llvm.inttoptr %48 : i64 to !llvm.ptr
    %50 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %51 = llvm.insertvalue %42, %50[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %52 = llvm.insertvalue %49, %51[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %53 = llvm.mlir.constant(0 : index) : i64
    %54 = llvm.insertvalue %53, %52[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %55 = llvm.insertvalue %29, %54[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %56 = llvm.insertvalue %30, %55[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %57 = llvm.insertvalue %31, %56[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %58 = llvm.insertvalue %32, %57[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %59 = llvm.insertvalue %35, %58[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %60 = llvm.insertvalue %34, %59[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %61 = llvm.insertvalue %32, %60[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %62 = llvm.insertvalue %33, %61[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.br ^bb1(%28 : i64)
  ^bb1(%63: i64):  // 2 preds: ^bb0, ^bb20
    %64 = llvm.icmp "slt" %63, %27 : i64
    llvm.cond_br %64, ^bb2, ^bb21
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%28 : i64)
  ^bb3(%65: i64):  // 2 preds: ^bb2, ^bb19
    %66 = llvm.icmp "slt" %65, %26 : i64
    llvm.cond_br %66, ^bb4, ^bb20
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%28 : i64)
  ^bb5(%67: i64):  // 2 preds: ^bb4, ^bb18
    %68 = llvm.icmp "slt" %67, %26 : i64
    llvm.cond_br %68, ^bb6, ^bb19
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%28 : i64)
  ^bb7(%69: i64):  // 2 preds: ^bb6, ^bb17
    %70 = llvm.icmp "slt" %69, %25 : i64
    llvm.cond_br %70, ^bb8, ^bb18
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%28 : i64)
  ^bb9(%71: i64):  // 2 preds: ^bb8, ^bb16
    %72 = llvm.icmp "slt" %71, %24 : i64
    llvm.cond_br %72, ^bb10, ^bb17
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%28 : i64)
  ^bb11(%73: i64):  // 2 preds: ^bb10, ^bb15
    %74 = llvm.icmp "slt" %73, %24 : i64
    llvm.cond_br %74, ^bb12, ^bb16
  ^bb12:  // pred: ^bb11
    llvm.br ^bb13(%28 : i64)
  ^bb13(%75: i64):  // 2 preds: ^bb12, ^bb14
    %76 = llvm.icmp "slt" %75, %25 : i64
    llvm.cond_br %76, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %77 = llvm.add %65, %71 : i64
    %78 = llvm.add %67, %73 : i64
    %79 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %80 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %81 = llvm.getelementptr %79[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %82 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %83 = llvm.mul %63, %82 : i64
    %84 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %85 = llvm.mul %77, %84 : i64
    %86 = llvm.add %83, %85 : i64
    %87 = llvm.extractvalue %23[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %88 = llvm.mul %78, %87 : i64
    %89 = llvm.add %86, %88 : i64
    %90 = llvm.extractvalue %23[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %91 = llvm.mul %75, %90 : i64
    %92 = llvm.add %89, %91 : i64
    %93 = llvm.getelementptr %81[%92] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %94 = llvm.load %93 : !llvm.ptr -> f32
    %95 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %96 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %97 = llvm.getelementptr %95[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %98 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %99 = llvm.mul %71, %98 : i64
    %100 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %101 = llvm.mul %73, %100 : i64
    %102 = llvm.add %99, %101 : i64
    %103 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %104 = llvm.mul %75, %103 : i64
    %105 = llvm.add %102, %104 : i64
    %106 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %107 = llvm.mul %69, %106 : i64
    %108 = llvm.add %105, %107 : i64
    %109 = llvm.getelementptr %97[%108] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %110 = llvm.load %109 : !llvm.ptr -> f32
    %111 = llvm.extractvalue %62[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %112 = llvm.mlir.constant(288 : index) : i64
    %113 = llvm.mul %63, %112 : i64
    %114 = llvm.mlir.constant(48 : index) : i64
    %115 = llvm.mul %65, %114 : i64
    %116 = llvm.add %113, %115 : i64
    %117 = llvm.mlir.constant(8 : index) : i64
    %118 = llvm.mul %67, %117 : i64
    %119 = llvm.add %116, %118 : i64
    %120 = llvm.add %119, %69 : i64
    %121 = llvm.getelementptr %111[%120] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %122 = llvm.load %121 : !llvm.ptr -> f32
    %123 = llvm.fmul %94, %110 : f32
    %124 = llvm.fadd %122, %123 : f32
    %125 = llvm.extractvalue %62[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %126 = llvm.mlir.constant(288 : index) : i64
    %127 = llvm.mul %63, %126 : i64
    %128 = llvm.mlir.constant(48 : index) : i64
    %129 = llvm.mul %65, %128 : i64
    %130 = llvm.add %127, %129 : i64
    %131 = llvm.mlir.constant(8 : index) : i64
    %132 = llvm.mul %67, %131 : i64
    %133 = llvm.add %130, %132 : i64
    %134 = llvm.add %133, %69 : i64
    %135 = llvm.getelementptr %125[%134] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %124, %135 : f32, !llvm.ptr
    %136 = llvm.add %75, %27 : i64
    llvm.br ^bb13(%136 : i64)
  ^bb15:  // pred: ^bb13
    %137 = llvm.add %73, %27 : i64
    llvm.br ^bb11(%137 : i64)
  ^bb16:  // pred: ^bb11
    %138 = llvm.add %71, %27 : i64
    llvm.br ^bb9(%138 : i64)
  ^bb17:  // pred: ^bb9
    %139 = llvm.add %69, %27 : i64
    llvm.br ^bb7(%139 : i64)
  ^bb18:  // pred: ^bb7
    %140 = llvm.add %67, %27 : i64
    llvm.br ^bb5(%140 : i64)
  ^bb19:  // pred: ^bb5
    %141 = llvm.add %65, %27 : i64
    llvm.br ^bb3(%141 : i64)
  ^bb20:  // pred: ^bb3
    %142 = llvm.add %63, %27 : i64
    llvm.br ^bb1(%142 : i64)
  ^bb21:  // pred: ^bb1
    llvm.return %62 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  }
}

