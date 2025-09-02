module {
  llvm.func @malloc(i64) -> !llvm.ptr
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
    %12 = llvm.mlir.constant(8 : index) : i64
    %13 = llvm.mlir.constant(4 : index) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.mlir.constant(4 : index) : i64
    %18 = llvm.mlir.constant(4 : index) : i64
    %19 = llvm.mlir.constant(8 : index) : i64
    %20 = llvm.mlir.constant(1 : index) : i64
    %21 = llvm.mlir.constant(32 : index) : i64
    %22 = llvm.mlir.constant(128 : index) : i64
    %23 = llvm.mlir.constant(128 : index) : i64
    %24 = llvm.mlir.zero : !llvm.ptr
    %25 = llvm.getelementptr %24[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.mlir.constant(64 : index) : i64
    %28 = llvm.add %26, %27 : i64
    %29 = llvm.call @malloc(%28) : (i64) -> !llvm.ptr
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.mlir.constant(1 : index) : i64
    %32 = llvm.sub %27, %31 : i64
    %33 = llvm.add %30, %32 : i64
    %34 = llvm.urem %33, %27 : i64
    %35 = llvm.sub %33, %34 : i64
    %36 = llvm.inttoptr %35 : i64 to !llvm.ptr
    %37 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %38 = llvm.insertvalue %29, %37[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %39 = llvm.insertvalue %36, %38[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %40 = llvm.mlir.constant(0 : index) : i64
    %41 = llvm.insertvalue %40, %39[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %42 = llvm.insertvalue %16, %41[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %43 = llvm.insertvalue %17, %42[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %44 = llvm.insertvalue %18, %43[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %45 = llvm.insertvalue %19, %44[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %46 = llvm.insertvalue %22, %45[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %47 = llvm.insertvalue %21, %46[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %48 = llvm.insertvalue %19, %47[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %49 = llvm.insertvalue %20, %48[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.br ^bb1(%15 : i64)
  ^bb1(%50: i64):  // 2 preds: ^bb0, ^bb11
    %51 = llvm.icmp "slt" %50, %14 : i64
    llvm.cond_br %51, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%15 : i64)
  ^bb3(%52: i64):  // 2 preds: ^bb2, ^bb10
    %53 = llvm.icmp "slt" %52, %13 : i64
    llvm.cond_br %53, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%15 : i64)
  ^bb5(%54: i64):  // 2 preds: ^bb4, ^bb9
    %55 = llvm.icmp "slt" %54, %13 : i64
    llvm.cond_br %55, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%15 : i64)
  ^bb7(%56: i64):  // 2 preds: ^bb6, ^bb8
    %57 = llvm.icmp "slt" %56, %12 : i64
    llvm.cond_br %57, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %58 = llvm.mlir.constant(2 : index) : i64
    %59 = llvm.mul %52, %58 overflow<nsw> : i64
    %60 = llvm.mlir.constant(2 : index) : i64
    %61 = llvm.mul %54, %60 overflow<nsw> : i64
    %62 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %63 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %64 = llvm.getelementptr %62[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %65 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %66 = llvm.mul %50, %65 : i64
    %67 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %68 = llvm.mul %59, %67 : i64
    %69 = llvm.add %66, %68 : i64
    %70 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %71 = llvm.mul %61, %70 : i64
    %72 = llvm.add %69, %71 : i64
    %73 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %74 = llvm.mul %56, %73 : i64
    %75 = llvm.add %72, %74 : i64
    %76 = llvm.getelementptr %64[%75] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %77 = llvm.load %76 : !llvm.ptr -> f32
    %78 = llvm.extractvalue %49[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %79 = llvm.mlir.constant(128 : index) : i64
    %80 = llvm.mul %50, %79 : i64
    %81 = llvm.mlir.constant(32 : index) : i64
    %82 = llvm.mul %52, %81 : i64
    %83 = llvm.add %80, %82 : i64
    %84 = llvm.mlir.constant(8 : index) : i64
    %85 = llvm.mul %54, %84 : i64
    %86 = llvm.add %83, %85 : i64
    %87 = llvm.add %86, %56 : i64
    %88 = llvm.getelementptr %78[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %77, %88 : f32, !llvm.ptr
    %89 = llvm.add %56, %14 : i64
    llvm.br ^bb7(%89 : i64)
  ^bb9:  // pred: ^bb7
    %90 = llvm.add %54, %14 : i64
    llvm.br ^bb5(%90 : i64)
  ^bb10:  // pred: ^bb5
    %91 = llvm.add %52, %14 : i64
    llvm.br ^bb3(%91 : i64)
  ^bb11:  // pred: ^bb3
    %92 = llvm.add %50, %14 : i64
    llvm.br ^bb1(%92 : i64)
  ^bb12:  // pred: ^bb1
    llvm.return %49 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  }
}

