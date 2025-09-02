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
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.mlir.constant(8 : index) : i64
    %17 = llvm.mlir.constant(8 : index) : i64
    %18 = llvm.mlir.constant(8 : index) : i64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.mlir.constant(64 : index) : i64
    %21 = llvm.mlir.constant(512 : index) : i64
    %22 = llvm.mlir.constant(512 : index) : i64
    %23 = llvm.mlir.zero : !llvm.ptr
    %24 = llvm.getelementptr %23[%22] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.mlir.constant(64 : index) : i64
    %27 = llvm.add %25, %26 : i64
    %28 = llvm.call @malloc(%27) : (i64) -> !llvm.ptr
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.sub %26, %30 : i64
    %32 = llvm.add %29, %31 : i64
    %33 = llvm.urem %32, %26 : i64
    %34 = llvm.sub %32, %33 : i64
    %35 = llvm.inttoptr %34 : i64 to !llvm.ptr
    %36 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %37 = llvm.insertvalue %28, %36[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %39 = llvm.mlir.constant(0 : index) : i64
    %40 = llvm.insertvalue %39, %38[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %41 = llvm.insertvalue %15, %40[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %42 = llvm.insertvalue %16, %41[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %43 = llvm.insertvalue %17, %42[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %44 = llvm.insertvalue %18, %43[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %45 = llvm.insertvalue %21, %44[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %46 = llvm.insertvalue %20, %45[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %47 = llvm.insertvalue %18, %46[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %48 = llvm.insertvalue %19, %47[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.br ^bb1(%14 : i64)
  ^bb1(%49: i64):  // 2 preds: ^bb0, ^bb11
    %50 = llvm.icmp "slt" %49, %13 : i64
    llvm.cond_br %50, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%14 : i64)
  ^bb3(%51: i64):  // 2 preds: ^bb2, ^bb10
    %52 = llvm.icmp "slt" %51, %12 : i64
    llvm.cond_br %52, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%14 : i64)
  ^bb5(%53: i64):  // 2 preds: ^bb4, ^bb9
    %54 = llvm.icmp "slt" %53, %12 : i64
    llvm.cond_br %54, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%14 : i64)
  ^bb7(%55: i64):  // 2 preds: ^bb6, ^bb8
    %56 = llvm.icmp "slt" %55, %12 : i64
    llvm.cond_br %56, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %57 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %58 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %59 = llvm.getelementptr %57[%58] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %60 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %61 = llvm.mul %49, %60 : i64
    %62 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %63 = llvm.mul %51, %62 : i64
    %64 = llvm.add %61, %63 : i64
    %65 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %66 = llvm.mul %53, %65 : i64
    %67 = llvm.add %64, %66 : i64
    %68 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %69 = llvm.mul %55, %68 : i64
    %70 = llvm.add %67, %69 : i64
    %71 = llvm.getelementptr %59[%70] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %72 = llvm.load %71 : !llvm.ptr -> f32
    %73 = llvm.fadd %72, %72 : f32
    %74 = llvm.extractvalue %48[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %75 = llvm.mlir.constant(512 : index) : i64
    %76 = llvm.mul %49, %75 : i64
    %77 = llvm.mlir.constant(64 : index) : i64
    %78 = llvm.mul %51, %77 : i64
    %79 = llvm.add %76, %78 : i64
    %80 = llvm.mlir.constant(8 : index) : i64
    %81 = llvm.mul %53, %80 : i64
    %82 = llvm.add %79, %81 : i64
    %83 = llvm.add %82, %55 : i64
    %84 = llvm.getelementptr %74[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %73, %84 : f32, !llvm.ptr
    %85 = llvm.add %55, %13 : i64
    llvm.br ^bb7(%85 : i64)
  ^bb9:  // pred: ^bb7
    %86 = llvm.add %53, %13 : i64
    llvm.br ^bb5(%86 : i64)
  ^bb10:  // pred: ^bb5
    %87 = llvm.add %51, %13 : i64
    llvm.br ^bb3(%87 : i64)
  ^bb11:  // pred: ^bb3
    %88 = llvm.add %49, %13 : i64
    llvm.br ^bb1(%88 : i64)
  ^bb12:  // pred: ^bb1
    llvm.return %48 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  }
}

