module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
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
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.mlir.constant(8 : index) : i64
    %19 = llvm.mlir.zero : !llvm.ptr
    %20 = llvm.getelementptr %19[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %21 = llvm.ptrtoint %20 : !llvm.ptr to i64
    %22 = llvm.mlir.constant(64 : index) : i64
    %23 = llvm.add %21, %22 : i64
    %24 = llvm.call @malloc(%23) : (i64) -> !llvm.ptr
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.sub %22, %26 : i64
    %28 = llvm.add %25, %27 : i64
    %29 = llvm.urem %28, %22 : i64
    %30 = llvm.sub %28, %29 : i64
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    %32 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.insertvalue %24, %32[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %31, %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.mlir.constant(0 : index) : i64
    %36 = llvm.insertvalue %35, %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %15, %36[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %16, %37[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %16, %38[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.insertvalue %17, %39[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%14 : i64)
  ^bb1(%41: i64):  // 2 preds: ^bb0, ^bb11
    %42 = llvm.icmp "slt" %41, %13 : i64
    llvm.cond_br %42, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%14 : i64)
  ^bb3(%43: i64):  // 2 preds: ^bb2, ^bb10
    %44 = llvm.icmp "slt" %43, %12 : i64
    llvm.cond_br %44, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%14 : i64)
  ^bb5(%45: i64):  // 2 preds: ^bb4, ^bb9
    %46 = llvm.icmp "slt" %45, %12 : i64
    llvm.cond_br %46, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%14 : i64)
  ^bb7(%47: i64):  // 2 preds: ^bb6, ^bb8
    %48 = llvm.icmp "slt" %47, %12 : i64
    llvm.cond_br %48, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %49 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %50 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %52 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %53 = llvm.mul %41, %52 : i64
    %54 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %55 = llvm.mul %43, %54 : i64
    %56 = llvm.add %53, %55 : i64
    %57 = llvm.extractvalue %11[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %58 = llvm.mul %45, %57 : i64
    %59 = llvm.add %56, %58 : i64
    %60 = llvm.extractvalue %11[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %61 = llvm.mul %47, %60 : i64
    %62 = llvm.add %59, %61 : i64
    %63 = llvm.getelementptr %51[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %64 = llvm.load %63 : !llvm.ptr -> f32
    %65 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mlir.constant(8 : index) : i64
    %67 = llvm.mul %41, %66 : i64
    %68 = llvm.add %67, %47 : i64
    %69 = llvm.getelementptr %65[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %70 = llvm.load %69 : !llvm.ptr -> f32
    %71 = llvm.fadd %64, %70 : f32
    %72 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.mlir.constant(8 : index) : i64
    %74 = llvm.mul %41, %73 : i64
    %75 = llvm.add %74, %47 : i64
    %76 = llvm.getelementptr %72[%75] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %71, %76 : f32, !llvm.ptr
    %77 = llvm.add %47, %13 : i64
    llvm.br ^bb7(%77 : i64)
  ^bb9:  // pred: ^bb7
    %78 = llvm.add %45, %13 : i64
    llvm.br ^bb5(%78 : i64)
  ^bb10:  // pred: ^bb5
    %79 = llvm.add %43, %13 : i64
    llvm.br ^bb3(%79 : i64)
  ^bb11:  // pred: ^bb3
    %80 = llvm.add %41, %13 : i64
    llvm.br ^bb1(%80 : i64)
  ^bb12:  // pred: ^bb1
    llvm.return %40 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
}

