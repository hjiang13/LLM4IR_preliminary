; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [4 x i64], [4 x i64] } @main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10) {
  %12 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %0, 0
  %13 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %12, ptr %1, 1
  %14 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %13, i64 %2, 2
  %15 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %14, i64 %3, 3, 0
  %16 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, i64 %7, 4, 0
  %17 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %16, i64 %4, 3, 1
  %18 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %17, i64 %8, 4, 1
  %19 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %18, i64 %5, 3, 2
  %20 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %19, i64 %9, 4, 2
  %21 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %20, i64 %6, 3, 3
  %22 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %21, i64 %10, 4, 3
  %23 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i64 512) to i64), i64 64))
  %24 = ptrtoint ptr %23 to i64
  %25 = add i64 %24, 63
  %26 = urem i64 %25, 64
  %27 = sub i64 %25, %26
  %28 = inttoptr i64 %27 to ptr
  %29 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %23, 0
  %30 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %29, ptr %28, 1
  %31 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %30, i64 0, 2
  %32 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %31, i64 1, 3, 0
  %33 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %32, i64 8, 3, 1
  %34 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %33, i64 8, 3, 2
  %35 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %34, i64 8, 3, 3
  %36 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %35, i64 512, 4, 0
  %37 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %36, i64 64, 4, 1
  %38 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %37, i64 8, 4, 2
  %39 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %38, i64 1, 4, 3
  br label %40

40:                                               ; preds = %86, %11
  %41 = phi i64 [ %87, %86 ], [ 0, %11 ]
  %42 = icmp slt i64 %41, 1
  br i1 %42, label %43, label %88

43:                                               ; preds = %40
  br label %44

44:                                               ; preds = %84, %43
  %45 = phi i64 [ %85, %84 ], [ 0, %43 ]
  %46 = icmp slt i64 %45, 8
  br i1 %46, label %47, label %86

47:                                               ; preds = %44
  br label %48

48:                                               ; preds = %82, %47
  %49 = phi i64 [ %83, %82 ], [ 0, %47 ]
  %50 = icmp slt i64 %49, 8
  br i1 %50, label %51, label %84

51:                                               ; preds = %48
  br label %52

52:                                               ; preds = %55, %51
  %53 = phi i64 [ %81, %55 ], [ 0, %51 ]
  %54 = icmp slt i64 %53, 8
  br i1 %54, label %55, label %82

55:                                               ; preds = %52
  %56 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 1
  %57 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 2
  %58 = getelementptr float, ptr %56, i64 %57
  %59 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 0
  %60 = mul i64 %41, %59
  %61 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 1
  %62 = mul i64 %45, %61
  %63 = add i64 %60, %62
  %64 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 2
  %65 = mul i64 %49, %64
  %66 = add i64 %63, %65
  %67 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 3
  %68 = mul i64 %53, %67
  %69 = add i64 %66, %68
  %70 = getelementptr float, ptr %58, i64 %69
  %71 = load float, ptr %70, align 4
  %72 = fadd float %71, %71
  %73 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %39, 1
  %74 = mul i64 %41, 512
  %75 = mul i64 %45, 64
  %76 = add i64 %74, %75
  %77 = mul i64 %49, 8
  %78 = add i64 %76, %77
  %79 = add i64 %78, %53
  %80 = getelementptr float, ptr %73, i64 %79
  store float %72, ptr %80, align 4
  %81 = add i64 %53, 1
  br label %52

82:                                               ; preds = %52
  %83 = add i64 %49, 1
  br label %48

84:                                               ; preds = %48
  %85 = add i64 %45, 1
  br label %44

86:                                               ; preds = %44
  %87 = add i64 %41, 1
  br label %40

88:                                               ; preds = %40
  ret { ptr, ptr, i64, [4 x i64], [4 x i64] } %39
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
