; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10) {
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
  %23 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i64 8) to i64), i64 64))
  %24 = ptrtoint ptr %23 to i64
  %25 = add i64 %24, 63
  %26 = urem i64 %25, 64
  %27 = sub i64 %25, %26
  %28 = inttoptr i64 %27 to ptr
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %23, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, ptr %28, 1
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 0, 2
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 1, 3, 0
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 8, 3, 1
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 8, 4, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 1, 4, 1
  br label %36

36:                                               ; preds = %83, %11
  %37 = phi i64 [ %84, %83 ], [ 0, %11 ]
  %38 = icmp slt i64 %37, 1
  br i1 %38, label %39, label %85

39:                                               ; preds = %36
  br label %40

40:                                               ; preds = %81, %39
  %41 = phi i64 [ %82, %81 ], [ 0, %39 ]
  %42 = icmp slt i64 %41, 8
  br i1 %42, label %43, label %83

43:                                               ; preds = %40
  br label %44

44:                                               ; preds = %79, %43
  %45 = phi i64 [ %80, %79 ], [ 0, %43 ]
  %46 = icmp slt i64 %45, 8
  br i1 %46, label %47, label %81

47:                                               ; preds = %44
  br label %48

48:                                               ; preds = %51, %47
  %49 = phi i64 [ %78, %51 ], [ 0, %47 ]
  %50 = icmp slt i64 %49, 8
  br i1 %50, label %51, label %79

51:                                               ; preds = %48
  %52 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 1
  %53 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 2
  %54 = getelementptr float, ptr %52, i64 %53
  %55 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 0
  %56 = mul i64 %37, %55
  %57 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 1
  %58 = mul i64 %41, %57
  %59 = add i64 %56, %58
  %60 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 2
  %61 = mul i64 %45, %60
  %62 = add i64 %59, %61
  %63 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 3
  %64 = mul i64 %49, %63
  %65 = add i64 %62, %64
  %66 = getelementptr float, ptr %54, i64 %65
  %67 = load float, ptr %66, align 4
  %68 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %69 = mul i64 %37, 8
  %70 = add i64 %69, %49
  %71 = getelementptr float, ptr %68, i64 %70
  %72 = load float, ptr %71, align 4
  %73 = fadd float %67, %72
  %74 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %75 = mul i64 %37, 8
  %76 = add i64 %75, %49
  %77 = getelementptr float, ptr %74, i64 %76
  store float %73, ptr %77, align 4
  %78 = add i64 %49, 1
  br label %48

79:                                               ; preds = %48
  %80 = add i64 %45, 1
  br label %44

81:                                               ; preds = %44
  %82 = add i64 %41, 1
  br label %40

83:                                               ; preds = %40
  %84 = add i64 %37, 1
  br label %36

85:                                               ; preds = %36
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %35
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
