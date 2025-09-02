; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) {
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %7, 0
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, ptr %8, 1
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, i64 %9, 2
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, i64 %10, 3, 0
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 %12, 4, 0
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 %11, 3, 1
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 %13, 4, 1
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %1, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %2, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %3, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %5, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %4, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %6, 4, 1
  %29 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i64 256) to i64), i64 64))
  %30 = ptrtoint ptr %29 to i64
  %31 = add i64 %30, 63
  %32 = urem i64 %31, 64
  %33 = sub i64 %31, %32
  %34 = inttoptr i64 %33 to ptr
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %29, 0
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, ptr %34, 1
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 0, 2
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 16, 3, 0
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 16, 3, 1
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 16, 4, 0
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 1, 4, 1
  br label %42

42:                                               ; preds = %88, %14
  %43 = phi i64 [ %89, %88 ], [ 0, %14 ]
  %44 = icmp slt i64 %43, 16
  br i1 %44, label %45, label %90

45:                                               ; preds = %42
  br label %46

46:                                               ; preds = %86, %45
  %47 = phi i64 [ %87, %86 ], [ 0, %45 ]
  %48 = icmp slt i64 %47, 16
  br i1 %48, label %49, label %88

49:                                               ; preds = %46
  br label %50

50:                                               ; preds = %53, %49
  %51 = phi i64 [ %85, %53 ], [ 0, %49 ]
  %52 = icmp slt i64 %51, 16
  br i1 %52, label %53, label %86

53:                                               ; preds = %50
  %54 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %55 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 2
  %56 = getelementptr float, ptr %54, i64 %55
  %57 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 0
  %58 = mul i64 %43, %57
  %59 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 1
  %60 = mul i64 %51, %59
  %61 = add i64 %58, %60
  %62 = getelementptr float, ptr %56, i64 %61
  %63 = load float, ptr %62, align 4
  %64 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %65 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 2
  %66 = getelementptr float, ptr %64, i64 %65
  %67 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 0
  %68 = mul i64 %51, %67
  %69 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 1
  %70 = mul i64 %47, %69
  %71 = add i64 %68, %70
  %72 = getelementptr float, ptr %66, i64 %71
  %73 = load float, ptr %72, align 4
  %74 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  %75 = mul i64 %43, 16
  %76 = add i64 %75, %47
  %77 = getelementptr float, ptr %74, i64 %76
  %78 = load float, ptr %77, align 4
  %79 = fmul float %63, %73
  %80 = fadd float %78, %79
  %81 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  %82 = mul i64 %43, 16
  %83 = add i64 %82, %47
  %84 = getelementptr float, ptr %81, i64 %83
  store float %80, ptr %84, align 4
  %85 = add i64 %51, 1
  br label %50

86:                                               ; preds = %50
  %87 = add i64 %47, 1
  br label %46

88:                                               ; preds = %46
  %89 = add i64 %43, 1
  br label %42

90:                                               ; preds = %42
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %41
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
