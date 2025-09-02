; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [4 x i64], [4 x i64] } @main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21) {
  %23 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %11, 0
  %24 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %23, ptr %12, 1
  %25 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %24, i64 %13, 2
  %26 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %25, i64 %14, 3, 0
  %27 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %26, i64 %18, 4, 0
  %28 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %27, i64 %15, 3, 1
  %29 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %28, i64 %19, 4, 1
  %30 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %29, i64 %16, 3, 2
  %31 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %30, i64 %20, 4, 2
  %32 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %31, i64 %17, 3, 3
  %33 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %32, i64 %21, 4, 3
  %34 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %0, 0
  %35 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %34, ptr %1, 1
  %36 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %35, i64 %2, 2
  %37 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %36, i64 %3, 3, 0
  %38 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %37, i64 %7, 4, 0
  %39 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %38, i64 %4, 3, 1
  %40 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %39, i64 %8, 4, 1
  %41 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %40, i64 %5, 3, 2
  %42 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %41, i64 %9, 4, 2
  %43 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %42, i64 %6, 3, 3
  %44 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %43, i64 %10, 4, 3
  %45 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i64 512) to i64), i64 64))
  %46 = ptrtoint ptr %45 to i64
  %47 = add i64 %46, 63
  %48 = urem i64 %47, 64
  %49 = sub i64 %47, %48
  %50 = inttoptr i64 %49 to ptr
  %51 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %45, 0
  %52 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %51, ptr %50, 1
  %53 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %52, i64 0, 2
  %54 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %53, i64 1, 3, 0
  %55 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %54, i64 8, 3, 1
  %56 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %55, i64 8, 3, 2
  %57 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %56, i64 8, 3, 3
  %58 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %57, i64 512, 4, 0
  %59 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %58, i64 64, 4, 1
  %60 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %59, i64 8, 4, 2
  %61 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %60, i64 1, 4, 3
  br label %62

62:                                               ; preds = %124, %22
  %63 = phi i64 [ %125, %124 ], [ 0, %22 ]
  %64 = icmp slt i64 %63, 1
  br i1 %64, label %65, label %126

65:                                               ; preds = %62
  br label %66

66:                                               ; preds = %122, %65
  %67 = phi i64 [ %123, %122 ], [ 0, %65 ]
  %68 = icmp slt i64 %67, 8
  br i1 %68, label %69, label %124

69:                                               ; preds = %66
  br label %70

70:                                               ; preds = %120, %69
  %71 = phi i64 [ %121, %120 ], [ 0, %69 ]
  %72 = icmp slt i64 %71, 8
  br i1 %72, label %73, label %122

73:                                               ; preds = %70
  br label %74

74:                                               ; preds = %77, %73
  %75 = phi i64 [ %119, %77 ], [ 0, %73 ]
  %76 = icmp slt i64 %75, 8
  br i1 %76, label %77, label %120

77:                                               ; preds = %74
  %78 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %44, 1
  %79 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %44, 2
  %80 = getelementptr float, ptr %78, i64 %79
  %81 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %44, 4, 0
  %82 = mul i64 %63, %81
  %83 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %44, 4, 1
  %84 = mul i64 %67, %83
  %85 = add i64 %82, %84
  %86 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %44, 4, 2
  %87 = mul i64 %71, %86
  %88 = add i64 %85, %87
  %89 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %44, 4, 3
  %90 = mul i64 %75, %89
  %91 = add i64 %88, %90
  %92 = getelementptr float, ptr %80, i64 %91
  %93 = load float, ptr %92, align 4
  %94 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %33, 1
  %95 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %33, 2
  %96 = getelementptr float, ptr %94, i64 %95
  %97 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %33, 4, 0
  %98 = mul i64 %63, %97
  %99 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %33, 4, 1
  %100 = mul i64 %67, %99
  %101 = add i64 %98, %100
  %102 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %33, 4, 2
  %103 = mul i64 %71, %102
  %104 = add i64 %101, %103
  %105 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %33, 4, 3
  %106 = mul i64 %75, %105
  %107 = add i64 %104, %106
  %108 = getelementptr float, ptr %96, i64 %107
  %109 = load float, ptr %108, align 4
  %110 = fdiv float %93, %109
  %111 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %61, 1
  %112 = mul i64 %63, 512
  %113 = mul i64 %67, 64
  %114 = add i64 %112, %113
  %115 = mul i64 %71, 8
  %116 = add i64 %114, %115
  %117 = add i64 %116, %75
  %118 = getelementptr float, ptr %111, i64 %117
  store float %110, ptr %118, align 4
  %119 = add i64 %75, 1
  br label %74

120:                                              ; preds = %74
  %121 = add i64 %71, 1
  br label %70

122:                                              ; preds = %70
  %123 = add i64 %67, 1
  br label %66

124:                                              ; preds = %66
  %125 = add i64 %63, 1
  br label %62

126:                                              ; preds = %62
  ret { ptr, ptr, i64, [4 x i64], [4 x i64] } %61
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
