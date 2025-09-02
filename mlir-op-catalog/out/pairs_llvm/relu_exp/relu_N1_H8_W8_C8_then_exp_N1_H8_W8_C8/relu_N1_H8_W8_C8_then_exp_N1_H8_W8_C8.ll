; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

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
  br label %23

23:                                               ; preds = %76, %11
  %24 = phi i64 [ %77, %76 ], [ 0, %11 ]
  %25 = icmp slt i64 %24, 1
  br i1 %25, label %26, label %78

26:                                               ; preds = %23
  br label %27

27:                                               ; preds = %74, %26
  %28 = phi i64 [ %75, %74 ], [ 0, %26 ]
  %29 = icmp slt i64 %28, 8
  br i1 %29, label %30, label %76

30:                                               ; preds = %27
  br label %31

31:                                               ; preds = %72, %30
  %32 = phi i64 [ %73, %72 ], [ 0, %30 ]
  %33 = icmp slt i64 %32, 8
  br i1 %33, label %34, label %74

34:                                               ; preds = %31
  br label %35

35:                                               ; preds = %38, %34
  %36 = phi i64 [ %71, %38 ], [ 0, %34 ]
  %37 = icmp slt i64 %36, 8
  br i1 %37, label %38, label %72

38:                                               ; preds = %35
  %39 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 1
  %40 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 2
  %41 = getelementptr float, ptr %39, i64 %40
  %42 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 0
  %43 = mul i64 %24, %42
  %44 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 1
  %45 = mul i64 %28, %44
  %46 = add i64 %43, %45
  %47 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 2
  %48 = mul i64 %32, %47
  %49 = add i64 %46, %48
  %50 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 3
  %51 = mul i64 %36, %50
  %52 = add i64 %49, %51
  %53 = getelementptr float, ptr %41, i64 %52
  %54 = load float, ptr %53, align 4
  %55 = call float @llvm.maximum.f32(float %54, float 0.000000e+00)
  %56 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 1
  %57 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 2
  %58 = getelementptr float, ptr %56, i64 %57
  %59 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 0
  %60 = mul i64 %24, %59
  %61 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 1
  %62 = mul i64 %28, %61
  %63 = add i64 %60, %62
  %64 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 2
  %65 = mul i64 %32, %64
  %66 = add i64 %63, %65
  %67 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 3
  %68 = mul i64 %36, %67
  %69 = add i64 %66, %68
  %70 = getelementptr float, ptr %58, i64 %69
  store float %55, ptr %70, align 4
  %71 = add i64 %36, 1
  br label %35

72:                                               ; preds = %35
  %73 = add i64 %32, 1
  br label %31

74:                                               ; preds = %31
  %75 = add i64 %28, 1
  br label %27

76:                                               ; preds = %27
  %77 = add i64 %24, 1
  br label %23

78:                                               ; preds = %23
  br label %79

79:                                               ; preds = %132, %78
  %80 = phi i64 [ %133, %132 ], [ 0, %78 ]
  %81 = icmp slt i64 %80, 1
  br i1 %81, label %82, label %134

82:                                               ; preds = %79
  br label %83

83:                                               ; preds = %130, %82
  %84 = phi i64 [ %131, %130 ], [ 0, %82 ]
  %85 = icmp slt i64 %84, 8
  br i1 %85, label %86, label %132

86:                                               ; preds = %83
  br label %87

87:                                               ; preds = %128, %86
  %88 = phi i64 [ %129, %128 ], [ 0, %86 ]
  %89 = icmp slt i64 %88, 8
  br i1 %89, label %90, label %130

90:                                               ; preds = %87
  br label %91

91:                                               ; preds = %94, %90
  %92 = phi i64 [ %127, %94 ], [ 0, %90 ]
  %93 = icmp slt i64 %92, 8
  br i1 %93, label %94, label %128

94:                                               ; preds = %91
  %95 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 1
  %96 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 2
  %97 = getelementptr float, ptr %95, i64 %96
  %98 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 0
  %99 = mul i64 %80, %98
  %100 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 1
  %101 = mul i64 %84, %100
  %102 = add i64 %99, %101
  %103 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 2
  %104 = mul i64 %88, %103
  %105 = add i64 %102, %104
  %106 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 3
  %107 = mul i64 %92, %106
  %108 = add i64 %105, %107
  %109 = getelementptr float, ptr %97, i64 %108
  %110 = load float, ptr %109, align 4
  %111 = call float @llvm.exp.f32(float %110)
  %112 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 1
  %113 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 2
  %114 = getelementptr float, ptr %112, i64 %113
  %115 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 0
  %116 = mul i64 %80, %115
  %117 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 1
  %118 = mul i64 %84, %117
  %119 = add i64 %116, %118
  %120 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 2
  %121 = mul i64 %88, %120
  %122 = add i64 %119, %121
  %123 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 3
  %124 = mul i64 %92, %123
  %125 = add i64 %122, %124
  %126 = getelementptr float, ptr %114, i64 %125
  store float %111, ptr %126, align 4
  %127 = add i64 %92, 1
  br label %91

128:                                              ; preds = %91
  %129 = add i64 %88, 1
  br label %87

130:                                              ; preds = %87
  %131 = add i64 %84, 1
  br label %83

132:                                              ; preds = %83
  %133 = add i64 %80, 1
  br label %79

134:                                              ; preds = %79
  ret { ptr, ptr, i64, [4 x i64], [4 x i64] } %22
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
