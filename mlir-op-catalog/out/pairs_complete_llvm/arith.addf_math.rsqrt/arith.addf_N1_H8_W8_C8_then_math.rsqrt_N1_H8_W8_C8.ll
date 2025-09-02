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

23:                                               ; preds = %92, %11
  %24 = phi i64 [ %93, %92 ], [ 0, %11 ]
  %25 = icmp slt i64 %24, 1
  br i1 %25, label %26, label %94

26:                                               ; preds = %23
  br label %27

27:                                               ; preds = %90, %26
  %28 = phi i64 [ %91, %90 ], [ 0, %26 ]
  %29 = icmp slt i64 %28, 8
  br i1 %29, label %30, label %92

30:                                               ; preds = %27
  br label %31

31:                                               ; preds = %88, %30
  %32 = phi i64 [ %89, %88 ], [ 0, %30 ]
  %33 = icmp slt i64 %32, 8
  br i1 %33, label %34, label %90

34:                                               ; preds = %31
  br label %35

35:                                               ; preds = %38, %34
  %36 = phi i64 [ %87, %38 ], [ 0, %34 ]
  %37 = icmp slt i64 %36, 8
  br i1 %37, label %38, label %88

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
  %55 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 1
  %56 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 2
  %57 = getelementptr float, ptr %55, i64 %56
  %58 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 0
  %59 = mul i64 %24, %58
  %60 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 1
  %61 = mul i64 %28, %60
  %62 = add i64 %59, %61
  %63 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 2
  %64 = mul i64 %32, %63
  %65 = add i64 %62, %64
  %66 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 3
  %67 = mul i64 %36, %66
  %68 = add i64 %65, %67
  %69 = getelementptr float, ptr %57, i64 %68
  %70 = load float, ptr %69, align 4
  %71 = fadd float %54, %70
  %72 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 1
  %73 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 2
  %74 = getelementptr float, ptr %72, i64 %73
  %75 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 0
  %76 = mul i64 %24, %75
  %77 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 1
  %78 = mul i64 %28, %77
  %79 = add i64 %76, %78
  %80 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 2
  %81 = mul i64 %32, %80
  %82 = add i64 %79, %81
  %83 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 3
  %84 = mul i64 %36, %83
  %85 = add i64 %82, %84
  %86 = getelementptr float, ptr %74, i64 %85
  store float %71, ptr %86, align 4
  %87 = add i64 %36, 1
  br label %35

88:                                               ; preds = %35
  %89 = add i64 %32, 1
  br label %31

90:                                               ; preds = %31
  %91 = add i64 %28, 1
  br label %27

92:                                               ; preds = %27
  %93 = add i64 %24, 1
  br label %23

94:                                               ; preds = %23
  br label %95

95:                                               ; preds = %148, %94
  %96 = phi i64 [ %149, %148 ], [ 0, %94 ]
  %97 = icmp slt i64 %96, 1
  br i1 %97, label %98, label %150

98:                                               ; preds = %95
  br label %99

99:                                               ; preds = %146, %98
  %100 = phi i64 [ %147, %146 ], [ 0, %98 ]
  %101 = icmp slt i64 %100, 8
  br i1 %101, label %102, label %148

102:                                              ; preds = %99
  br label %103

103:                                              ; preds = %144, %102
  %104 = phi i64 [ %145, %144 ], [ 0, %102 ]
  %105 = icmp slt i64 %104, 8
  br i1 %105, label %106, label %146

106:                                              ; preds = %103
  br label %107

107:                                              ; preds = %110, %106
  %108 = phi i64 [ %143, %110 ], [ 0, %106 ]
  %109 = icmp slt i64 %108, 8
  br i1 %109, label %110, label %144

110:                                              ; preds = %107
  %111 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 1
  %112 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 2
  %113 = getelementptr float, ptr %111, i64 %112
  %114 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 0
  %115 = mul i64 %96, %114
  %116 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 1
  %117 = mul i64 %100, %116
  %118 = add i64 %115, %117
  %119 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 2
  %120 = mul i64 %104, %119
  %121 = add i64 %118, %120
  %122 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 3
  %123 = mul i64 %108, %122
  %124 = add i64 %121, %123
  %125 = getelementptr float, ptr %113, i64 %124
  %126 = load float, ptr %125, align 4
  %127 = call float @llvm.sqrt.f32(float %126)
  %128 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 1
  %129 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 2
  %130 = getelementptr float, ptr %128, i64 %129
  %131 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 0
  %132 = mul i64 %96, %131
  %133 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 1
  %134 = mul i64 %100, %133
  %135 = add i64 %132, %134
  %136 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 2
  %137 = mul i64 %104, %136
  %138 = add i64 %135, %137
  %139 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 3
  %140 = mul i64 %108, %139
  %141 = add i64 %138, %140
  %142 = getelementptr float, ptr %130, i64 %141
  store float %127, ptr %142, align 4
  %143 = add i64 %108, 1
  br label %107

144:                                              ; preds = %107
  %145 = add i64 %104, 1
  br label %103

146:                                              ; preds = %103
  %147 = add i64 %100, 1
  br label %99

148:                                              ; preds = %99
  %149 = add i64 %96, 1
  br label %95

150:                                              ; preds = %95
  ret { ptr, ptr, i64, [4 x i64], [4 x i64] } %22
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
