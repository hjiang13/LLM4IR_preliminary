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

23:                                               ; preds = %75, %11
  %24 = phi i64 [ %76, %75 ], [ 0, %11 ]
  %25 = icmp slt i64 %24, 1
  br i1 %25, label %26, label %77

26:                                               ; preds = %23
  br label %27

27:                                               ; preds = %73, %26
  %28 = phi i64 [ %74, %73 ], [ 0, %26 ]
  %29 = icmp slt i64 %28, 8
  br i1 %29, label %30, label %75

30:                                               ; preds = %27
  br label %31

31:                                               ; preds = %71, %30
  %32 = phi i64 [ %72, %71 ], [ 0, %30 ]
  %33 = icmp slt i64 %32, 8
  br i1 %33, label %34, label %73

34:                                               ; preds = %31
  br label %35

35:                                               ; preds = %38, %34
  %36 = phi i64 [ %70, %38 ], [ 0, %34 ]
  %37 = icmp slt i64 %36, 8
  br i1 %37, label %38, label %71

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
  store float %54, ptr %69, align 4
  %70 = add i64 %36, 1
  br label %35

71:                                               ; preds = %35
  %72 = add i64 %32, 1
  br label %31

73:                                               ; preds = %31
  %74 = add i64 %28, 1
  br label %27

75:                                               ; preds = %27
  %76 = add i64 %24, 1
  br label %23

77:                                               ; preds = %23
  br label %78

78:                                               ; preds = %147, %77
  %79 = phi i64 [ %148, %147 ], [ 0, %77 ]
  %80 = icmp slt i64 %79, 1
  br i1 %80, label %81, label %149

81:                                               ; preds = %78
  br label %82

82:                                               ; preds = %145, %81
  %83 = phi i64 [ %146, %145 ], [ 0, %81 ]
  %84 = icmp slt i64 %83, 8
  br i1 %84, label %85, label %147

85:                                               ; preds = %82
  br label %86

86:                                               ; preds = %143, %85
  %87 = phi i64 [ %144, %143 ], [ 0, %85 ]
  %88 = icmp slt i64 %87, 8
  br i1 %88, label %89, label %145

89:                                               ; preds = %86
  br label %90

90:                                               ; preds = %93, %89
  %91 = phi i64 [ %142, %93 ], [ 0, %89 ]
  %92 = icmp slt i64 %91, 8
  br i1 %92, label %93, label %143

93:                                               ; preds = %90
  %94 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 1
  %95 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 2
  %96 = getelementptr float, ptr %94, i64 %95
  %97 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 0
  %98 = mul i64 %79, %97
  %99 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 1
  %100 = mul i64 %83, %99
  %101 = add i64 %98, %100
  %102 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 2
  %103 = mul i64 %87, %102
  %104 = add i64 %101, %103
  %105 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 3
  %106 = mul i64 %91, %105
  %107 = add i64 %104, %106
  %108 = getelementptr float, ptr %96, i64 %107
  %109 = load float, ptr %108, align 4
  %110 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 1
  %111 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 2
  %112 = getelementptr float, ptr %110, i64 %111
  %113 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 0
  %114 = mul i64 %79, %113
  %115 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 1
  %116 = mul i64 %83, %115
  %117 = add i64 %114, %116
  %118 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 2
  %119 = mul i64 %87, %118
  %120 = add i64 %117, %119
  %121 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 3
  %122 = mul i64 %91, %121
  %123 = add i64 %120, %122
  %124 = getelementptr float, ptr %112, i64 %123
  %125 = load float, ptr %124, align 4
  %126 = fdiv float %109, %125
  %127 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 1
  %128 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 2
  %129 = getelementptr float, ptr %127, i64 %128
  %130 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 0
  %131 = mul i64 %79, %130
  %132 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 1
  %133 = mul i64 %83, %132
  %134 = add i64 %131, %133
  %135 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 2
  %136 = mul i64 %87, %135
  %137 = add i64 %134, %136
  %138 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %22, 4, 3
  %139 = mul i64 %91, %138
  %140 = add i64 %137, %139
  %141 = getelementptr float, ptr %129, i64 %140
  store float %126, ptr %141, align 4
  %142 = add i64 %91, 1
  br label %90

143:                                              ; preds = %90
  %144 = add i64 %87, 1
  br label %86

145:                                              ; preds = %86
  %146 = add i64 %83, 1
  br label %82

147:                                              ; preds = %82
  %148 = add i64 %79, 1
  br label %78

149:                                              ; preds = %78
  ret { ptr, ptr, i64, [4 x i64], [4 x i64] } %22
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
