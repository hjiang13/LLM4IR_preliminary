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
  %45 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i64 288) to i64), i64 64))
  %46 = ptrtoint ptr %45 to i64
  %47 = add i64 %46, 63
  %48 = urem i64 %47, 64
  %49 = sub i64 %47, %48
  %50 = inttoptr i64 %49 to ptr
  %51 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %45, 0
  %52 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %51, ptr %50, 1
  %53 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %52, i64 0, 2
  %54 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %53, i64 1, 3, 0
  %55 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %54, i64 6, 3, 1
  %56 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %55, i64 6, 3, 2
  %57 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %56, i64 8, 3, 3
  %58 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %57, i64 288, 4, 0
  %59 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %58, i64 48, 4, 1
  %60 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %59, i64 8, 4, 2
  %61 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %60, i64 1, 4, 3
  br label %62

62:                                               ; preds = %154, %22
  %63 = phi i64 [ %155, %154 ], [ 0, %22 ]
  %64 = icmp slt i64 %63, 1
  br i1 %64, label %65, label %156

65:                                               ; preds = %62
  br label %66

66:                                               ; preds = %152, %65
  %67 = phi i64 [ %153, %152 ], [ 0, %65 ]
  %68 = icmp slt i64 %67, 6
  br i1 %68, label %69, label %154

69:                                               ; preds = %66
  br label %70

70:                                               ; preds = %150, %69
  %71 = phi i64 [ %151, %150 ], [ 0, %69 ]
  %72 = icmp slt i64 %71, 6
  br i1 %72, label %73, label %152

73:                                               ; preds = %70
  br label %74

74:                                               ; preds = %148, %73
  %75 = phi i64 [ %149, %148 ], [ 0, %73 ]
  %76 = icmp slt i64 %75, 8
  br i1 %76, label %77, label %150

77:                                               ; preds = %74
  br label %78

78:                                               ; preds = %146, %77
  %79 = phi i64 [ %147, %146 ], [ 0, %77 ]
  %80 = icmp slt i64 %79, 3
  br i1 %80, label %81, label %148

81:                                               ; preds = %78
  br label %82

82:                                               ; preds = %144, %81
  %83 = phi i64 [ %145, %144 ], [ 0, %81 ]
  %84 = icmp slt i64 %83, 3
  br i1 %84, label %85, label %146

85:                                               ; preds = %82
  br label %86

86:                                               ; preds = %89, %85
  %87 = phi i64 [ %143, %89 ], [ 0, %85 ]
  %88 = icmp slt i64 %87, 8
  br i1 %88, label %89, label %144

89:                                               ; preds = %86
  %90 = add i64 %67, %79
  %91 = add i64 %71, %83
  %92 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %44, 1
  %93 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %44, 2
  %94 = getelementptr float, ptr %92, i64 %93
  %95 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %44, 4, 0
  %96 = mul i64 %63, %95
  %97 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %44, 4, 1
  %98 = mul i64 %90, %97
  %99 = add i64 %96, %98
  %100 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %44, 4, 2
  %101 = mul i64 %91, %100
  %102 = add i64 %99, %101
  %103 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %44, 4, 3
  %104 = mul i64 %87, %103
  %105 = add i64 %102, %104
  %106 = getelementptr float, ptr %94, i64 %105
  %107 = load float, ptr %106, align 4
  %108 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %33, 1
  %109 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %33, 2
  %110 = getelementptr float, ptr %108, i64 %109
  %111 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %33, 4, 0
  %112 = mul i64 %79, %111
  %113 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %33, 4, 1
  %114 = mul i64 %83, %113
  %115 = add i64 %112, %114
  %116 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %33, 4, 2
  %117 = mul i64 %87, %116
  %118 = add i64 %115, %117
  %119 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %33, 4, 3
  %120 = mul i64 %75, %119
  %121 = add i64 %118, %120
  %122 = getelementptr float, ptr %110, i64 %121
  %123 = load float, ptr %122, align 4
  %124 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %61, 1
  %125 = mul i64 %63, 288
  %126 = mul i64 %67, 48
  %127 = add i64 %125, %126
  %128 = mul i64 %71, 8
  %129 = add i64 %127, %128
  %130 = add i64 %129, %75
  %131 = getelementptr float, ptr %124, i64 %130
  %132 = load float, ptr %131, align 4
  %133 = fmul float %107, %123
  %134 = fadd float %132, %133
  %135 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %61, 1
  %136 = mul i64 %63, 288
  %137 = mul i64 %67, 48
  %138 = add i64 %136, %137
  %139 = mul i64 %71, 8
  %140 = add i64 %138, %139
  %141 = add i64 %140, %75
  %142 = getelementptr float, ptr %135, i64 %141
  store float %134, ptr %142, align 4
  %143 = add i64 %87, 1
  br label %86

144:                                              ; preds = %86
  %145 = add i64 %83, 1
  br label %82

146:                                              ; preds = %82
  %147 = add i64 %79, 1
  br label %78

148:                                              ; preds = %78
  %149 = add i64 %75, 1
  br label %74

150:                                              ; preds = %74
  %151 = add i64 %71, 1
  br label %70

152:                                              ; preds = %70
  %153 = add i64 %67, 1
  br label %66

154:                                              ; preds = %66
  %155 = add i64 %63, 1
  br label %62

156:                                              ; preds = %62
  ret { ptr, ptr, i64, [4 x i64], [4 x i64] } %61
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
