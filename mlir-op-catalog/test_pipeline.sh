#!/bin/bash
mlir-opt out/single/relu/relu_N1_H8_W8_C8.mlir \
  --one-shot-bufferize=bufferize-function-boundaries \
  --convert-linalg-to-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts | \
  mlir-translate --mlir-to-llvmir -o out/single/relu/relu_N1_H8_W8_C8.ll
