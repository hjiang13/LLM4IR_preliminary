	.file	"LLVMDialectModule"
	.text
	.globl	main                            # -- Begin function main
	.p2align	4
	.type	main,@function
main:                                   # @mlir_add
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$72, %rsp
	.cfi_def_cfa_offset 128
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rcx, 16(%rsp)                  # 8-byte Spill
	movq	%rdx, 8(%rsp)                   # 8-byte Spill
	movq	%rdi, 40(%rsp)                  # 8-byte Spill
	movq	256(%rsp), %r12
	movq	248(%rsp), %r13
	movq	192(%rsp), %r14
	movq	184(%rsp), %rbp
	movq	168(%rsp), %r15
	movq	160(%rsp), %rbx
	movl	$2112, %edi                     # imm = 0x840
	callq	malloc@PLT
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	leaq	63(%rax), %rdi
	andq	$-64, %rdi
	movq	$0, 24(%rsp)                    # 8-byte Folded Spill
	movq	16(%rsp), %rax                  # 8-byte Reload
	movq	8(%rsp), %rcx                   # 8-byte Reload
	leaq	(%rcx,%rax,4), %r10
	leaq	(%rbp,%r14,4), %r11
	xorl	%eax, %eax
	jmp	.LBB0_1
	.p2align	4
.LBB0_11:                               #   in Loop: Header=BB0_1 Depth=1
	movq	48(%rsp), %rax                  # 8-byte Reload
	incq	%rax
	addq	$2048, 24(%rsp)                 # 8-byte Folded Spill
                                        # imm = 0x800
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
                                        #       Child Loop BB0_5 Depth 3
                                        #         Child Loop BB0_8 Depth 4
	testq	%rax, %rax
	jg	.LBB0_12
# %bb.2:                                #   in Loop: Header=BB0_1 Depth=1
	movq	%rax, %rcx
	imulq	144(%rsp), %rcx
	movq	%rcx, 64(%rsp)                  # 8-byte Spill
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	imulq	232(%rsp), %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	movq	24(%rsp), %r9                   # 8-byte Reload
	xorl	%eax, %eax
	jmp	.LBB0_3
	.p2align	4
.LBB0_10:                               #   in Loop: Header=BB0_3 Depth=2
	movq	8(%rsp), %rax                   # 8-byte Reload
	incq	%rax
	movq	16(%rsp), %r9                   # 8-byte Reload
	addq	$256, %r9                       # imm = 0x100
.LBB0_3:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_5 Depth 3
                                        #         Child Loop BB0_8 Depth 4
	cmpq	$7, %rax
	jg	.LBB0_11
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=2
	movq	%rax, %rdx
	imulq	152(%rsp), %rdx
	addq	64(%rsp), %rdx                  # 8-byte Folded Reload
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	imulq	240(%rsp), %rax
	addq	56(%rsp), %rax                  # 8-byte Folded Reload
	movq	%r9, 16(%rsp)                   # 8-byte Spill
	xorl	%r8d, %r8d
	jmp	.LBB0_5
	.p2align	4
.LBB0_9:                                #   in Loop: Header=BB0_5 Depth=3
	incq	%r8
	addq	$32, %r9
.LBB0_5:                                #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_8 Depth 4
	cmpq	$7, %r8
	jg	.LBB0_10
# %bb.6:                                #   in Loop: Header=BB0_5 Depth=3
	movq	%r8, %r14
	imulq	%rbx, %r14
	movq	%r8, %rsi
	imulq	%r13, %rsi
	xorl	%ecx, %ecx
	cmpq	$7, %rcx
	jg	.LBB0_9
	.p2align	4
.LBB0_8:                                #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        #       Parent Loop BB0_5 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	movq	%rcx, %rbp
	imulq	%r15, %rbp
	addq	%r14, %rbp
	addq	%rdx, %rbp
	movss	(%r10,%rbp,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movq	%rcx, %rbp
	imulq	%r12, %rbp
	addq	%rsi, %rbp
	addq	%rax, %rbp
	addss	(%r11,%rbp,4), %xmm0
	leaq	(%rdi,%r9), %rbp
	movss	%xmm0, (%rbp,%rcx,4)
	incq	%rcx
	cmpq	$7, %rcx
	jle	.LBB0_8
	jmp	.LBB0_9
.LBB0_12:
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	32(%rsp), %rcx                  # 8-byte Reload
	movq	%rcx, (%rax)
	movq	%rdi, 8(%rax)
	movl	$1, %ecx
	movq	%rcx, 24(%rax)
	movl	$8, %edx
	movq	%rdx, 32(%rax)
	movq	%rdx, 40(%rax)
	movq	%rdx, 48(%rax)
	movl	$512, %esi                      # imm = 0x200
	movq	%rsi, 56(%rax)
	movl	$64, %esi
	movq	%rsi, 64(%rax)
	movq	%rdx, 72(%rax)
	movq	%rcx, 80(%rax)
	movq	$0, 16(%rax)
	addq	$72, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
