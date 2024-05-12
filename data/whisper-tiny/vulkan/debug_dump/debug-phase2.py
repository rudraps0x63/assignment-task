# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def NT_matmul(A: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), B: T.Buffer((T.int64(384), T.int64(384)), "float32"), NT_matmul: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(384), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]

    @T.prim_func(private=True)
    def NT_matmul10(var_A: T.handle, B: T.Buffer((T.int64(384), T.int64(384)), "float32"), var_NT_matmul: T.handle):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), seq_len, T.int64(384)))
        NT_matmul = T.match_buffer(var_NT_matmul, (T.int64(1), seq_len, T.int64(384)))
        # with T.block("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(384), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]

    @T.prim_func(private=True)
    def NT_matmul15(var_A: T.handle, B: T.Buffer((T.int64(51865), T.int64(384)), "float32"), var_NT_matmul: T.handle):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), seq_len, T.int64(384)))
        NT_matmul = T.match_buffer(var_NT_matmul, (T.int64(1), seq_len, T.int64(51865)))
        # with T.block("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(51865), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]

    @T.prim_func(private=True)
    def NT_matmul2(A: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32"), B: T.Buffer((T.int64(384), T.int64(384)), "float32"), NT_matmul: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1500), T.int64(384), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]

    @T.prim_func(private=True)
    def NT_matmul6(A: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), B: T.Buffer((T.int64(51865), T.int64(384)), "float32"), NT_matmul: T.Buffer((T.int64(1), T.int64(1), T.int64(51865)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(51865), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]

    @T.prim_func(private=True)
    def add9(var_A: T.handle, var_B: T.handle, var_T_add: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), seq_len, T.int64(384)))
        B = T.match_buffer(var_B, (T.int64(1), seq_len, T.int64(384)))
        T_add = T.match_buffer(var_T_add, (T.int64(1), seq_len, T.int64(384)))
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], B[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2] + B[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul10_add10(p_layer_norm22: T.handle, model_decoder_layers_0_self_attn_v_proj_weight2: T.Buffer((T.int64(384), T.int64(384)), "float32"), model_decoder_layers_0_self_attn_v_proj_bias2: T.Buffer((T.int64(384),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        layer_norm22 = T.match_buffer(p_layer_norm22, (T.int64(1), seq_len, T.int64(384)))
        var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(384)))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(384)))
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(384), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(layer_norm22[v_i0, v_i1, v_k], model_decoder_layers_0_self_attn_v_proj_weight2[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + layer_norm22[v_i0, v_i1, v_k] * model_decoder_layers_0_self_attn_v_proj_weight2[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], model_decoder_layers_0_self_attn_v_proj_bias2[v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + model_decoder_layers_0_self_attn_v_proj_bias2[v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul10_add10_add9(p_reshape65: T.handle, model_decoder_layers_0_self_attn_out_proj_weight2: T.Buffer((T.int64(384), T.int64(384)), "float32"), model_decoder_layers_0_self_attn_out_proj_bias2: T.Buffer((T.int64(384),), "float32"), p_add74: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        reshape65 = T.match_buffer(p_reshape65, (T.int64(1), seq_len, T.int64(384)))
        add74 = T.match_buffer(p_add74, (T.int64(1), seq_len, T.int64(384)))
        var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(384)))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(384)))
        var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), seq_len, T.int64(384)))
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(384), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(reshape65[v_i0, v_i1, v_k], model_decoder_layers_0_self_attn_out_proj_weight2[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + reshape65[v_i0, v_i1, v_k] * model_decoder_layers_0_self_attn_out_proj_weight2[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], model_decoder_layers_0_self_attn_out_proj_bias2[v_ax2])
                T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + model_decoder_layers_0_self_attn_out_proj_bias2[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(384)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(add74[v_ax0, v_ax1, v_ax2], var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = add74[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul10_add10_multiply2(p_layer_norm22: T.handle, model_decoder_layers_0_self_attn_q_proj_weight2: T.Buffer((T.int64(384), T.int64(384)), "float32"), model_decoder_layers_0_self_attn_q_proj_bias2: T.Buffer((T.int64(384),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        layer_norm22 = T.match_buffer(p_layer_norm22, (T.int64(1), seq_len, T.int64(384)))
        var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(384)))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(384)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(384)))
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(384), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(layer_norm22[v_i0, v_i1, v_k], model_decoder_layers_0_self_attn_q_proj_weight2[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + layer_norm22[v_i0, v_i1, v_k] * model_decoder_layers_0_self_attn_q_proj_weight2[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], model_decoder_layers_0_self_attn_q_proj_bias2[v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + model_decoder_layers_0_self_attn_q_proj_bias2[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(384)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float32(0.125)

    @T.prim_func(private=True)
    def fused_NT_matmul11_maximum4_minimum4(p_permute_dims129: T.handle, p_permute_dims130: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        permute_dims129 = T.match_buffer(p_permute_dims129, (T.int64(1), T.int64(6), seq_len, T.int64(64)))
        total_seq_len = T.int64()
        permute_dims130 = T.match_buffer(p_permute_dims130, (T.int64(1), T.int64(6), total_seq_len, T.int64(64)))
        var_T_minimum_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(6), seq_len, total_seq_len))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(6), seq_len, total_seq_len))
        var_T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(6), seq_len, total_seq_len))
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(6), seq_len, total_seq_len, T.int64(64)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(permute_dims129[v_i0, v_i1, v_i2, v_k], permute_dims130[v_i0, v_i1, v_i3, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + permute_dims129[v_i0, v_i1, v_i2, v_k] * permute_dims130[v_i0, v_i1, v_i3, v_k]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(6), seq_len, total_seq_len):
            with T.block("T_maximum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(-3.4028234663852886e+38))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(6), seq_len, total_seq_len):
            with T.block("T_minimum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(3.4028234663852886e+38))

    @T.prim_func(private=True)
    def fused_NT_matmul12_maximum5_minimum5(p_permute_dims136: T.handle, permute_dims137: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        permute_dims136 = T.match_buffer(p_permute_dims136, (T.int64(1), T.int64(6), seq_len, T.int64(64)))
        var_T_minimum_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(6), seq_len, T.int64(1500)))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(6), seq_len, T.int64(1500)))
        var_T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(6), seq_len, T.int64(1500)))
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(6), seq_len, T.int64(1500), T.int64(64)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(permute_dims136[v_i0, v_i1, v_i2, v_k], permute_dims137[v_i0, v_i1, v_i3, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + permute_dims136[v_i0, v_i1, v_i2, v_k] * permute_dims137[v_i0, v_i1, v_i3, v_k]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(6), seq_len, T.int64(1500)):
            with T.block("T_maximum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(-3.4028234663852886e+38))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(6), seq_len, T.int64(1500)):
            with T.block("T_minimum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(3.4028234663852886e+38))

    @T.prim_func(private=True)
    def fused_NT_matmul13_add11_gelu4(p_layer_norm24: T.handle, model_decoder_layers_0_fc1_weight2: T.Buffer((T.int64(1536), T.int64(384)), "float32"), model_decoder_layers_0_fc1_bias2: T.Buffer((T.int64(1536),), "float32"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        layer_norm24 = T.match_buffer(p_layer_norm24, (T.int64(1), seq_len, T.int64(384)))
        var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(1536)))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(1536)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(1536)))
        T_multiply = T.alloc_buffer((T.int64(1), seq_len, T.int64(1536)))
        compute = T.alloc_buffer((T.int64(1), seq_len, T.int64(1536)))
        T_multiply_1 = T.alloc_buffer((T.int64(1), seq_len, T.int64(1536)))
        T_add = T.alloc_buffer((T.int64(1), seq_len, T.int64(1536)))
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(1536), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(layer_norm24[v_i0, v_i1, v_k], model_decoder_layers_0_fc1_weight2[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + layer_norm24[v_i0, v_i1, v_k] * model_decoder_layers_0_fc1_weight2[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(1536)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], model_decoder_layers_0_fc1_bias2[v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + model_decoder_layers_0_fc1_bias2[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(1536)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float32(0.70710678118654757)
        for i0, i1, i2 in T.grid(T.int64(1), seq_len, T.int64(1536)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_multiply[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.erf(T_multiply[v_i0, v_i1, v_i2])
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(1536)):
            with T.block("T_multiply_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(compute[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T_multiply_1[v_ax0, v_ax1, v_ax2] = compute[v_ax0, v_ax1, v_ax2] * T.float32(0.5)
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(1536)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = T.float32(0.5) + T_multiply_1[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(1536)):
            with T.block("T_multiply_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul14_add10_add9(p_gelu10: T.handle, model_decoder_layers_0_fc2_weight2: T.Buffer((T.int64(384), T.int64(1536)), "float32"), model_decoder_layers_0_fc2_bias2: T.Buffer((T.int64(384),), "float32"), p_add81: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        gelu10 = T.match_buffer(p_gelu10, (T.int64(1), seq_len, T.int64(1536)))
        add81 = T.match_buffer(p_add81, (T.int64(1), seq_len, T.int64(384)))
        var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(384)))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(384)))
        var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), seq_len, T.int64(384)))
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(384), T.int64(1536)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(gelu10[v_i0, v_i1, v_k], model_decoder_layers_0_fc2_weight2[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + gelu10[v_i0, v_i1, v_k] * model_decoder_layers_0_fc2_weight2[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], model_decoder_layers_0_fc2_bias2[v_ax2])
                T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + model_decoder_layers_0_fc2_bias2[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(384)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(add81[v_ax0, v_ax1, v_ax2], var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = add81[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul1_maximum2_minimum2(permute_dims48: T.Buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(64)), "float32"), p_permute_dims49: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        total_seq_len = T.int64()
        permute_dims49 = T.match_buffer(p_permute_dims49, (T.int64(1), T.int64(6), total_seq_len, T.int64(64)))
        var_T_minimum_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(6), T.int64(1), total_seq_len))
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(6), T.int64(1), total_seq_len))
        var_T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(6), T.int64(1), total_seq_len))
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(6), T.int64(1), total_seq_len, T.int64(64)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(permute_dims48[v_i0, v_i1, v_i2, v_k], permute_dims49[v_i0, v_i1, v_i3, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + permute_dims48[v_i0, v_i1, v_i2, v_k] * permute_dims49[v_i0, v_i1, v_i3, v_k]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(6), T.int64(1), total_seq_len):
            with T.block("T_maximum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(-3.4028234663852886e+38))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(6), T.int64(1), total_seq_len):
            with T.block("T_minimum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(3.4028234663852886e+38))

    @T.prim_func(private=True)
    def fused_NT_matmul2_add3(layer_norm: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32"), model_encoder_layers_0_self_attn_v_proj_weight: T.Buffer((T.int64(384), T.int64(384)), "float32"), model_encoder_layers_0_self_attn_v_proj_bias: T.Buffer((T.int64(384),), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(384)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1500), T.int64(384), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(layer_norm[v_i0, v_i1, v_k], model_encoder_layers_0_self_attn_v_proj_weight[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + layer_norm[v_i0, v_i1, v_k] * model_encoder_layers_0_self_attn_v_proj_weight[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], model_encoder_layers_0_self_attn_v_proj_bias[v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + model_encoder_layers_0_self_attn_v_proj_bias[v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul2_add3_add4(reshape3: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32"), model_encoder_layers_0_self_attn_out_proj_weight: T.Buffer((T.int64(384), T.int64(384)), "float32"), model_encoder_layers_0_self_attn_out_proj_bias: T.Buffer((T.int64(384),), "float32"), add: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(384)))
        var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(384)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1500), T.int64(384), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(reshape3[v_i0, v_i1, v_k], model_encoder_layers_0_self_attn_out_proj_weight[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + reshape3[v_i0, v_i1, v_k] * model_encoder_layers_0_self_attn_out_proj_weight[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], model_encoder_layers_0_self_attn_out_proj_bias[v_ax2])
                T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + model_encoder_layers_0_self_attn_out_proj_bias[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(384)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(add[v_ax0, v_ax1, v_ax2], var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = add[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul2_add3_multiply(layer_norm: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32"), model_encoder_layers_0_self_attn_q_proj_weight: T.Buffer((T.int64(384), T.int64(384)), "float32"), model_encoder_layers_0_self_attn_q_proj_bias: T.Buffer((T.int64(384),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(384)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(384)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1500), T.int64(384), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(layer_norm[v_i0, v_i1, v_k], model_encoder_layers_0_self_attn_q_proj_weight[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + layer_norm[v_i0, v_i1, v_k] * model_encoder_layers_0_self_attn_q_proj_weight[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], model_encoder_layers_0_self_attn_q_proj_bias[v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + model_encoder_layers_0_self_attn_q_proj_bias[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(384)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float32(0.125)

    @T.prim_func(private=True)
    def fused_NT_matmul3_maximum3_minimum3(permute_dims57: T.Buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(64)), "float32"), permute_dims58: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32"), var_T_minimum_intermediate: T.Buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(1500)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(1500)))
        var_T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(1500)))
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(6), T.int64(1), T.int64(1500), T.int64(64)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(permute_dims57[v_i0, v_i1, v_i2, v_k], permute_dims58[v_i0, v_i1, v_i3, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + permute_dims57[v_i0, v_i1, v_i2, v_k] * permute_dims58[v_i0, v_i1, v_i3, v_k]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(6), T.int64(1), T.int64(1500)):
            with T.block("T_maximum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(-3.4028234663852886e+38))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(6), T.int64(1), T.int64(1500)):
            with T.block("T_minimum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(3.4028234663852886e+38))

    @T.prim_func(private=True)
    def fused_NT_matmul4_add8_gelu3(layer_norm11: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), model_decoder_layers_0_fc1_weight1: T.Buffer((T.int64(1536), T.int64(384)), "float32"), model_decoder_layers_0_fc1_bias1: T.Buffer((T.int64(1536),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(1536)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(1536)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(1536)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(1536)))
        compute = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(1536)))
        T_multiply_1 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(1536)))
        T_add = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(1536)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(1536), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(layer_norm11[v_i0, v_i1, v_k], model_decoder_layers_0_fc1_weight1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + layer_norm11[v_i0, v_i1, v_k] * model_decoder_layers_0_fc1_weight1[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1536)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], model_decoder_layers_0_fc1_bias1[v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + model_decoder_layers_0_fc1_bias1[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1536)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float32(0.70710678118654757)
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(1536)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_multiply[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.erf(T_multiply[v_i0, v_i1, v_i2])
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1536)):
            with T.block("T_multiply_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(compute[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T_multiply_1[v_ax0, v_ax1, v_ax2] = compute[v_ax0, v_ax1, v_ax2] * T.float32(0.5)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1536)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = T.float32(0.5) + T_multiply_1[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1536)):
            with T.block("T_multiply_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul5_add7_add6(gelu6: T.Buffer((T.int64(1), T.int64(1), T.int64(1536)), "float32"), model_decoder_layers_0_fc2_weight1: T.Buffer((T.int64(384), T.int64(1536)), "float32"), model_decoder_layers_0_fc2_bias1: T.Buffer((T.int64(384),), "float32"), add37: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(384)))
        var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(384)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(384), T.int64(1536)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(gelu6[v_i0, v_i1, v_k], model_decoder_layers_0_fc2_weight1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + gelu6[v_i0, v_i1, v_k] * model_decoder_layers_0_fc2_weight1[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], model_decoder_layers_0_fc2_bias1[v_ax2])
                T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + model_decoder_layers_0_fc2_bias1[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(384)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(add37[v_ax0, v_ax1, v_ax2], var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = add37[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul7_maximum_minimum(permute_dims4: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32"), permute_dims5: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32"), var_T_minimum_intermediate: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(1500)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(1500)))
        var_T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(1500)))
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(6), T.int64(1500), T.int64(1500), T.int64(64)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(permute_dims4[v_i0, v_i1, v_i2, v_k], permute_dims5[v_i0, v_i1, v_i3, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + permute_dims4[v_i0, v_i1, v_i2, v_k] * permute_dims5[v_i0, v_i1, v_i3, v_k]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(6), T.int64(1500), T.int64(1500)):
            with T.block("T_maximum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(-3.4028234663852886e+38))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(6), T.int64(1500), T.int64(1500)):
            with T.block("T_minimum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(3.4028234663852886e+38))

    @T.prim_func(private=True)
    def fused_NT_matmul8_add5_gelu2(layer_norm1: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32"), model_encoder_layers_0_fc1_weight: T.Buffer((T.int64(1536), T.int64(384)), "float32"), model_encoder_layers_0_fc1_bias: T.Buffer((T.int64(1536),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(1500), T.int64(1536)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(1536)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(1536)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(1536)))
        compute = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(1536)))
        T_multiply_1 = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(1536)))
        T_add = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(1536)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1500), T.int64(1536), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(layer_norm1[v_i0, v_i1, v_k], model_encoder_layers_0_fc1_weight[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + layer_norm1[v_i0, v_i1, v_k] * model_encoder_layers_0_fc1_weight[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(1536)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], model_encoder_layers_0_fc1_bias[v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + model_encoder_layers_0_fc1_bias[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(1536)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float32(0.70710678118654757)
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1500), T.int64(1536)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_multiply[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.erf(T_multiply[v_i0, v_i1, v_i2])
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(1536)):
            with T.block("T_multiply_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(compute[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T_multiply_1[v_ax0, v_ax1, v_ax2] = compute[v_ax0, v_ax1, v_ax2] * T.float32(0.5)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(1536)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = T.float32(0.5) + T_multiply_1[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(1536)):
            with T.block("T_multiply_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul9_add3_add4_maximum1_minimum1(gelu2: T.Buffer((T.int64(1), T.int64(1500), T.int64(1536)), "float32"), model_encoder_layers_0_fc2_weight: T.Buffer((T.int64(384), T.int64(1536)), "float32"), model_encoder_layers_0_fc2_bias: T.Buffer((T.int64(384),), "float32"), add4: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32"), var_T_minimum_intermediate: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(384)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(384)))
        var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(384)))
        var_T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(384)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1500), T.int64(384), T.int64(1536)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(gelu2[v_i0, v_i1, v_k], model_encoder_layers_0_fc2_weight[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + gelu2[v_i0, v_i1, v_k] * model_encoder_layers_0_fc2_weight[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], model_encoder_layers_0_fc2_bias[v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + model_encoder_layers_0_fc2_bias[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(384)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(add4[v_ax0, v_ax1, v_ax2], var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = add4[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(384)):
            with T.block("T_maximum"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2] = T.max(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2], T.float32(-3.4028234663852886e+38))
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(384)):
            with T.block("T_minimum"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2], var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2] = T.min(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2], var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])

    @T.prim_func(private=True)
    def fused_NT_matmul_add7(layer_norm9: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), model_decoder_layers_0_self_attn_v_proj_weight1: T.Buffer((T.int64(384), T.int64(384)), "float32"), model_decoder_layers_0_self_attn_v_proj_bias1: T.Buffer((T.int64(384),), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(384)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(384), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(layer_norm9[v_i0, v_i1, v_k], model_decoder_layers_0_self_attn_v_proj_weight1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + layer_norm9[v_i0, v_i1, v_k] * model_decoder_layers_0_self_attn_v_proj_weight1[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], model_decoder_layers_0_self_attn_v_proj_bias1[v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + model_decoder_layers_0_self_attn_v_proj_bias1[v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul_add7_add6(reshape23: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), model_decoder_layers_0_self_attn_out_proj_weight1: T.Buffer((T.int64(384), T.int64(384)), "float32"), model_decoder_layers_0_self_attn_out_proj_bias1: T.Buffer((T.int64(384),), "float32"), add29: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(384)))
        var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(384)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(384), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(reshape23[v_i0, v_i1, v_k], model_decoder_layers_0_self_attn_out_proj_weight1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + reshape23[v_i0, v_i1, v_k] * model_decoder_layers_0_self_attn_out_proj_weight1[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], model_decoder_layers_0_self_attn_out_proj_bias1[v_ax2])
                T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + model_decoder_layers_0_self_attn_out_proj_bias1[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(384)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(add29[v_ax0, v_ax1, v_ax2], var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = add29[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_NT_matmul_add7_multiply1(layer_norm9: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), model_decoder_layers_0_self_attn_q_proj_weight1: T.Buffer((T.int64(384), T.int64(384)), "float32"), model_decoder_layers_0_self_attn_q_proj_bias1: T.Buffer((T.int64(384),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(384)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(384)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(384), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(layer_norm9[v_i0, v_i1, v_k], model_decoder_layers_0_self_attn_q_proj_weight1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + layer_norm9[v_i0, v_i1, v_k] * model_decoder_layers_0_self_attn_q_proj_weight1[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], model_decoder_layers_0_self_attn_q_proj_bias1[v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + model_decoder_layers_0_self_attn_q_proj_bias1[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(384)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float32(0.125)

    @T.prim_func(private=True)
    def fused_conv1d1_add1_gelu1(gelu: T.Buffer((T.int64(1), T.int64(384), T.int64(3000)), "float32"), model_encoder_conv2_weight: T.Buffer((T.int64(384), T.int64(384), T.int64(3)), "float32"), lv20: T.Buffer((T.int64(1), T.int64(384), T.int64(1)), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(384), T.int64(1500)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(3002)))
        var_conv1d_ncw_intermediate = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(1500)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(1500)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(1500)))
        compute = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(1500)))
        T_multiply_1 = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(1500)))
        T_add = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(1500)))
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(384), T.int64(3002)):
            with T.block("pad_temp"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(gelu[v_i0, v_i1, v_i2 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2])
                pad_temp[v_i0, v_i1, v_i2] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(3001), gelu[v_i0, v_i1, v_i2 - T.int64(1)], T.float32(0))
        for nn, ff, yy, rc, ry in T.grid(T.int64(1), T.int64(384), T.int64(1500), T.int64(384), T.int64(3)):
            with T.block("conv1d_ncw"):
                v_nn, v_ff, v_yy, v_rc, v_ry = T.axis.remap("SSSRR", [nn, ff, yy, rc, ry])
                T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry], model_encoder_conv2_weight[v_ff, v_rc, v_ry])
                T.writes(var_conv1d_ncw_intermediate[v_nn, v_ff, v_yy])
                with T.init():
                    var_conv1d_ncw_intermediate[v_nn, v_ff, v_yy] = T.float32(0)
                var_conv1d_ncw_intermediate[v_nn, v_ff, v_yy] = var_conv1d_ncw_intermediate[v_nn, v_ff, v_yy] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry] * model_encoder_conv2_weight[v_ff, v_rc, v_ry]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(1500)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_conv1d_ncw_intermediate[v_ax0, v_ax1, v_ax2], lv20[v_ax0, v_ax1, T.int64(0)])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_conv1d_ncw_intermediate[v_ax0, v_ax1, v_ax2] + lv20[v_ax0, v_ax1, T.int64(0)]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(1500)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float32(0.70710678118654757)
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(384), T.int64(1500)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_multiply[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.erf(T_multiply[v_i0, v_i1, v_i2])
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(1500)):
            with T.block("T_multiply_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(compute[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T_multiply_1[v_ax0, v_ax1, v_ax2] = compute[v_ax0, v_ax1, v_ax2] * T.float32(0.5)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(1500)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = T.float32(0.5) + T_multiply_1[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(1500)):
            with T.block("T_multiply_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_conv1d_add_gelu(input_ids: T.Buffer((T.int64(1), T.int64(80), T.int64(3000)), "float32"), model_encoder_conv1_weight: T.Buffer((T.int64(384), T.int64(80), T.int64(3)), "float32"), lv18: T.Buffer((T.int64(1), T.int64(384), T.int64(1)), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(384), T.int64(3000)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(80), T.int64(3002)))
        var_conv1d_ncw_intermediate = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(3000)))
        var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(3000)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(3000)))
        compute = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(3000)))
        T_multiply_1 = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(3000)))
        T_add = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(3000)))
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(80), T.int64(3002)):
            with T.block("pad_temp"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(input_ids[v_i0, v_i1, v_i2 - T.int64(1)])
                T.writes(pad_temp[v_i0, v_i1, v_i2])
                pad_temp[v_i0, v_i1, v_i2] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(3001), input_ids[v_i0, v_i1, v_i2 - T.int64(1)], T.float32(0))
        for nn, ff, yy, rc, ry in T.grid(T.int64(1), T.int64(384), T.int64(3000), T.int64(80), T.int64(3)):
            with T.block("conv1d_ncw"):
                v_nn, v_ff, v_yy, v_rc, v_ry = T.axis.remap("SSSRR", [nn, ff, yy, rc, ry])
                T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry], model_encoder_conv1_weight[v_ff, v_rc, v_ry])
                T.writes(var_conv1d_ncw_intermediate[v_nn, v_ff, v_yy])
                with T.init():
                    var_conv1d_ncw_intermediate[v_nn, v_ff, v_yy] = T.float32(0)
                var_conv1d_ncw_intermediate[v_nn, v_ff, v_yy] = var_conv1d_ncw_intermediate[v_nn, v_ff, v_yy] + pad_temp[v_nn, v_rc, v_yy + v_ry] * model_encoder_conv1_weight[v_ff, v_rc, v_ry]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(3000)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_conv1d_ncw_intermediate[v_ax0, v_ax1, v_ax2], lv18[v_ax0, v_ax1, T.int64(0)])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_conv1d_ncw_intermediate[v_ax0, v_ax1, v_ax2] + lv18[v_ax0, v_ax1, T.int64(0)]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(3000)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float32(0.70710678118654757)
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(384), T.int64(3000)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_multiply[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.erf(T_multiply[v_i0, v_i1, v_i2])
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(3000)):
            with T.block("T_multiply_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(compute[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T_multiply_1[v_ax0, v_ax1, v_ax2] = compute[v_ax0, v_ax1, v_ax2] * T.float32(0.5)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(3000)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = T.float32(0.5) + T_multiply_1[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(3000)):
            with T.block("T_multiply_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_reshape(packed_params_1: T.Buffer((T.int64(384),), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(1), T.int64(384), T.int64(1)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(1)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(packed_params_1[(v_ax1 + v_ax2) % T.int64(384)])
                T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = packed_params_1[(v_ax1 + v_ax2) % T.int64(384)]

    @T.prim_func(private=True)
    def fused_reshape1_transpose7(mul: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(6), T.int64(64)))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1500), T.int64(6), T.int64(64)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(mul[T.int64(0), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(384) + v_ax1) % T.int64(1500), (v_ax2 * T.int64(64) + v_ax3) % T.int64(384)])
                T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = mul[T.int64(0), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(384) + v_ax1) % T.int64(1500), (v_ax2 * T.int64(64) + v_ax3) % T.int64(384)]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(6), T.int64(1500), T.int64(64)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]

    @T.prim_func(private=True)
    def fused_reshape4_add6(take: T.Buffer((T.int64(1), T.int64(384)), "float32"), lv21: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(384)))
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(384)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(take[T.int64(0), v_ax2 % T.int64(384)])
                T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = take[T.int64(0), v_ax2 % T.int64(384)]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2], lv21[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] + lv21[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_reshape5_squeeze(lv29: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), var_T_squeeze_intermediate: T.Buffer((T.int64(1), T.int64(6), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(6), T.int64(64)))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(6), T.int64(64)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv29[T.int64(0), T.int64(0), (v_ax2 * T.int64(64) + v_ax3) % T.int64(384)])
                T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv29[T.int64(0), T.int64(0), (v_ax2 * T.int64(64) + v_ax3) % T.int64(384)]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(6), T.int64(64)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_reshape_intermediate[T.int64(0), v_ax0, v_ax1, v_ax2])
                T.writes(var_T_squeeze_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_squeeze_intermediate[v_ax0, v_ax1, v_ax2] = var_T_reshape_intermediate[T.int64(0), v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_reshape5_transpose9(mul4: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(6), T.int64(64)))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(6), T.int64(64)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(mul4[T.int64(0), T.int64(0), (v_ax2 * T.int64(64) + v_ax3) % T.int64(384)])
                T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = mul4[T.int64(0), T.int64(0), (v_ax2 * T.int64(64) + v_ax3) % T.int64(384)]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(6), T.int64(1), T.int64(64)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]

    @T.prim_func(private=True)
    def fused_transpose11_reshape7(matmul36: T.Buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(64)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(6), T.int64(64)))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(6), T.int64(64)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(matmul36[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = matmul36[v_ax0, v_ax2, v_ax1, v_ax3]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(384)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_transpose_intermediate[T.int64(0), T.int64(0), v_ax2 % T.int64(384) // T.int64(64), v_ax2 % T.int64(64)])
                T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[T.int64(0), T.int64(0), v_ax2 % T.int64(384) // T.int64(64), v_ax2 % T.int64(64)]

    @T.prim_func(private=True)
    def fused_transpose6_add2(packed_params_4: T.Buffer((T.int64(1500), T.int64(384)), "float32"), gelu1: T.Buffer((T.int64(1), T.int64(384), T.int64(1500)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(384)))
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(384)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(gelu1[v_ax0, v_ax2, v_ax1])
                T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2] = gelu1[v_ax0, v_ax2, v_ax1]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2], packed_params_4[v_ax1, v_ax2])
                T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2] + packed_params_4[v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_transpose7(cached_encoder_key_value_0_0: T.Buffer((T.int64(1), T.int64(1500), T.int64(6), T.int64(64)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(6), T.int64(1500), T.int64(64)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(cached_encoder_key_value_0_0[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = cached_encoder_key_value_0_0[v_ax0, v_ax2, v_ax1, v_ax3]

    @T.prim_func(private=True)
    def fused_transpose8_reshape2(matmul4: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        var_T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(1500), T.int64(6), T.int64(64)))
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1500), T.int64(6), T.int64(64)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(matmul4[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = matmul4[v_ax0, v_ax2, v_ax1, v_ax3]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(384)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_transpose_intermediate[T.int64(0), (v_ax2 // T.int64(384) + v_ax1) % T.int64(1500), v_ax2 % T.int64(384) // T.int64(64), v_ax2 % T.int64(64)])
                T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[T.int64(0), (v_ax2 // T.int64(384) + v_ax1) % T.int64(1500), v_ax2 % T.int64(384) // T.int64(64), v_ax2 % T.int64(64)]

    @T.prim_func(private=True)
    def layer_norm(A: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32"), B: T.Buffer((T.int64(384),), "float32"), C: T.Buffer((T.int64(384),), "float32"), T_layer_norm: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(1500)))
        A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(1500)))
        for ax0, ax1, k2 in T.grid(T.int64(1), T.int64(1500), T.int64(384)):
            with T.block("A_red_temp"):
                v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                T.reads(A[v_ax0, v_ax1, v_k2])
                T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
                with T.init():
                    A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                    A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2]
                v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2] * A[v_ax0, v_ax1, v_k2]
                A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
                A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1500), T.int64(384)):
            with T.block("T_layer_norm"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], B[v_ax2], C[v_ax2])
                T.writes(T_layer_norm[v_ax0, v_ax1, v_ax2])
                T_layer_norm[v_ax0, v_ax1, v_ax2] = (A[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.0026041666666666665)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.0026041666666666665) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.0026041666666666665) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.0026041666666666665)) + T.float32(1.0000000000000001e-05)) * B[v_ax2] + C[v_ax2]

    @T.prim_func(private=True)
    def layer_norm1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), B: T.Buffer((T.int64(384),), "float32"), C: T.Buffer((T.int64(384),), "float32"), T_layer_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(1)))
        A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(1)))
        for ax0, ax1, k2 in T.grid(T.int64(1), T.int64(1), T.int64(384)):
            with T.block("A_red_temp"):
                v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                T.reads(A[v_ax0, v_ax1, v_k2])
                T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
                with T.init():
                    A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                    A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2]
                v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2] * A[v_ax0, v_ax1, v_k2]
                A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
                A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(384)):
            with T.block("T_layer_norm"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], B[v_ax2], C[v_ax2])
                T.writes(T_layer_norm[v_ax0, v_ax1, v_ax2])
                T_layer_norm[v_ax0, v_ax1, v_ax2] = (A[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.0026041666666666665)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.0026041666666666665) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.0026041666666666665) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.0026041666666666665)) + T.float32(1.0000000000000001e-05)) * B[v_ax2] + C[v_ax2]

    @T.prim_func(private=True)
    def layer_norm2(var_A: T.handle, B: T.Buffer((T.int64(384),), "float32"), C: T.Buffer((T.int64(384),), "float32"), var_T_layer_norm: T.handle):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), seq_len, T.int64(384)))
        T_layer_norm = T.match_buffer(var_T_layer_norm, (T.int64(1), seq_len, T.int64(384)))
        # with T.block("root"):
        A_red_temp_v0 = T.alloc_buffer((T.int64(1), seq_len))
        A_red_temp_v1 = T.alloc_buffer((T.int64(1), seq_len))
        for ax0, ax1, k2 in T.grid(T.int64(1), seq_len, T.int64(384)):
            with T.block("A_red_temp"):
                v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                T.reads(A[v_ax0, v_ax1, v_k2])
                T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
                with T.init():
                    A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                    A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2]
                v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2] * A[v_ax0, v_ax1, v_k2]
                A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
                A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(384)):
            with T.block("T_layer_norm"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], B[v_ax2], C[v_ax2])
                T.writes(T_layer_norm[v_ax0, v_ax1, v_ax2])
                T_layer_norm[v_ax0, v_ax1, v_ax2] = (A[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.0026041666666666665)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.0026041666666666665) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.0026041666666666665) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.0026041666666666665)) + T.float32(1.0000000000000001e-05)) * B[v_ax2] + C[v_ax2]

    @T.prim_func(private=True)
    def matmul11(A: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(1500)), "float32"), B: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(6), T.int64(1500), T.int64(64), T.int64(1500)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

    @T.prim_func(private=True)
    def matmul13(var_A: T.handle, var_B: T.handle, matmul: T.Buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(64)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        total_seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(6), T.int64(1), total_seq_len))
        B = T.match_buffer(var_B, (T.int64(1), T.int64(6), total_seq_len, T.int64(64)))
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(6), T.int64(1), T.int64(64), total_seq_len):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

    @T.prim_func(private=True)
    def matmul14(A: T.Buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(1500)), "float32"), B: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(64)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(6), T.int64(1), T.int64(64), T.int64(1500)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

    @T.prim_func(private=True)
    def matmul19(var_A: T.handle, var_B: T.handle, var_matmul: T.handle):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        seq_len, total_seq_len = T.int64(), T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(6), seq_len, total_seq_len))
        B = T.match_buffer(var_B, (T.int64(1), T.int64(6), total_seq_len, T.int64(64)))
        matmul = T.match_buffer(var_matmul, (T.int64(1), T.int64(6), seq_len, T.int64(64)))
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(6), seq_len, T.int64(64), total_seq_len):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

    @T.prim_func(private=True)
    def matmul20(var_A: T.handle, B: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32"), var_matmul: T.handle):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(6), seq_len, T.int64(1500)))
        matmul = T.match_buffer(var_matmul, (T.int64(1), T.int64(6), seq_len, T.int64(64)))
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(6), seq_len, T.int64(64), T.int64(1500)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

    @T.prim_func(private=True)
    def position_embedding(A: T.Buffer((T.int64(1), T.int64(1)), "int32"), B: T.Buffer((T.int64(448), T.int64(384)), "float32"), position_embedding: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), total_seq_len: T.int64):
        T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j, k in T.grid(T.int64(1), T.int64(1), T.int64(384)):
            with T.block("position_embedding"):
                v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
                T.reads(B[total_seq_len + v_j - T.int64(1), v_k])
                T.writes(position_embedding[v_i, v_j, v_k])
                position_embedding[v_i, v_j, v_k] = B[total_seq_len + v_j - T.int64(1), v_k]

    @T.prim_func(private=True)
    def position_embedding1(var_A: T.handle, B: T.Buffer((T.int64(448), T.int64(384)), "float32"), var_position_embedding: T.handle, total_seq_len: T.int64):
        T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), seq_len), "int32")
        position_embedding = T.match_buffer(var_position_embedding, (T.int64(1), seq_len, T.int64(384)))
        # with T.block("root"):
        for i, j, k in T.grid(T.int64(1), seq_len, T.int64(384)):
            with T.block("position_embedding"):
                v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
                T.reads(B[total_seq_len + v_j - T.int64(1), v_k])
                T.writes(position_embedding[v_i, v_j, v_k])
                position_embedding[v_i, v_j, v_k] = B[total_seq_len + v_j - T.int64(1), v_k]

    @T.prim_func(private=True)
    def reshape1(A: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(1500), T.int64(6), T.int64(64)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1500), T.int64(6), T.int64(64)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[T.int64(0), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(384) + v_ax1) % T.int64(1500), (v_ax2 * T.int64(64) + v_ax3) % T.int64(384)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(384) + v_ax1) % T.int64(1500), (v_ax2 * T.int64(64) + v_ax3) % T.int64(384)]

    @T.prim_func(private=True)
    def reshape10(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), seq_len, T.int64(384)))
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), seq_len, T.int64(6), T.int64(64)))
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), seq_len, T.int64(6), T.int64(64)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[T.int64(0), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(384) + v_ax0 * seq_len + v_ax1) % seq_len, (v_ax2 * T.int64(64) + v_ax3) % T.int64(384)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(384) + v_ax0 * seq_len + v_ax1) % seq_len, (v_ax2 * T.int64(64) + v_ax3) % T.int64(384)]

    @T.prim_func(private=True)
    def reshape11(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), seq_len, T.int64(6), T.int64(64)))
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), seq_len, T.int64(384)))
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(384)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[T.int64(0), (v_ax2 // T.int64(384) + v_ax0 * seq_len + v_ax1) % seq_len, v_ax2 % T.int64(384) // T.int64(64), v_ax2 % T.int64(64)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[T.int64(0), (v_ax2 // T.int64(384) + v_ax0 * seq_len + v_ax1) % seq_len, v_ax2 % T.int64(384) // T.int64(64), v_ax2 % T.int64(64)]

    @T.prim_func(private=True)
    def reshape3(A: T.Buffer((T.int64(1), T.int64(1)), "int32"), T_reshape: T.Buffer((T.int64(1),), "int32")):
        T.func_attr({"op_pattern": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0 in range(T.int64(1)):
            with T.block("T_reshape"):
                v_ax0 = T.axis.spatial(T.int64(1), ax0)
                T.reads(A[T.int64(0), T.int64(0)])
                T.writes(T_reshape[v_ax0])
                T_reshape[v_ax0] = A[T.int64(0), T.int64(0)]

    @T.prim_func(private=True)
    def reshape6(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
        total_seq_len = T.int64()
        A = T.match_buffer(var_A, (total_seq_len, T.int64(6), T.int64(64)))
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), total_seq_len, T.int64(6), T.int64(64)))
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), total_seq_len, T.int64(6), T.int64(64)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[((v_ax3 // T.int64(64) + v_ax2) // T.int64(6) + v_ax0 * total_seq_len + v_ax1) % total_seq_len, (v_ax3 // T.int64(64) + v_ax2) % T.int64(6), v_ax3 % T.int64(64)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[((v_ax3 // T.int64(64) + v_ax2) // T.int64(6) + v_ax0 * total_seq_len + v_ax1) % total_seq_len, (v_ax3 // T.int64(64) + v_ax2) % T.int64(6), v_ax3 % T.int64(64)]

    @T.prim_func(private=True)
    def reshape8(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), seq_len), "int32")
        T_reshape = T.match_buffer(var_T_reshape, (seq_len,), "int32")
        # with T.block("root"):
        for ax0 in range(seq_len):
            with T.block("T_reshape"):
                v_ax0 = T.axis.spatial(seq_len, ax0)
                T.reads(A[T.int64(0), v_ax0 % seq_len])
                T.writes(T_reshape[v_ax0])
                T_reshape[v_ax0] = A[T.int64(0), v_ax0 % seq_len]

    @T.prim_func(private=True)
    def reshape9(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (seq_len, T.int64(384)))
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), seq_len, T.int64(384)))
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(384)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[(v_ax2 // T.int64(384) + v_ax0 * seq_len + v_ax1) % seq_len, v_ax2 % T.int64(384)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[(v_ax2 // T.int64(384) + v_ax0 * seq_len + v_ax1) % seq_len, v_ax2 % T.int64(384)]

    @T.prim_func(private=True)
    def softmax(A: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(1500)), "float32"), T_softmax_norm: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(1500)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(6), T.int64(1500)))
        T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(1500)))
        T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(6), T.int64(1500)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(6), T.int64(1500), T.int64(1500)):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(6), T.int64(1500), T.int64(1500)):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(6), T.int64(1500), T.int64(1500)):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
                T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(6), T.int64(1500), T.int64(1500)):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
                T.block_attr({"axis": 3})
                T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]

    @T.prim_func(private=True)
    def softmax1(var_A: T.handle, var_T_softmax_norm: T.handle):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        total_seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(6), T.int64(1), total_seq_len))
        T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(6), T.int64(1), total_seq_len))
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(6), T.int64(1)))
        T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(6), T.int64(1), total_seq_len))
        T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(6), T.int64(1)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(6), T.int64(1), total_seq_len):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(6), T.int64(1), total_seq_len):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(6), T.int64(1), total_seq_len):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
                T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(6), T.int64(1), total_seq_len):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
                T.block_attr({"axis": 3})
                T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]

    @T.prim_func(private=True)
    def softmax2(A: T.Buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(1500)), "float32"), T_softmax_norm: T.Buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(1500)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(6), T.int64(1)))
        T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(1500)))
        T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(6), T.int64(1)))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(6), T.int64(1), T.int64(1500)):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(6), T.int64(1), T.int64(1500)):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(6), T.int64(1), T.int64(1500)):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
                T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(6), T.int64(1), T.int64(1500)):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
                T.block_attr({"axis": 3})
                T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]

    @T.prim_func(private=True)
    def softmax3(var_A: T.handle, var_T_softmax_norm: T.handle):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        seq_len, total_seq_len = T.int64(), T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(6), seq_len, total_seq_len))
        T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(6), seq_len, total_seq_len))
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(6), seq_len))
        T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(6), seq_len, total_seq_len))
        T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(6), seq_len))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(6), seq_len, total_seq_len):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(6), seq_len, total_seq_len):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(6), seq_len, total_seq_len):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
                T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(6), seq_len, total_seq_len):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
                T.block_attr({"axis": 3})
                T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]

    @T.prim_func(private=True)
    def softmax4(var_A: T.handle, var_T_softmax_norm: T.handle):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(6), seq_len, T.int64(1500)))
        T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(6), seq_len, T.int64(1500)))
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(6), seq_len))
        T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(6), seq_len, T.int64(1500)))
        T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(6), seq_len))
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(6), seq_len, T.int64(1500)):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(6), seq_len, T.int64(1500)):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(6), seq_len, T.int64(1500)):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
                T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                with T.init():
                    T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
                T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(6), seq_len, T.int64(1500)):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
                T.block_attr({"axis": 3})
                T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]

    @T.prim_func(private=True)
    def squeeze1(var_A: T.handle, var_T_squeeze: T.handle):
        T.func_attr({"op_pattern": 1, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), seq_len, T.int64(6), T.int64(64)))
        T_squeeze = T.match_buffer(var_T_squeeze, (seq_len, T.int64(6), T.int64(64)))
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(seq_len, T.int64(6), T.int64(64)):
            with T.block("T_squeeze"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[T.int64(0), v_ax0, v_ax1, v_ax2])
                T.writes(T_squeeze[v_ax0, v_ax1, v_ax2])
                T_squeeze[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def take(A: T.Buffer((T.int64(51865), T.int64(384)), "float32"), B: T.Buffer((T.int64(1),), "int32"), T_take: T.Buffer((T.int64(1), T.int64(384)), "float32")):
        T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(384)):
            with T.block("T_take"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[B[v_ax0], v_ax1], B[v_ax0])
                T.writes(T_take[v_ax0, v_ax1])
                T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]

    @T.prim_func(private=True)
    def take1(A: T.Buffer((T.int64(51865), T.int64(384)), "float32"), var_B: T.handle, var_T_take: T.handle):
        T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        B = T.match_buffer(var_B, (seq_len,), "int32")
        T_take = T.match_buffer(var_T_take, (seq_len, T.int64(384)))
        # with T.block("root"):
        for ax0, ax1 in T.grid(seq_len, T.int64(384)):
            with T.block("T_take"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[B[v_ax0], v_ax1], B[v_ax0])
                T.writes(T_take[v_ax0, v_ax1])
                T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]

    @T.prim_func(private=True)
    def transpose10(var_A: T.handle, var_T_transpose: T.handle):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        total_seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), total_seq_len, T.int64(6), T.int64(64)))
        T_transpose = T.match_buffer(var_T_transpose, (T.int64(1), T.int64(6), total_seq_len, T.int64(64)))
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(6), total_seq_len, T.int64(64)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]

    @T.prim_func(private=True)
    def transpose12(var_A: T.handle, var_T_transpose: T.handle):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(6), seq_len, T.int64(64)))
        T_transpose = T.match_buffer(var_T_transpose, (T.int64(1), seq_len, T.int64(6), T.int64(64)))
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), seq_len, T.int64(6), T.int64(64)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]

    @T.prim_func(private=True)
    def transpose7(A: T.Buffer((T.int64(1), T.int64(1500), T.int64(6), T.int64(64)), "float32"), T_transpose: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(6), T.int64(1500), T.int64(64)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]

    @R.function
    def _initialize_effect() -> R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object):
        R.func_attr({"tir_var_upper_bound": {"seq_len": 448, "total_seq_len": 448}})
        with R.dataflow():
            model_decoder_layers_0_self_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][0], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            model_decoder_layers_0_self_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][1], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            model_decoder_layers_0_encoder_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][2], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            model_decoder_layers_0_encoder_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][3], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            model_decoder_layers_1_self_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][4], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            model_decoder_layers_1_self_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][5], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            model_decoder_layers_1_encoder_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][6], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            model_decoder_layers_1_encoder_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][7], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            model_decoder_layers_2_self_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][8], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            model_decoder_layers_2_self_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][9], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            model_decoder_layers_2_encoder_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][10], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            model_decoder_layers_2_encoder_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][11], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            model_decoder_layers_3_self_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][12], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            model_decoder_layers_3_self_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][13], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            model_decoder_layers_3_encoder_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][14], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            model_decoder_layers_3_encoder_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", metadata["relax.expr.Constant"][15], R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv16: R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object) = model_decoder_layers_0_self_attn_k_cache, model_decoder_layers_0_self_attn_v_cache, model_decoder_layers_0_encoder_attn_k_cache, model_decoder_layers_0_encoder_attn_v_cache, model_decoder_layers_1_self_attn_k_cache, model_decoder_layers_1_self_attn_v_cache, model_decoder_layers_1_encoder_attn_k_cache, model_decoder_layers_1_encoder_attn_v_cache, model_decoder_layers_2_self_attn_k_cache, model_decoder_layers_2_self_attn_v_cache, model_decoder_layers_2_encoder_attn_k_cache, model_decoder_layers_2_encoder_attn_v_cache, model_decoder_layers_3_self_attn_k_cache, model_decoder_layers_3_self_attn_v_cache, model_decoder_layers_3_encoder_attn_k_cache, model_decoder_layers_3_encoder_attn_v_cache
            gv: R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object) = lv16
            R.output(gv)
        return gv

    @R.function
    def decode(input_ids: R.Tensor((1, 1), dtype="int32"), total_seq_len_1: R.Shape(["total_seq_len"]), encoder_hidden_states: R.Tensor((1, 1500, 384), dtype="float32"), packed_effects: R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object), packed_params: R.Tuple(R.Tensor((384, 80, 3), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384, 3), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1500, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((51865, 384), dtype="float32"), R.Tensor((448, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((51865, 384), dtype="float32"))) -> R.Tuple(R.Tuple(R.Tensor((1, 1, 51865), dtype="float32"), R.Tuple(R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")), R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")), R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")), R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")))), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)):
        total_seq_len = T.int64()
        R.func_attr({"num_input": 4, "tir_var_upper_bound": {"seq_len": 448, "total_seq_len": 448}})
        cls = Module
        with R.dataflow():
            model_decoder_layers_0_self_attn_k_cache2: R.Object = packed_effects[0]
            model_decoder_layers_0_self_attn_v_cache2: R.Object = packed_effects[1]
            model_decoder_layers_0_encoder_attn_k_cache2: R.Object = packed_effects[2]
            model_decoder_layers_0_encoder_attn_v_cache2: R.Object = packed_effects[3]
            model_decoder_layers_1_self_attn_k_cache2: R.Object = packed_effects[4]
            model_decoder_layers_1_self_attn_v_cache2: R.Object = packed_effects[5]
            model_decoder_layers_1_encoder_attn_k_cache2: R.Object = packed_effects[6]
            model_decoder_layers_1_encoder_attn_v_cache2: R.Object = packed_effects[7]
            model_decoder_layers_2_self_attn_k_cache2: R.Object = packed_effects[8]
            model_decoder_layers_2_self_attn_v_cache2: R.Object = packed_effects[9]
            model_decoder_layers_2_encoder_attn_k_cache2: R.Object = packed_effects[10]
            model_decoder_layers_2_encoder_attn_v_cache2: R.Object = packed_effects[11]
            model_decoder_layers_3_self_attn_k_cache2: R.Object = packed_effects[12]
            model_decoder_layers_3_self_attn_v_cache2: R.Object = packed_effects[13]
            model_decoder_layers_3_encoder_attn_k_cache2: R.Object = packed_effects[14]
            model_decoder_layers_3_encoder_attn_v_cache2: R.Object = packed_effects[15]
            model_decoder_embed_tokens_weight1: R.Tensor((51865, 384), dtype="float32") = packed_params[67]
            model_decoder_embed_positions_weight1: R.Tensor((448, 384), dtype="float32") = packed_params[68]
            model_decoder_layers_0_self_attn_k_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[69]
            model_decoder_layers_0_self_attn_v_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[70]
            model_decoder_layers_0_self_attn_v_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[71]
            model_decoder_layers_0_self_attn_q_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[72]
            model_decoder_layers_0_self_attn_q_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[73]
            model_decoder_layers_0_self_attn_out_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[74]
            model_decoder_layers_0_self_attn_out_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[75]
            model_decoder_layers_0_self_attn_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[76]
            model_decoder_layers_0_self_attn_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[77]
            model_decoder_layers_0_encoder_attn_k_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[78]
            model_decoder_layers_0_encoder_attn_v_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[79]
            model_decoder_layers_0_encoder_attn_v_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[80]
            model_decoder_layers_0_encoder_attn_q_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[81]
            model_decoder_layers_0_encoder_attn_q_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[82]
            model_decoder_layers_0_encoder_attn_out_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[83]
            model_decoder_layers_0_encoder_attn_out_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[84]
            model_decoder_layers_0_encoder_attn_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[85]
            model_decoder_layers_0_encoder_attn_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[86]
            model_decoder_layers_0_fc1_weight1: R.Tensor((1536, 384), dtype="float32") = packed_params[87]
            model_decoder_layers_0_fc1_bias1: R.Tensor((1536,), dtype="float32") = packed_params[88]
            model_decoder_layers_0_fc2_weight1: R.Tensor((384, 1536), dtype="float32") = packed_params[89]
            model_decoder_layers_0_fc2_bias1: R.Tensor((384,), dtype="float32") = packed_params[90]
            model_decoder_layers_0_final_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[91]
            model_decoder_layers_0_final_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[92]
            model_decoder_layers_1_self_attn_k_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[93]
            model_decoder_layers_1_self_attn_v_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[94]
            model_decoder_layers_1_self_attn_v_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[95]
            model_decoder_layers_1_self_attn_q_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[96]
            model_decoder_layers_1_self_attn_q_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[97]
            model_decoder_layers_1_self_attn_out_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[98]
            model_decoder_layers_1_self_attn_out_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[99]
            model_decoder_layers_1_self_attn_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[100]
            model_decoder_layers_1_self_attn_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[101]
            model_decoder_layers_1_encoder_attn_k_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[102]
            model_decoder_layers_1_encoder_attn_v_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[103]
            model_decoder_layers_1_encoder_attn_v_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[104]
            model_decoder_layers_1_encoder_attn_q_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[105]
            model_decoder_layers_1_encoder_attn_q_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[106]
            model_decoder_layers_1_encoder_attn_out_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[107]
            model_decoder_layers_1_encoder_attn_out_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[108]
            model_decoder_layers_1_encoder_attn_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[109]
            model_decoder_layers_1_encoder_attn_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[110]
            model_decoder_layers_1_fc1_weight1: R.Tensor((1536, 384), dtype="float32") = packed_params[111]
            model_decoder_layers_1_fc1_bias1: R.Tensor((1536,), dtype="float32") = packed_params[112]
            model_decoder_layers_1_fc2_weight1: R.Tensor((384, 1536), dtype="float32") = packed_params[113]
            model_decoder_layers_1_fc2_bias1: R.Tensor((384,), dtype="float32") = packed_params[114]
            model_decoder_layers_1_final_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[115]
            model_decoder_layers_1_final_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[116]
            model_decoder_layers_2_self_attn_k_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[117]
            model_decoder_layers_2_self_attn_v_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[118]
            model_decoder_layers_2_self_attn_v_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[119]
            model_decoder_layers_2_self_attn_q_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[120]
            model_decoder_layers_2_self_attn_q_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[121]
            model_decoder_layers_2_self_attn_out_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[122]
            model_decoder_layers_2_self_attn_out_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[123]
            model_decoder_layers_2_self_attn_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[124]
            model_decoder_layers_2_self_attn_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[125]
            model_decoder_layers_2_encoder_attn_k_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[126]
            model_decoder_layers_2_encoder_attn_v_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[127]
            model_decoder_layers_2_encoder_attn_v_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[128]
            model_decoder_layers_2_encoder_attn_q_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[129]
            model_decoder_layers_2_encoder_attn_q_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[130]
            model_decoder_layers_2_encoder_attn_out_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[131]
            model_decoder_layers_2_encoder_attn_out_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[132]
            model_decoder_layers_2_encoder_attn_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[133]
            model_decoder_layers_2_encoder_attn_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[134]
            model_decoder_layers_2_fc1_weight1: R.Tensor((1536, 384), dtype="float32") = packed_params[135]
            model_decoder_layers_2_fc1_bias1: R.Tensor((1536,), dtype="float32") = packed_params[136]
            model_decoder_layers_2_fc2_weight1: R.Tensor((384, 1536), dtype="float32") = packed_params[137]
            model_decoder_layers_2_fc2_bias1: R.Tensor((384,), dtype="float32") = packed_params[138]
            model_decoder_layers_2_final_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[139]
            model_decoder_layers_2_final_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[140]
            model_decoder_layers_3_self_attn_k_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[141]
            model_decoder_layers_3_self_attn_v_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[142]
            model_decoder_layers_3_self_attn_v_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[143]
            model_decoder_layers_3_self_attn_q_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[144]
            model_decoder_layers_3_self_attn_q_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[145]
            model_decoder_layers_3_self_attn_out_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[146]
            model_decoder_layers_3_self_attn_out_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[147]
            model_decoder_layers_3_self_attn_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[148]
            model_decoder_layers_3_self_attn_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[149]
            model_decoder_layers_3_encoder_attn_k_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[150]
            model_decoder_layers_3_encoder_attn_v_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[151]
            model_decoder_layers_3_encoder_attn_v_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[152]
            model_decoder_layers_3_encoder_attn_q_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[153]
            model_decoder_layers_3_encoder_attn_q_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[154]
            model_decoder_layers_3_encoder_attn_out_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[155]
            model_decoder_layers_3_encoder_attn_out_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[156]
            model_decoder_layers_3_encoder_attn_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[157]
            model_decoder_layers_3_encoder_attn_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[158]
            model_decoder_layers_3_fc1_weight1: R.Tensor((1536, 384), dtype="float32") = packed_params[159]
            model_decoder_layers_3_fc1_bias1: R.Tensor((1536,), dtype="float32") = packed_params[160]
            model_decoder_layers_3_fc2_weight1: R.Tensor((384, 1536), dtype="float32") = packed_params[161]
            model_decoder_layers_3_fc2_bias1: R.Tensor((384,), dtype="float32") = packed_params[162]
            model_decoder_layers_3_final_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[163]
            model_decoder_layers_3_final_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[164]
            model_decoder_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[165]
            model_decoder_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[166]
            proj_out_weight1: R.Tensor((51865, 384), dtype="float32") = packed_params[167]
            reshape16 = R.call_tir(cls.reshape3, (input_ids,), out_sinfo=R.Tensor((1,), dtype="int32"))
            take = R.call_tir(cls.take, (model_decoder_embed_tokens_weight1, reshape16), out_sinfo=R.Tensor((1, 384), dtype="float32"))
            lv21 = R.call_tir(cls.position_embedding, (input_ids, model_decoder_embed_positions_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"), tir_vars=R.shape([total_seq_len]))
            lv48 = R.call_tir(cls.fused_reshape4_add6, (take, lv21), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            layer_norm9 = R.call_tir(cls.layer_norm1, (lv48, model_decoder_layers_0_self_attn_layer_norm_weight1, model_decoder_layers_0_self_attn_layer_norm_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv49 = R.call_tir(cls.fused_NT_matmul_add7_multiply1, (layer_norm9, model_decoder_layers_0_self_attn_q_proj_weight1, model_decoder_layers_0_self_attn_q_proj_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv50 = R.call_tir(cls.fused_reshape5_transpose9, (lv49,), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv29 = R.call_tir(cls.NT_matmul, (layer_norm9, model_decoder_layers_0_self_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv51 = R.call_tir(cls.fused_reshape5_squeeze, (lv29,), out_sinfo=R.Tensor((1, 6, 64), dtype="float32"))
            lv52 = R.call_tir(cls.fused_NT_matmul_add7, (layer_norm9, model_decoder_layers_0_self_attn_v_proj_weight1, model_decoder_layers_0_self_attn_v_proj_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv53 = R.call_tir(cls.fused_reshape5_squeeze, (lv52,), out_sinfo=R.Tensor((1, 6, 64), dtype="float32"))
            lv22: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_0_self_attn_k_cache2, lv51, sinfo_args=(R.Object,))
            lv23: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_0_self_attn_v_cache2, lv53, sinfo_args=(R.Object,))
            lv24: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv22, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape21 = R.call_tir(cls.reshape6, (lv24,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            lv25: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv23, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape22 = R.call_tir(cls.reshape6, (lv25,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            permute_dims49 = R.call_tir(cls.transpose10, (reshape21,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            permute_dims50 = R.call_tir(cls.transpose10, (reshape22,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            lv54 = R.call_tir(cls.fused_NT_matmul1_maximum2_minimum2, (lv50, permute_dims49), out_sinfo=R.Tensor((1, 6, 1, total_seq_len), dtype="float32"))
            softmax4 = R.call_tir(cls.softmax1, (lv54,), out_sinfo=R.Tensor((1, 6, 1, total_seq_len), dtype="float32"))
            matmul36 = R.call_tir(cls.matmul13, (softmax4, permute_dims50), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv55 = R.call_tir(cls.fused_transpose11_reshape7, (matmul36,), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv56 = R.call_tir(cls.fused_NT_matmul_add7_add6, (lv55, model_decoder_layers_0_self_attn_out_proj_weight1, model_decoder_layers_0_self_attn_out_proj_bias1, lv48), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            layer_norm10 = R.call_tir(cls.layer_norm1, (lv56, model_decoder_layers_0_encoder_attn_layer_norm_weight1, model_decoder_layers_0_encoder_attn_layer_norm_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv57 = R.call_tir(cls.fused_NT_matmul_add7_multiply1, (layer_norm10, model_decoder_layers_0_encoder_attn_q_proj_weight1, model_decoder_layers_0_encoder_attn_q_proj_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv58 = R.call_tir(cls.fused_reshape5_transpose9, (lv57,), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv34 = R.call_tir(cls.NT_matmul2, (encoder_hidden_states, model_decoder_layers_0_encoder_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape25 = R.call_tir(cls.reshape1, (lv34,), out_sinfo=R.Tensor((1, 1500, 6, 64), dtype="float32"))
            lv59 = R.call_tir(cls.fused_NT_matmul2_add3, (encoder_hidden_states, model_decoder_layers_0_encoder_attn_v_proj_weight1, model_decoder_layers_0_encoder_attn_v_proj_bias1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape26 = R.call_tir(cls.reshape1, (lv59,), out_sinfo=R.Tensor((1, 1500, 6, 64), dtype="float32"))
            permute_dims58 = R.call_tir(cls.transpose7, (reshape25,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            permute_dims59 = R.call_tir(cls.transpose7, (reshape26,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv60 = R.call_tir(cls.fused_NT_matmul3_maximum3_minimum3, (lv58, permute_dims58), out_sinfo=R.Tensor((1, 6, 1, 1500), dtype="float32"))
            softmax5 = R.call_tir(cls.softmax2, (lv60,), out_sinfo=R.Tensor((1, 6, 1, 1500), dtype="float32"))
            matmul42 = R.call_tir(cls.matmul14, (softmax5, permute_dims59), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv61 = R.call_tir(cls.fused_transpose11_reshape7, (matmul42,), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv62 = R.call_tir(cls.fused_NT_matmul_add7_add6, (lv61, model_decoder_layers_0_encoder_attn_out_proj_weight1, model_decoder_layers_0_encoder_attn_out_proj_bias1, lv56), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            layer_norm11 = R.call_tir(cls.layer_norm1, (lv62, model_decoder_layers_0_final_layer_norm_weight1, model_decoder_layers_0_final_layer_norm_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv63 = R.call_tir(cls.fused_NT_matmul4_add8_gelu3, (layer_norm11, model_decoder_layers_0_fc1_weight1, model_decoder_layers_0_fc1_bias1), out_sinfo=R.Tensor((1, 1, 1536), dtype="float32"))
            lv64 = R.call_tir(cls.fused_NT_matmul5_add7_add6, (lv63, model_decoder_layers_0_fc2_weight1, model_decoder_layers_0_fc2_bias1, lv62), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            layer_norm12 = R.call_tir(cls.layer_norm1, (lv64, model_decoder_layers_1_self_attn_layer_norm_weight1, model_decoder_layers_1_self_attn_layer_norm_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv65 = R.call_tir(cls.fused_NT_matmul_add7_multiply1, (layer_norm12, model_decoder_layers_1_self_attn_q_proj_weight1, model_decoder_layers_1_self_attn_q_proj_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv66 = R.call_tir(cls.fused_reshape5_transpose9, (lv65,), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv41 = R.call_tir(cls.NT_matmul, (layer_norm12, model_decoder_layers_1_self_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv67 = R.call_tir(cls.fused_reshape5_squeeze, (lv41,), out_sinfo=R.Tensor((1, 6, 64), dtype="float32"))
            lv68 = R.call_tir(cls.fused_NT_matmul_add7, (layer_norm12, model_decoder_layers_1_self_attn_v_proj_weight1, model_decoder_layers_1_self_attn_v_proj_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv69 = R.call_tir(cls.fused_reshape5_squeeze, (lv68,), out_sinfo=R.Tensor((1, 6, 64), dtype="float32"))
            lv26: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_1_self_attn_k_cache2, lv67, sinfo_args=(R.Object,))
            lv27: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_1_self_attn_v_cache2, lv69, sinfo_args=(R.Object,))
            lv28: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv26, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape31 = R.call_tir(cls.reshape6, (lv28,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            lv29_1: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv27, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape32 = R.call_tir(cls.reshape6, (lv29_1,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            permute_dims69 = R.call_tir(cls.transpose10, (reshape31,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            permute_dims70 = R.call_tir(cls.transpose10, (reshape32,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            lv70 = R.call_tir(cls.fused_NT_matmul1_maximum2_minimum2, (lv66, permute_dims69), out_sinfo=R.Tensor((1, 6, 1, total_seq_len), dtype="float32"))
            softmax6 = R.call_tir(cls.softmax1, (lv70,), out_sinfo=R.Tensor((1, 6, 1, total_seq_len), dtype="float32"))
            matmul50 = R.call_tir(cls.matmul13, (softmax6, permute_dims70), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv71 = R.call_tir(cls.fused_transpose11_reshape7, (matmul50,), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv72 = R.call_tir(cls.fused_NT_matmul_add7_add6, (lv71, model_decoder_layers_1_self_attn_out_proj_weight1, model_decoder_layers_1_self_attn_out_proj_bias1, lv64), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            layer_norm13 = R.call_tir(cls.layer_norm1, (lv72, model_decoder_layers_1_encoder_attn_layer_norm_weight1, model_decoder_layers_1_encoder_attn_layer_norm_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv73 = R.call_tir(cls.fused_NT_matmul_add7_multiply1, (layer_norm13, model_decoder_layers_1_encoder_attn_q_proj_weight1, model_decoder_layers_1_encoder_attn_q_proj_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv74 = R.call_tir(cls.fused_reshape5_transpose9, (lv73,), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv46 = R.call_tir(cls.NT_matmul2, (encoder_hidden_states, model_decoder_layers_1_encoder_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape35 = R.call_tir(cls.reshape1, (lv46,), out_sinfo=R.Tensor((1, 1500, 6, 64), dtype="float32"))
            lv75 = R.call_tir(cls.fused_NT_matmul2_add3, (encoder_hidden_states, model_decoder_layers_1_encoder_attn_v_proj_weight1, model_decoder_layers_1_encoder_attn_v_proj_bias1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape36 = R.call_tir(cls.reshape1, (lv75,), out_sinfo=R.Tensor((1, 1500, 6, 64), dtype="float32"))
            permute_dims78 = R.call_tir(cls.transpose7, (reshape35,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            permute_dims79 = R.call_tir(cls.transpose7, (reshape36,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv76 = R.call_tir(cls.fused_NT_matmul3_maximum3_minimum3, (lv74, permute_dims78), out_sinfo=R.Tensor((1, 6, 1, 1500), dtype="float32"))
            softmax7 = R.call_tir(cls.softmax2, (lv76,), out_sinfo=R.Tensor((1, 6, 1, 1500), dtype="float32"))
            matmul56 = R.call_tir(cls.matmul14, (softmax7, permute_dims79), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv77 = R.call_tir(cls.fused_transpose11_reshape7, (matmul56,), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv78 = R.call_tir(cls.fused_NT_matmul_add7_add6, (lv77, model_decoder_layers_1_encoder_attn_out_proj_weight1, model_decoder_layers_1_encoder_attn_out_proj_bias1, lv72), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            layer_norm14 = R.call_tir(cls.layer_norm1, (lv78, model_decoder_layers_1_final_layer_norm_weight1, model_decoder_layers_1_final_layer_norm_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv79 = R.call_tir(cls.fused_NT_matmul4_add8_gelu3, (layer_norm14, model_decoder_layers_1_fc1_weight1, model_decoder_layers_1_fc1_bias1), out_sinfo=R.Tensor((1, 1, 1536), dtype="float32"))
            lv80 = R.call_tir(cls.fused_NT_matmul5_add7_add6, (lv79, model_decoder_layers_1_fc2_weight1, model_decoder_layers_1_fc2_bias1, lv78), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            layer_norm15 = R.call_tir(cls.layer_norm1, (lv80, model_decoder_layers_2_self_attn_layer_norm_weight1, model_decoder_layers_2_self_attn_layer_norm_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv81 = R.call_tir(cls.fused_NT_matmul_add7_multiply1, (layer_norm15, model_decoder_layers_2_self_attn_q_proj_weight1, model_decoder_layers_2_self_attn_q_proj_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv82 = R.call_tir(cls.fused_reshape5_transpose9, (lv81,), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv53_1 = R.call_tir(cls.NT_matmul, (layer_norm15, model_decoder_layers_2_self_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv83 = R.call_tir(cls.fused_reshape5_squeeze, (lv53_1,), out_sinfo=R.Tensor((1, 6, 64), dtype="float32"))
            lv84 = R.call_tir(cls.fused_NT_matmul_add7, (layer_norm15, model_decoder_layers_2_self_attn_v_proj_weight1, model_decoder_layers_2_self_attn_v_proj_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv85 = R.call_tir(cls.fused_reshape5_squeeze, (lv84,), out_sinfo=R.Tensor((1, 6, 64), dtype="float32"))
            lv30: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_2_self_attn_k_cache2, lv83, sinfo_args=(R.Object,))
            lv31: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_2_self_attn_v_cache2, lv85, sinfo_args=(R.Object,))
            lv32: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv30, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape41 = R.call_tir(cls.reshape6, (lv32,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            lv33: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv31, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape42 = R.call_tir(cls.reshape6, (lv33,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            permute_dims89 = R.call_tir(cls.transpose10, (reshape41,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            permute_dims90 = R.call_tir(cls.transpose10, (reshape42,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            lv86 = R.call_tir(cls.fused_NT_matmul1_maximum2_minimum2, (lv82, permute_dims89), out_sinfo=R.Tensor((1, 6, 1, total_seq_len), dtype="float32"))
            softmax8 = R.call_tir(cls.softmax1, (lv86,), out_sinfo=R.Tensor((1, 6, 1, total_seq_len), dtype="float32"))
            matmul64 = R.call_tir(cls.matmul13, (softmax8, permute_dims90), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv87 = R.call_tir(cls.fused_transpose11_reshape7, (matmul64,), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv88 = R.call_tir(cls.fused_NT_matmul_add7_add6, (lv87, model_decoder_layers_2_self_attn_out_proj_weight1, model_decoder_layers_2_self_attn_out_proj_bias1, lv80), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            layer_norm16 = R.call_tir(cls.layer_norm1, (lv88, model_decoder_layers_2_encoder_attn_layer_norm_weight1, model_decoder_layers_2_encoder_attn_layer_norm_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv89 = R.call_tir(cls.fused_NT_matmul_add7_multiply1, (layer_norm16, model_decoder_layers_2_encoder_attn_q_proj_weight1, model_decoder_layers_2_encoder_attn_q_proj_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv90 = R.call_tir(cls.fused_reshape5_transpose9, (lv89,), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv58_1 = R.call_tir(cls.NT_matmul2, (encoder_hidden_states, model_decoder_layers_2_encoder_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape45 = R.call_tir(cls.reshape1, (lv58_1,), out_sinfo=R.Tensor((1, 1500, 6, 64), dtype="float32"))
            lv91 = R.call_tir(cls.fused_NT_matmul2_add3, (encoder_hidden_states, model_decoder_layers_2_encoder_attn_v_proj_weight1, model_decoder_layers_2_encoder_attn_v_proj_bias1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape46 = R.call_tir(cls.reshape1, (lv91,), out_sinfo=R.Tensor((1, 1500, 6, 64), dtype="float32"))
            permute_dims98 = R.call_tir(cls.transpose7, (reshape45,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            permute_dims99 = R.call_tir(cls.transpose7, (reshape46,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv92 = R.call_tir(cls.fused_NT_matmul3_maximum3_minimum3, (lv90, permute_dims98), out_sinfo=R.Tensor((1, 6, 1, 1500), dtype="float32"))
            softmax9 = R.call_tir(cls.softmax2, (lv92,), out_sinfo=R.Tensor((1, 6, 1, 1500), dtype="float32"))
            matmul70 = R.call_tir(cls.matmul14, (softmax9, permute_dims99), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv93 = R.call_tir(cls.fused_transpose11_reshape7, (matmul70,), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv94 = R.call_tir(cls.fused_NT_matmul_add7_add6, (lv93, model_decoder_layers_2_encoder_attn_out_proj_weight1, model_decoder_layers_2_encoder_attn_out_proj_bias1, lv88), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            layer_norm17 = R.call_tir(cls.layer_norm1, (lv94, model_decoder_layers_2_final_layer_norm_weight1, model_decoder_layers_2_final_layer_norm_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv95 = R.call_tir(cls.fused_NT_matmul4_add8_gelu3, (layer_norm17, model_decoder_layers_2_fc1_weight1, model_decoder_layers_2_fc1_bias1), out_sinfo=R.Tensor((1, 1, 1536), dtype="float32"))
            lv96 = R.call_tir(cls.fused_NT_matmul5_add7_add6, (lv95, model_decoder_layers_2_fc2_weight1, model_decoder_layers_2_fc2_bias1, lv94), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            layer_norm18 = R.call_tir(cls.layer_norm1, (lv96, model_decoder_layers_3_self_attn_layer_norm_weight1, model_decoder_layers_3_self_attn_layer_norm_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv97 = R.call_tir(cls.fused_NT_matmul_add7_multiply1, (layer_norm18, model_decoder_layers_3_self_attn_q_proj_weight1, model_decoder_layers_3_self_attn_q_proj_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv98 = R.call_tir(cls.fused_reshape5_transpose9, (lv97,), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv65_1 = R.call_tir(cls.NT_matmul, (layer_norm18, model_decoder_layers_3_self_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv99 = R.call_tir(cls.fused_reshape5_squeeze, (lv65_1,), out_sinfo=R.Tensor((1, 6, 64), dtype="float32"))
            lv100 = R.call_tir(cls.fused_NT_matmul_add7, (layer_norm18, model_decoder_layers_3_self_attn_v_proj_weight1, model_decoder_layers_3_self_attn_v_proj_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv101 = R.call_tir(cls.fused_reshape5_squeeze, (lv100,), out_sinfo=R.Tensor((1, 6, 64), dtype="float32"))
            lv34_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_3_self_attn_k_cache2, lv99, sinfo_args=(R.Object,))
            lv35: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_3_self_attn_v_cache2, lv101, sinfo_args=(R.Object,))
            lv36: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv34_1, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape51 = R.call_tir(cls.reshape6, (lv36,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            lv37: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv35, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape52 = R.call_tir(cls.reshape6, (lv37,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            permute_dims109 = R.call_tir(cls.transpose10, (reshape51,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            permute_dims110 = R.call_tir(cls.transpose10, (reshape52,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            lv102 = R.call_tir(cls.fused_NT_matmul1_maximum2_minimum2, (lv98, permute_dims109), out_sinfo=R.Tensor((1, 6, 1, total_seq_len), dtype="float32"))
            softmax10 = R.call_tir(cls.softmax1, (lv102,), out_sinfo=R.Tensor((1, 6, 1, total_seq_len), dtype="float32"))
            matmul78 = R.call_tir(cls.matmul13, (softmax10, permute_dims110), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv103 = R.call_tir(cls.fused_transpose11_reshape7, (matmul78,), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv104 = R.call_tir(cls.fused_NT_matmul_add7_add6, (lv103, model_decoder_layers_3_self_attn_out_proj_weight1, model_decoder_layers_3_self_attn_out_proj_bias1, lv96), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            layer_norm19 = R.call_tir(cls.layer_norm1, (lv104, model_decoder_layers_3_encoder_attn_layer_norm_weight1, model_decoder_layers_3_encoder_attn_layer_norm_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv105 = R.call_tir(cls.fused_NT_matmul_add7_multiply1, (layer_norm19, model_decoder_layers_3_encoder_attn_q_proj_weight1, model_decoder_layers_3_encoder_attn_q_proj_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv106 = R.call_tir(cls.fused_reshape5_transpose9, (lv105,), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv70_1 = R.call_tir(cls.NT_matmul2, (encoder_hidden_states, model_decoder_layers_3_encoder_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape55 = R.call_tir(cls.reshape1, (lv70_1,), out_sinfo=R.Tensor((1, 1500, 6, 64), dtype="float32"))
            lv107 = R.call_tir(cls.fused_NT_matmul2_add3, (encoder_hidden_states, model_decoder_layers_3_encoder_attn_v_proj_weight1, model_decoder_layers_3_encoder_attn_v_proj_bias1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape56 = R.call_tir(cls.reshape1, (lv107,), out_sinfo=R.Tensor((1, 1500, 6, 64), dtype="float32"))
            permute_dims118 = R.call_tir(cls.transpose7, (reshape55,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            permute_dims119 = R.call_tir(cls.transpose7, (reshape56,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv108 = R.call_tir(cls.fused_NT_matmul3_maximum3_minimum3, (lv106, permute_dims118), out_sinfo=R.Tensor((1, 6, 1, 1500), dtype="float32"))
            softmax11 = R.call_tir(cls.softmax2, (lv108,), out_sinfo=R.Tensor((1, 6, 1, 1500), dtype="float32"))
            matmul84 = R.call_tir(cls.matmul14, (softmax11, permute_dims119), out_sinfo=R.Tensor((1, 6, 1, 64), dtype="float32"))
            lv109 = R.call_tir(cls.fused_transpose11_reshape7, (matmul84,), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv110 = R.call_tir(cls.fused_NT_matmul_add7_add6, (lv109, model_decoder_layers_3_encoder_attn_out_proj_weight1, model_decoder_layers_3_encoder_attn_out_proj_bias1, lv104), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            layer_norm20 = R.call_tir(cls.layer_norm1, (lv110, model_decoder_layers_3_final_layer_norm_weight1, model_decoder_layers_3_final_layer_norm_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv111 = R.call_tir(cls.fused_NT_matmul4_add8_gelu3, (layer_norm20, model_decoder_layers_3_fc1_weight1, model_decoder_layers_3_fc1_bias1), out_sinfo=R.Tensor((1, 1, 1536), dtype="float32"))
            lv112 = R.call_tir(cls.fused_NT_matmul5_add7_add6, (lv111, model_decoder_layers_3_fc2_weight1, model_decoder_layers_3_fc2_bias1, lv110), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            layer_norm21 = R.call_tir(cls.layer_norm1, (lv112, model_decoder_layer_norm_weight1, model_decoder_layer_norm_bias1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            lv76_1 = R.call_tir(cls.NT_matmul6, (layer_norm21, proj_out_weight1), out_sinfo=R.Tensor((1, 1, 51865), dtype="float32"))
            gv2: R.Tuple(R.Tuple(R.Tensor((1, 1, 51865), dtype="float32"), R.Tuple(R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")), R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")), R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")), R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")))), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)) = (lv76_1, ((reshape25, reshape26), (reshape35, reshape36), (reshape45, reshape46), (reshape55, reshape56))), (lv22, lv23, model_decoder_layers_0_encoder_attn_k_cache2, model_decoder_layers_0_encoder_attn_v_cache2, lv26, lv27, model_decoder_layers_1_encoder_attn_k_cache2, model_decoder_layers_1_encoder_attn_v_cache2, lv30, lv31, model_decoder_layers_2_encoder_attn_k_cache2, model_decoder_layers_2_encoder_attn_v_cache2, lv34_1, lv35, model_decoder_layers_3_encoder_attn_k_cache2, model_decoder_layers_3_encoder_attn_v_cache2)
            R.output(gv2)
        return gv2

    @R.function
    def encode(input_ids: R.Tensor((1, 80, 3000), dtype="float32"), packed_effects: R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object), packed_params: R.Tuple(R.Tensor((384, 80, 3), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384, 3), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1500, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((51865, 384), dtype="float32"), R.Tensor((448, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((51865, 384), dtype="float32"))) -> R.Tuple(R.Tensor((1, 1500, 384), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)):
        R.func_attr({"num_input": 2, "tir_var_upper_bound": {"seq_len": 448, "total_seq_len": 448}})
        cls = Module
        with R.dataflow():
            model_decoder_layers_0_self_attn_k_cache1: R.Object = packed_effects[0]
            model_decoder_layers_0_self_attn_v_cache1: R.Object = packed_effects[1]
            model_decoder_layers_0_encoder_attn_k_cache1: R.Object = packed_effects[2]
            model_decoder_layers_0_encoder_attn_v_cache1: R.Object = packed_effects[3]
            model_decoder_layers_1_self_attn_k_cache1: R.Object = packed_effects[4]
            model_decoder_layers_1_self_attn_v_cache1: R.Object = packed_effects[5]
            model_decoder_layers_1_encoder_attn_k_cache1: R.Object = packed_effects[6]
            model_decoder_layers_1_encoder_attn_v_cache1: R.Object = packed_effects[7]
            model_decoder_layers_2_self_attn_k_cache1: R.Object = packed_effects[8]
            model_decoder_layers_2_self_attn_v_cache1: R.Object = packed_effects[9]
            model_decoder_layers_2_encoder_attn_k_cache1: R.Object = packed_effects[10]
            model_decoder_layers_2_encoder_attn_v_cache1: R.Object = packed_effects[11]
            model_decoder_layers_3_self_attn_k_cache1: R.Object = packed_effects[12]
            model_decoder_layers_3_self_attn_v_cache1: R.Object = packed_effects[13]
            model_decoder_layers_3_encoder_attn_k_cache1: R.Object = packed_effects[14]
            model_decoder_layers_3_encoder_attn_v_cache1: R.Object = packed_effects[15]
            model_encoder_conv1_weight: R.Tensor((384, 80, 3), dtype="float32") = packed_params[0]
            lv: R.Tensor((384,), dtype="float32") = packed_params[1]
            lv1 = R.call_tir(cls.fused_reshape, (lv,), out_sinfo=R.Tensor((1, 384, 1), dtype="float32"))
            model_encoder_conv2_weight: R.Tensor((384, 384, 3), dtype="float32") = packed_params[2]
            lv2: R.Tensor((384,), dtype="float32") = packed_params[3]
            lv3 = R.call_tir(cls.fused_reshape, (lv2,), out_sinfo=R.Tensor((1, 384, 1), dtype="float32"))
            lv4 = R.call_tir(cls.fused_conv1d_add_gelu, (input_ids, model_encoder_conv1_weight, lv1), out_sinfo=R.Tensor((1, 384, 3000), dtype="float32"))
            lv5 = R.call_tir(cls.fused_conv1d1_add1_gelu1, (lv4, model_encoder_conv2_weight, lv3), out_sinfo=R.Tensor((1, 384, 1500), dtype="float32"))
            lv6: R.Tensor((1500, 384), dtype="float32") = packed_params[4]
            lv7 = R.call_tir(cls.fused_transpose6_add2, (lv6, lv5), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            model_encoder_layers_0_self_attn_k_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[5]
            model_encoder_layers_0_self_attn_v_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[6]
            model_encoder_layers_0_self_attn_v_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[7]
            model_encoder_layers_0_self_attn_q_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[8]
            model_encoder_layers_0_self_attn_q_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[9]
            model_encoder_layers_0_self_attn_out_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[10]
            model_encoder_layers_0_self_attn_out_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[11]
            model_encoder_layers_0_self_attn_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[12]
            model_encoder_layers_0_self_attn_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[13]
            model_encoder_layers_0_fc1_weight: R.Tensor((1536, 384), dtype="float32") = packed_params[14]
            model_encoder_layers_0_fc1_bias: R.Tensor((1536,), dtype="float32") = packed_params[15]
            model_encoder_layers_0_fc2_weight: R.Tensor((384, 1536), dtype="float32") = packed_params[16]
            model_encoder_layers_0_fc2_bias: R.Tensor((384,), dtype="float32") = packed_params[17]
            model_encoder_layers_0_final_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[18]
            model_encoder_layers_0_final_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[19]
            model_encoder_layers_1_self_attn_k_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[20]
            model_encoder_layers_1_self_attn_v_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[21]
            model_encoder_layers_1_self_attn_v_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[22]
            model_encoder_layers_1_self_attn_q_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[23]
            model_encoder_layers_1_self_attn_q_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[24]
            model_encoder_layers_1_self_attn_out_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[25]
            model_encoder_layers_1_self_attn_out_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[26]
            model_encoder_layers_1_self_attn_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[27]
            model_encoder_layers_1_self_attn_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[28]
            model_encoder_layers_1_fc1_weight: R.Tensor((1536, 384), dtype="float32") = packed_params[29]
            model_encoder_layers_1_fc1_bias: R.Tensor((1536,), dtype="float32") = packed_params[30]
            model_encoder_layers_1_fc2_weight: R.Tensor((384, 1536), dtype="float32") = packed_params[31]
            model_encoder_layers_1_fc2_bias: R.Tensor((384,), dtype="float32") = packed_params[32]
            model_encoder_layers_1_final_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[33]
            model_encoder_layers_1_final_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[34]
            model_encoder_layers_2_self_attn_k_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[35]
            model_encoder_layers_2_self_attn_v_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[36]
            model_encoder_layers_2_self_attn_v_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[37]
            model_encoder_layers_2_self_attn_q_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[38]
            model_encoder_layers_2_self_attn_q_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[39]
            model_encoder_layers_2_self_attn_out_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[40]
            model_encoder_layers_2_self_attn_out_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[41]
            model_encoder_layers_2_self_attn_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[42]
            model_encoder_layers_2_self_attn_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[43]
            model_encoder_layers_2_fc1_weight: R.Tensor((1536, 384), dtype="float32") = packed_params[44]
            model_encoder_layers_2_fc1_bias: R.Tensor((1536,), dtype="float32") = packed_params[45]
            model_encoder_layers_2_fc2_weight: R.Tensor((384, 1536), dtype="float32") = packed_params[46]
            model_encoder_layers_2_fc2_bias: R.Tensor((384,), dtype="float32") = packed_params[47]
            model_encoder_layers_2_final_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[48]
            model_encoder_layers_2_final_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[49]
            model_encoder_layers_3_self_attn_k_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[50]
            model_encoder_layers_3_self_attn_v_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[51]
            model_encoder_layers_3_self_attn_v_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[52]
            model_encoder_layers_3_self_attn_q_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[53]
            model_encoder_layers_3_self_attn_q_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[54]
            model_encoder_layers_3_self_attn_out_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[55]
            model_encoder_layers_3_self_attn_out_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[56]
            model_encoder_layers_3_self_attn_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[57]
            model_encoder_layers_3_self_attn_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[58]
            model_encoder_layers_3_fc1_weight: R.Tensor((1536, 384), dtype="float32") = packed_params[59]
            model_encoder_layers_3_fc1_bias: R.Tensor((1536,), dtype="float32") = packed_params[60]
            model_encoder_layers_3_fc2_weight: R.Tensor((384, 1536), dtype="float32") = packed_params[61]
            model_encoder_layers_3_fc2_bias: R.Tensor((384,), dtype="float32") = packed_params[62]
            model_encoder_layers_3_final_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[63]
            model_encoder_layers_3_final_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[64]
            model_encoder_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[65]
            model_encoder_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[66]
            layer_norm = R.call_tir(cls.layer_norm, (lv7, model_encoder_layers_0_self_attn_layer_norm_weight, model_encoder_layers_0_self_attn_layer_norm_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv8 = R.call_tir(cls.fused_NT_matmul2_add3_multiply, (layer_norm, model_encoder_layers_0_self_attn_q_proj_weight, model_encoder_layers_0_self_attn_q_proj_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv9 = R.call_tir(cls.fused_reshape1_transpose7, (lv8,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv1_1 = R.call_tir(cls.NT_matmul2, (layer_norm, model_encoder_layers_0_self_attn_k_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv10 = R.call_tir(cls.fused_reshape1_transpose7, (lv1_1,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv11 = R.call_tir(cls.fused_NT_matmul2_add3, (layer_norm, model_encoder_layers_0_self_attn_v_proj_weight, model_encoder_layers_0_self_attn_v_proj_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv12 = R.call_tir(cls.fused_reshape1_transpose7, (lv11,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv13 = R.call_tir(cls.fused_NT_matmul7_maximum_minimum, (lv9, lv10), out_sinfo=R.Tensor((1, 6, 1500, 1500), dtype="float32"))
            softmax = R.call_tir(cls.softmax, (lv13,), out_sinfo=R.Tensor((1, 6, 1500, 1500), dtype="float32"))
            matmul4 = R.call_tir(cls.matmul11, (softmax, lv12), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv14 = R.call_tir(cls.fused_transpose8_reshape2, (matmul4,), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv15 = R.call_tir(cls.fused_NT_matmul2_add3_add4, (lv14, model_encoder_layers_0_self_attn_out_proj_weight, model_encoder_layers_0_self_attn_out_proj_bias, lv7), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            layer_norm1 = R.call_tir(cls.layer_norm, (lv15, model_encoder_layers_0_final_layer_norm_weight, model_encoder_layers_0_final_layer_norm_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv16 = R.call_tir(cls.fused_NT_matmul8_add5_gelu2, (layer_norm1, model_encoder_layers_0_fc1_weight, model_encoder_layers_0_fc1_bias), out_sinfo=R.Tensor((1, 1500, 1536), dtype="float32"))
            lv17 = R.call_tir(cls.fused_NT_matmul9_add3_add4_maximum1_minimum1, (lv16, model_encoder_layers_0_fc2_weight, model_encoder_layers_0_fc2_bias, lv15), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            layer_norm2 = R.call_tir(cls.layer_norm, (lv17, model_encoder_layers_1_self_attn_layer_norm_weight, model_encoder_layers_1_self_attn_layer_norm_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv18 = R.call_tir(cls.fused_NT_matmul2_add3_multiply, (layer_norm2, model_encoder_layers_1_self_attn_q_proj_weight, model_encoder_layers_1_self_attn_q_proj_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv19 = R.call_tir(cls.fused_reshape1_transpose7, (lv18,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv8_1 = R.call_tir(cls.NT_matmul2, (layer_norm2, model_encoder_layers_1_self_attn_k_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv20 = R.call_tir(cls.fused_reshape1_transpose7, (lv8_1,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv21 = R.call_tir(cls.fused_NT_matmul2_add3, (layer_norm2, model_encoder_layers_1_self_attn_v_proj_weight, model_encoder_layers_1_self_attn_v_proj_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv22 = R.call_tir(cls.fused_reshape1_transpose7, (lv21,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv23 = R.call_tir(cls.fused_NT_matmul7_maximum_minimum, (lv19, lv20), out_sinfo=R.Tensor((1, 6, 1500, 1500), dtype="float32"))
            softmax1 = R.call_tir(cls.softmax, (lv23,), out_sinfo=R.Tensor((1, 6, 1500, 1500), dtype="float32"))
            matmul12 = R.call_tir(cls.matmul11, (softmax1, lv22), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv24 = R.call_tir(cls.fused_transpose8_reshape2, (matmul12,), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv25 = R.call_tir(cls.fused_NT_matmul2_add3_add4, (lv24, model_encoder_layers_1_self_attn_out_proj_weight, model_encoder_layers_1_self_attn_out_proj_bias, lv17), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            layer_norm3 = R.call_tir(cls.layer_norm, (lv25, model_encoder_layers_1_final_layer_norm_weight, model_encoder_layers_1_final_layer_norm_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv26 = R.call_tir(cls.fused_NT_matmul8_add5_gelu2, (layer_norm3, model_encoder_layers_1_fc1_weight, model_encoder_layers_1_fc1_bias), out_sinfo=R.Tensor((1, 1500, 1536), dtype="float32"))
            lv27 = R.call_tir(cls.fused_NT_matmul9_add3_add4_maximum1_minimum1, (lv26, model_encoder_layers_1_fc2_weight, model_encoder_layers_1_fc2_bias, lv25), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            layer_norm4 = R.call_tir(cls.layer_norm, (lv27, model_encoder_layers_2_self_attn_layer_norm_weight, model_encoder_layers_2_self_attn_layer_norm_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv28 = R.call_tir(cls.fused_NT_matmul2_add3_multiply, (layer_norm4, model_encoder_layers_2_self_attn_q_proj_weight, model_encoder_layers_2_self_attn_q_proj_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv29 = R.call_tir(cls.fused_reshape1_transpose7, (lv28,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv15_1 = R.call_tir(cls.NT_matmul2, (layer_norm4, model_encoder_layers_2_self_attn_k_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv30 = R.call_tir(cls.fused_reshape1_transpose7, (lv15_1,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv31 = R.call_tir(cls.fused_NT_matmul2_add3, (layer_norm4, model_encoder_layers_2_self_attn_v_proj_weight, model_encoder_layers_2_self_attn_v_proj_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv32 = R.call_tir(cls.fused_reshape1_transpose7, (lv31,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv33 = R.call_tir(cls.fused_NT_matmul7_maximum_minimum, (lv29, lv30), out_sinfo=R.Tensor((1, 6, 1500, 1500), dtype="float32"))
            softmax2 = R.call_tir(cls.softmax, (lv33,), out_sinfo=R.Tensor((1, 6, 1500, 1500), dtype="float32"))
            matmul20 = R.call_tir(cls.matmul11, (softmax2, lv32), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv34 = R.call_tir(cls.fused_transpose8_reshape2, (matmul20,), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv35 = R.call_tir(cls.fused_NT_matmul2_add3_add4, (lv34, model_encoder_layers_2_self_attn_out_proj_weight, model_encoder_layers_2_self_attn_out_proj_bias, lv27), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            layer_norm5 = R.call_tir(cls.layer_norm, (lv35, model_encoder_layers_2_final_layer_norm_weight, model_encoder_layers_2_final_layer_norm_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv36 = R.call_tir(cls.fused_NT_matmul8_add5_gelu2, (layer_norm5, model_encoder_layers_2_fc1_weight, model_encoder_layers_2_fc1_bias), out_sinfo=R.Tensor((1, 1500, 1536), dtype="float32"))
            lv37 = R.call_tir(cls.fused_NT_matmul9_add3_add4_maximum1_minimum1, (lv36, model_encoder_layers_2_fc2_weight, model_encoder_layers_2_fc2_bias, lv35), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            layer_norm6 = R.call_tir(cls.layer_norm, (lv37, model_encoder_layers_3_self_attn_layer_norm_weight, model_encoder_layers_3_self_attn_layer_norm_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv38 = R.call_tir(cls.fused_NT_matmul2_add3_multiply, (layer_norm6, model_encoder_layers_3_self_attn_q_proj_weight, model_encoder_layers_3_self_attn_q_proj_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv39 = R.call_tir(cls.fused_reshape1_transpose7, (lv38,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv22_1 = R.call_tir(cls.NT_matmul2, (layer_norm6, model_encoder_layers_3_self_attn_k_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv40 = R.call_tir(cls.fused_reshape1_transpose7, (lv22_1,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv41 = R.call_tir(cls.fused_NT_matmul2_add3, (layer_norm6, model_encoder_layers_3_self_attn_v_proj_weight, model_encoder_layers_3_self_attn_v_proj_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv42 = R.call_tir(cls.fused_reshape1_transpose7, (lv41,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv43 = R.call_tir(cls.fused_NT_matmul7_maximum_minimum, (lv39, lv40), out_sinfo=R.Tensor((1, 6, 1500, 1500), dtype="float32"))
            softmax3 = R.call_tir(cls.softmax, (lv43,), out_sinfo=R.Tensor((1, 6, 1500, 1500), dtype="float32"))
            matmul28 = R.call_tir(cls.matmul11, (softmax3, lv42), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv44 = R.call_tir(cls.fused_transpose8_reshape2, (matmul28,), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv45 = R.call_tir(cls.fused_NT_matmul2_add3_add4, (lv44, model_encoder_layers_3_self_attn_out_proj_weight, model_encoder_layers_3_self_attn_out_proj_bias, lv37), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            layer_norm7 = R.call_tir(cls.layer_norm, (lv45, model_encoder_layers_3_final_layer_norm_weight, model_encoder_layers_3_final_layer_norm_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            lv46 = R.call_tir(cls.fused_NT_matmul8_add5_gelu2, (layer_norm7, model_encoder_layers_3_fc1_weight, model_encoder_layers_3_fc1_bias), out_sinfo=R.Tensor((1, 1500, 1536), dtype="float32"))
            lv47 = R.call_tir(cls.fused_NT_matmul9_add3_add4_maximum1_minimum1, (lv46, model_encoder_layers_3_fc2_weight, model_encoder_layers_3_fc2_bias, lv45), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            layer_norm8 = R.call_tir(cls.layer_norm, (lv47, model_encoder_layer_norm_weight, model_encoder_layer_norm_bias), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            gv1: R.Tuple(R.Tensor((1, 1500, 384), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)) = layer_norm8, (model_decoder_layers_0_self_attn_k_cache1, model_decoder_layers_0_self_attn_v_cache1, model_decoder_layers_0_encoder_attn_k_cache1, model_decoder_layers_0_encoder_attn_v_cache1, model_decoder_layers_1_self_attn_k_cache1, model_decoder_layers_1_self_attn_v_cache1, model_decoder_layers_1_encoder_attn_k_cache1, model_decoder_layers_1_encoder_attn_v_cache1, model_decoder_layers_2_self_attn_k_cache1, model_decoder_layers_2_self_attn_v_cache1, model_decoder_layers_2_encoder_attn_k_cache1, model_decoder_layers_2_encoder_attn_v_cache1, model_decoder_layers_3_self_attn_k_cache1, model_decoder_layers_3_self_attn_v_cache1, model_decoder_layers_3_encoder_attn_k_cache1, model_decoder_layers_3_encoder_attn_v_cache1)
            R.output(gv1)
        return gv1

    @R.function
    def prefill(input_ids: R.Tensor((1, "seq_len"), dtype="int32"), total_seq_len_1: R.Shape(["total_seq_len"]), cached_encoder_key_value: R.Tuple(R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")), R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")), R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")), R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32"))), packed_effects: R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object), packed_params: R.Tuple(R.Tensor((384, 80, 3), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384, 3), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1500, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((51865, 384), dtype="float32"), R.Tensor((448, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((51865, 384), dtype="float32"))) -> R.Tuple(R.Tensor((1, "seq_len", 51865), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)):
        seq_len = T.int64()
        total_seq_len = T.int64()
        R.func_attr({"num_input": 4, "tir_var_upper_bound": {"seq_len": 448, "total_seq_len": 448}})
        cls = Module
        with R.dataflow():
            model_decoder_layers_0_self_attn_k_cache3: R.Object = packed_effects[0]
            model_decoder_layers_0_self_attn_v_cache3: R.Object = packed_effects[1]
            model_decoder_layers_0_encoder_attn_k_cache3: R.Object = packed_effects[2]
            model_decoder_layers_0_encoder_attn_v_cache3: R.Object = packed_effects[3]
            model_decoder_layers_1_self_attn_k_cache3: R.Object = packed_effects[4]
            model_decoder_layers_1_self_attn_v_cache3: R.Object = packed_effects[5]
            model_decoder_layers_1_encoder_attn_k_cache3: R.Object = packed_effects[6]
            model_decoder_layers_1_encoder_attn_v_cache3: R.Object = packed_effects[7]
            model_decoder_layers_2_self_attn_k_cache3: R.Object = packed_effects[8]
            model_decoder_layers_2_self_attn_v_cache3: R.Object = packed_effects[9]
            model_decoder_layers_2_encoder_attn_k_cache3: R.Object = packed_effects[10]
            model_decoder_layers_2_encoder_attn_v_cache3: R.Object = packed_effects[11]
            model_decoder_layers_3_self_attn_k_cache3: R.Object = packed_effects[12]
            model_decoder_layers_3_self_attn_v_cache3: R.Object = packed_effects[13]
            model_decoder_layers_3_encoder_attn_k_cache3: R.Object = packed_effects[14]
            model_decoder_layers_3_encoder_attn_v_cache3: R.Object = packed_effects[15]
            model_decoder_embed_tokens_weight2: R.Tensor((51865, 384), dtype="float32") = packed_params[67]
            model_decoder_embed_positions_weight2: R.Tensor((448, 384), dtype="float32") = packed_params[68]
            model_decoder_layers_0_self_attn_k_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[69]
            model_decoder_layers_0_self_attn_v_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[70]
            model_decoder_layers_0_self_attn_v_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[71]
            model_decoder_layers_0_self_attn_q_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[72]
            model_decoder_layers_0_self_attn_q_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[73]
            model_decoder_layers_0_self_attn_out_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[74]
            model_decoder_layers_0_self_attn_out_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[75]
            model_decoder_layers_0_self_attn_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[76]
            model_decoder_layers_0_self_attn_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[77]
            model_decoder_layers_0_encoder_attn_q_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[81]
            model_decoder_layers_0_encoder_attn_q_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[82]
            model_decoder_layers_0_encoder_attn_out_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[83]
            model_decoder_layers_0_encoder_attn_out_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[84]
            model_decoder_layers_0_encoder_attn_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[85]
            model_decoder_layers_0_encoder_attn_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[86]
            model_decoder_layers_0_fc1_weight2: R.Tensor((1536, 384), dtype="float32") = packed_params[87]
            model_decoder_layers_0_fc1_bias2: R.Tensor((1536,), dtype="float32") = packed_params[88]
            model_decoder_layers_0_fc2_weight2: R.Tensor((384, 1536), dtype="float32") = packed_params[89]
            model_decoder_layers_0_fc2_bias2: R.Tensor((384,), dtype="float32") = packed_params[90]
            model_decoder_layers_0_final_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[91]
            model_decoder_layers_0_final_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[92]
            model_decoder_layers_1_self_attn_k_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[93]
            model_decoder_layers_1_self_attn_v_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[94]
            model_decoder_layers_1_self_attn_v_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[95]
            model_decoder_layers_1_self_attn_q_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[96]
            model_decoder_layers_1_self_attn_q_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[97]
            model_decoder_layers_1_self_attn_out_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[98]
            model_decoder_layers_1_self_attn_out_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[99]
            model_decoder_layers_1_self_attn_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[100]
            model_decoder_layers_1_self_attn_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[101]
            model_decoder_layers_1_encoder_attn_q_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[105]
            model_decoder_layers_1_encoder_attn_q_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[106]
            model_decoder_layers_1_encoder_attn_out_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[107]
            model_decoder_layers_1_encoder_attn_out_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[108]
            model_decoder_layers_1_encoder_attn_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[109]
            model_decoder_layers_1_encoder_attn_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[110]
            model_decoder_layers_1_fc1_weight2: R.Tensor((1536, 384), dtype="float32") = packed_params[111]
            model_decoder_layers_1_fc1_bias2: R.Tensor((1536,), dtype="float32") = packed_params[112]
            model_decoder_layers_1_fc2_weight2: R.Tensor((384, 1536), dtype="float32") = packed_params[113]
            model_decoder_layers_1_fc2_bias2: R.Tensor((384,), dtype="float32") = packed_params[114]
            model_decoder_layers_1_final_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[115]
            model_decoder_layers_1_final_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[116]
            model_decoder_layers_2_self_attn_k_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[117]
            model_decoder_layers_2_self_attn_v_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[118]
            model_decoder_layers_2_self_attn_v_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[119]
            model_decoder_layers_2_self_attn_q_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[120]
            model_decoder_layers_2_self_attn_q_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[121]
            model_decoder_layers_2_self_attn_out_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[122]
            model_decoder_layers_2_self_attn_out_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[123]
            model_decoder_layers_2_self_attn_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[124]
            model_decoder_layers_2_self_attn_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[125]
            model_decoder_layers_2_encoder_attn_q_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[129]
            model_decoder_layers_2_encoder_attn_q_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[130]
            model_decoder_layers_2_encoder_attn_out_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[131]
            model_decoder_layers_2_encoder_attn_out_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[132]
            model_decoder_layers_2_encoder_attn_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[133]
            model_decoder_layers_2_encoder_attn_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[134]
            model_decoder_layers_2_fc1_weight2: R.Tensor((1536, 384), dtype="float32") = packed_params[135]
            model_decoder_layers_2_fc1_bias2: R.Tensor((1536,), dtype="float32") = packed_params[136]
            model_decoder_layers_2_fc2_weight2: R.Tensor((384, 1536), dtype="float32") = packed_params[137]
            model_decoder_layers_2_fc2_bias2: R.Tensor((384,), dtype="float32") = packed_params[138]
            model_decoder_layers_2_final_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[139]
            model_decoder_layers_2_final_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[140]
            model_decoder_layers_3_self_attn_k_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[141]
            model_decoder_layers_3_self_attn_v_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[142]
            model_decoder_layers_3_self_attn_v_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[143]
            model_decoder_layers_3_self_attn_q_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[144]
            model_decoder_layers_3_self_attn_q_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[145]
            model_decoder_layers_3_self_attn_out_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[146]
            model_decoder_layers_3_self_attn_out_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[147]
            model_decoder_layers_3_self_attn_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[148]
            model_decoder_layers_3_self_attn_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[149]
            model_decoder_layers_3_encoder_attn_q_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[153]
            model_decoder_layers_3_encoder_attn_q_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[154]
            model_decoder_layers_3_encoder_attn_out_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[155]
            model_decoder_layers_3_encoder_attn_out_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[156]
            model_decoder_layers_3_encoder_attn_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[157]
            model_decoder_layers_3_encoder_attn_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[158]
            model_decoder_layers_3_fc1_weight2: R.Tensor((1536, 384), dtype="float32") = packed_params[159]
            model_decoder_layers_3_fc1_bias2: R.Tensor((1536,), dtype="float32") = packed_params[160]
            model_decoder_layers_3_fc2_weight2: R.Tensor((384, 1536), dtype="float32") = packed_params[161]
            model_decoder_layers_3_fc2_bias2: R.Tensor((384,), dtype="float32") = packed_params[162]
            model_decoder_layers_3_final_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[163]
            model_decoder_layers_3_final_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[164]
            model_decoder_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[165]
            model_decoder_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[166]
            proj_out_weight2: R.Tensor((51865, 384), dtype="float32") = packed_params[167]
            cached_encoder_key_value_0: R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")) = cached_encoder_key_value[0]
            lv113: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_0[0]
            lv114 = R.call_tir(cls.fused_transpose7, (lv113,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv115: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_0[1]
            lv116 = R.call_tir(cls.fused_transpose7, (lv115,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            cached_encoder_key_value_1: R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")) = cached_encoder_key_value[1]
            lv117: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_1[0]
            lv118 = R.call_tir(cls.fused_transpose7, (lv117,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv119: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_1[1]
            lv120 = R.call_tir(cls.fused_transpose7, (lv119,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            cached_encoder_key_value_2: R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")) = cached_encoder_key_value[2]
            lv121: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_2[0]
            lv122 = R.call_tir(cls.fused_transpose7, (lv121,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv123: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_2[1]
            lv124 = R.call_tir(cls.fused_transpose7, (lv123,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            cached_encoder_key_value_3: R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")) = cached_encoder_key_value[3]
            lv125: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_3[0]
            lv126 = R.call_tir(cls.fused_transpose7, (lv125,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            lv127: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_3[1]
            lv128 = R.call_tir(cls.fused_transpose7, (lv127,), out_sinfo=R.Tensor((1, 6, 1500, 64), dtype="float32"))
            reshape58 = R.call_tir(cls.reshape8, (input_ids,), out_sinfo=R.Tensor((seq_len,), dtype="int32"))
            take1 = R.call_tir(cls.take1, (model_decoder_embed_tokens_weight2, reshape58), out_sinfo=R.Tensor((seq_len, 384), dtype="float32"))
            reshape59 = R.call_tir(cls.reshape9, (take1,), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv38 = R.call_tir(cls.position_embedding1, (input_ids, model_decoder_embed_positions_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"), tir_vars=R.shape([total_seq_len]))
            add74 = R.call_tir(cls.add9, (reshape59, lv38), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            layer_norm22 = R.call_tir(cls.layer_norm2, (add74, model_decoder_layers_0_self_attn_layer_norm_weight2, model_decoder_layers_0_self_attn_layer_norm_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv129 = R.call_tir(cls.fused_NT_matmul10_add10_multiply2, (layer_norm22, model_decoder_layers_0_self_attn_q_proj_weight2, model_decoder_layers_0_self_attn_q_proj_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape60 = R.call_tir(cls.reshape10, (lv129,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            lv78 = R.call_tir(cls.NT_matmul10, (layer_norm22, model_decoder_layers_0_self_attn_k_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape61 = R.call_tir(cls.reshape10, (lv78,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            lv130 = R.call_tir(cls.fused_NT_matmul10_add10, (layer_norm22, model_decoder_layers_0_self_attn_v_proj_weight2, model_decoder_layers_0_self_attn_v_proj_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape62 = R.call_tir(cls.reshape10, (lv130,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            squeeze8 = R.call_tir(cls.squeeze1, (reshape61,), out_sinfo=R.Tensor((seq_len, 6, 64), dtype="float32"))
            lv39: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_0_self_attn_k_cache3, squeeze8, sinfo_args=(R.Object,))
            squeeze9 = R.call_tir(cls.squeeze1, (reshape62,), out_sinfo=R.Tensor((seq_len, 6, 64), dtype="float32"))
            lv40: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_0_self_attn_v_cache3, squeeze9, sinfo_args=(R.Object,))
            lv41: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv39, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape63 = R.call_tir(cls.reshape6, (lv41,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            lv42: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv40, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape64 = R.call_tir(cls.reshape6, (lv42,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            permute_dims129 = R.call_tir(cls.transpose10, (reshape60,), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            permute_dims130 = R.call_tir(cls.transpose10, (reshape63,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            permute_dims131 = R.call_tir(cls.transpose10, (reshape64,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            lv131 = R.call_tir(cls.fused_NT_matmul11_maximum4_minimum4, (permute_dims129, permute_dims130), out_sinfo=R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32"))
            softmax12 = R.call_tir(cls.softmax3, (lv131,), out_sinfo=R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32"))
            matmul93 = R.call_tir(cls.matmul19, (softmax12, permute_dims131), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            permute_dims133 = R.call_tir(cls.transpose12, (matmul93,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            reshape65 = R.call_tir(cls.reshape11, (permute_dims133,), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv132 = R.call_tir(cls.fused_NT_matmul10_add10_add9, (reshape65, model_decoder_layers_0_self_attn_out_proj_weight2, model_decoder_layers_0_self_attn_out_proj_bias2, add74), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            layer_norm23 = R.call_tir(cls.layer_norm2, (lv132, model_decoder_layers_0_encoder_attn_layer_norm_weight2, model_decoder_layers_0_encoder_attn_layer_norm_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv133 = R.call_tir(cls.fused_NT_matmul10_add10_multiply2, (layer_norm23, model_decoder_layers_0_encoder_attn_q_proj_weight2, model_decoder_layers_0_encoder_attn_q_proj_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape66 = R.call_tir(cls.reshape10, (lv133,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            permute_dims136 = R.call_tir(cls.transpose10, (reshape66,), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            lv134 = R.call_tir(cls.fused_NT_matmul12_maximum5_minimum5, (permute_dims136, lv114), out_sinfo=R.Tensor((1, 6, seq_len, 1500), dtype="float32"))
            softmax13 = R.call_tir(cls.softmax4, (lv134,), out_sinfo=R.Tensor((1, 6, seq_len, 1500), dtype="float32"))
            matmul97 = R.call_tir(cls.matmul20, (softmax13, lv116), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            permute_dims140 = R.call_tir(cls.transpose12, (matmul97,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            reshape67 = R.call_tir(cls.reshape11, (permute_dims140,), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv135 = R.call_tir(cls.fused_NT_matmul10_add10_add9, (reshape67, model_decoder_layers_0_encoder_attn_out_proj_weight2, model_decoder_layers_0_encoder_attn_out_proj_bias2, lv132), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            layer_norm24 = R.call_tir(cls.layer_norm2, (lv135, model_decoder_layers_0_final_layer_norm_weight2, model_decoder_layers_0_final_layer_norm_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv136 = R.call_tir(cls.fused_NT_matmul13_add11_gelu4, (layer_norm24, model_decoder_layers_0_fc1_weight2, model_decoder_layers_0_fc1_bias2), out_sinfo=R.Tensor((1, seq_len, 1536), dtype="float32"))
            lv137 = R.call_tir(cls.fused_NT_matmul14_add10_add9, (lv136, model_decoder_layers_0_fc2_weight2, model_decoder_layers_0_fc2_bias2, lv135), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            layer_norm25 = R.call_tir(cls.layer_norm2, (lv137, model_decoder_layers_1_self_attn_layer_norm_weight2, model_decoder_layers_1_self_attn_layer_norm_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv138 = R.call_tir(cls.fused_NT_matmul10_add10_multiply2, (layer_norm25, model_decoder_layers_1_self_attn_q_proj_weight2, model_decoder_layers_1_self_attn_q_proj_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape68 = R.call_tir(cls.reshape10, (lv138,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            lv88 = R.call_tir(cls.NT_matmul10, (layer_norm25, model_decoder_layers_1_self_attn_k_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape69 = R.call_tir(cls.reshape10, (lv88,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            lv139 = R.call_tir(cls.fused_NT_matmul10_add10, (layer_norm25, model_decoder_layers_1_self_attn_v_proj_weight2, model_decoder_layers_1_self_attn_v_proj_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape70 = R.call_tir(cls.reshape10, (lv139,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            squeeze10 = R.call_tir(cls.squeeze1, (reshape69,), out_sinfo=R.Tensor((seq_len, 6, 64), dtype="float32"))
            lv43: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_1_self_attn_k_cache3, squeeze10, sinfo_args=(R.Object,))
            squeeze11 = R.call_tir(cls.squeeze1, (reshape70,), out_sinfo=R.Tensor((seq_len, 6, 64), dtype="float32"))
            lv44: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_1_self_attn_v_cache3, squeeze11, sinfo_args=(R.Object,))
            lv45: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv43, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape71 = R.call_tir(cls.reshape6, (lv45,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            lv46: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv44, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape72 = R.call_tir(cls.reshape6, (lv46,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            permute_dims147 = R.call_tir(cls.transpose10, (reshape68,), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            permute_dims148 = R.call_tir(cls.transpose10, (reshape71,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            permute_dims149 = R.call_tir(cls.transpose10, (reshape72,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            lv140 = R.call_tir(cls.fused_NT_matmul11_maximum4_minimum4, (permute_dims147, permute_dims148), out_sinfo=R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32"))
            softmax14 = R.call_tir(cls.softmax3, (lv140,), out_sinfo=R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32"))
            matmul105 = R.call_tir(cls.matmul19, (softmax14, permute_dims149), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            permute_dims151 = R.call_tir(cls.transpose12, (matmul105,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            reshape73 = R.call_tir(cls.reshape11, (permute_dims151,), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv141 = R.call_tir(cls.fused_NT_matmul10_add10_add9, (reshape73, model_decoder_layers_1_self_attn_out_proj_weight2, model_decoder_layers_1_self_attn_out_proj_bias2, lv137), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            layer_norm26 = R.call_tir(cls.layer_norm2, (lv141, model_decoder_layers_1_encoder_attn_layer_norm_weight2, model_decoder_layers_1_encoder_attn_layer_norm_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv142 = R.call_tir(cls.fused_NT_matmul10_add10_multiply2, (layer_norm26, model_decoder_layers_1_encoder_attn_q_proj_weight2, model_decoder_layers_1_encoder_attn_q_proj_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape74 = R.call_tir(cls.reshape10, (lv142,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            permute_dims154 = R.call_tir(cls.transpose10, (reshape74,), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            lv143 = R.call_tir(cls.fused_NT_matmul12_maximum5_minimum5, (permute_dims154, lv118), out_sinfo=R.Tensor((1, 6, seq_len, 1500), dtype="float32"))
            softmax15 = R.call_tir(cls.softmax4, (lv143,), out_sinfo=R.Tensor((1, 6, seq_len, 1500), dtype="float32"))
            matmul109 = R.call_tir(cls.matmul20, (softmax15, lv120), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            permute_dims158 = R.call_tir(cls.transpose12, (matmul109,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            reshape75 = R.call_tir(cls.reshape11, (permute_dims158,), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv144 = R.call_tir(cls.fused_NT_matmul10_add10_add9, (reshape75, model_decoder_layers_1_encoder_attn_out_proj_weight2, model_decoder_layers_1_encoder_attn_out_proj_bias2, lv141), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            layer_norm27 = R.call_tir(cls.layer_norm2, (lv144, model_decoder_layers_1_final_layer_norm_weight2, model_decoder_layers_1_final_layer_norm_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv145 = R.call_tir(cls.fused_NT_matmul13_add11_gelu4, (layer_norm27, model_decoder_layers_1_fc1_weight2, model_decoder_layers_1_fc1_bias2), out_sinfo=R.Tensor((1, seq_len, 1536), dtype="float32"))
            lv146 = R.call_tir(cls.fused_NT_matmul14_add10_add9, (lv145, model_decoder_layers_1_fc2_weight2, model_decoder_layers_1_fc2_bias2, lv144), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            layer_norm28 = R.call_tir(cls.layer_norm2, (lv146, model_decoder_layers_2_self_attn_layer_norm_weight2, model_decoder_layers_2_self_attn_layer_norm_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv147 = R.call_tir(cls.fused_NT_matmul10_add10_multiply2, (layer_norm28, model_decoder_layers_2_self_attn_q_proj_weight2, model_decoder_layers_2_self_attn_q_proj_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape76 = R.call_tir(cls.reshape10, (lv147,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            lv98 = R.call_tir(cls.NT_matmul10, (layer_norm28, model_decoder_layers_2_self_attn_k_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape77 = R.call_tir(cls.reshape10, (lv98,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            lv148 = R.call_tir(cls.fused_NT_matmul10_add10, (layer_norm28, model_decoder_layers_2_self_attn_v_proj_weight2, model_decoder_layers_2_self_attn_v_proj_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape78 = R.call_tir(cls.reshape10, (lv148,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            squeeze12 = R.call_tir(cls.squeeze1, (reshape77,), out_sinfo=R.Tensor((seq_len, 6, 64), dtype="float32"))
            lv47: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_2_self_attn_k_cache3, squeeze12, sinfo_args=(R.Object,))
            squeeze13 = R.call_tir(cls.squeeze1, (reshape78,), out_sinfo=R.Tensor((seq_len, 6, 64), dtype="float32"))
            lv48: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_2_self_attn_v_cache3, squeeze13, sinfo_args=(R.Object,))
            lv49: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv47, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape79 = R.call_tir(cls.reshape6, (lv49,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            lv50: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv48, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape80 = R.call_tir(cls.reshape6, (lv50,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            permute_dims165 = R.call_tir(cls.transpose10, (reshape76,), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            permute_dims166 = R.call_tir(cls.transpose10, (reshape79,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            permute_dims167 = R.call_tir(cls.transpose10, (reshape80,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            lv149 = R.call_tir(cls.fused_NT_matmul11_maximum4_minimum4, (permute_dims165, permute_dims166), out_sinfo=R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32"))
            softmax16 = R.call_tir(cls.softmax3, (lv149,), out_sinfo=R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32"))
            matmul117 = R.call_tir(cls.matmul19, (softmax16, permute_dims167), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            permute_dims169 = R.call_tir(cls.transpose12, (matmul117,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            reshape81 = R.call_tir(cls.reshape11, (permute_dims169,), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv150 = R.call_tir(cls.fused_NT_matmul10_add10_add9, (reshape81, model_decoder_layers_2_self_attn_out_proj_weight2, model_decoder_layers_2_self_attn_out_proj_bias2, lv146), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            layer_norm29 = R.call_tir(cls.layer_norm2, (lv150, model_decoder_layers_2_encoder_attn_layer_norm_weight2, model_decoder_layers_2_encoder_attn_layer_norm_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv151 = R.call_tir(cls.fused_NT_matmul10_add10_multiply2, (layer_norm29, model_decoder_layers_2_encoder_attn_q_proj_weight2, model_decoder_layers_2_encoder_attn_q_proj_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape82 = R.call_tir(cls.reshape10, (lv151,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            permute_dims172 = R.call_tir(cls.transpose10, (reshape82,), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            lv152 = R.call_tir(cls.fused_NT_matmul12_maximum5_minimum5, (permute_dims172, lv122), out_sinfo=R.Tensor((1, 6, seq_len, 1500), dtype="float32"))
            softmax17 = R.call_tir(cls.softmax4, (lv152,), out_sinfo=R.Tensor((1, 6, seq_len, 1500), dtype="float32"))
            matmul121 = R.call_tir(cls.matmul20, (softmax17, lv124), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            permute_dims176 = R.call_tir(cls.transpose12, (matmul121,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            reshape83 = R.call_tir(cls.reshape11, (permute_dims176,), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv153 = R.call_tir(cls.fused_NT_matmul10_add10_add9, (reshape83, model_decoder_layers_2_encoder_attn_out_proj_weight2, model_decoder_layers_2_encoder_attn_out_proj_bias2, lv150), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            layer_norm30 = R.call_tir(cls.layer_norm2, (lv153, model_decoder_layers_2_final_layer_norm_weight2, model_decoder_layers_2_final_layer_norm_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv154 = R.call_tir(cls.fused_NT_matmul13_add11_gelu4, (layer_norm30, model_decoder_layers_2_fc1_weight2, model_decoder_layers_2_fc1_bias2), out_sinfo=R.Tensor((1, seq_len, 1536), dtype="float32"))
            lv155 = R.call_tir(cls.fused_NT_matmul14_add10_add9, (lv154, model_decoder_layers_2_fc2_weight2, model_decoder_layers_2_fc2_bias2, lv153), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            layer_norm31 = R.call_tir(cls.layer_norm2, (lv155, model_decoder_layers_3_self_attn_layer_norm_weight2, model_decoder_layers_3_self_attn_layer_norm_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv156 = R.call_tir(cls.fused_NT_matmul10_add10_multiply2, (layer_norm31, model_decoder_layers_3_self_attn_q_proj_weight2, model_decoder_layers_3_self_attn_q_proj_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape84 = R.call_tir(cls.reshape10, (lv156,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            lv108 = R.call_tir(cls.NT_matmul10, (layer_norm31, model_decoder_layers_3_self_attn_k_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape85 = R.call_tir(cls.reshape10, (lv108,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            lv157 = R.call_tir(cls.fused_NT_matmul10_add10, (layer_norm31, model_decoder_layers_3_self_attn_v_proj_weight2, model_decoder_layers_3_self_attn_v_proj_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape86 = R.call_tir(cls.reshape10, (lv157,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            squeeze14 = R.call_tir(cls.squeeze1, (reshape85,), out_sinfo=R.Tensor((seq_len, 6, 64), dtype="float32"))
            lv51: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_3_self_attn_k_cache3, squeeze14, sinfo_args=(R.Object,))
            squeeze15 = R.call_tir(cls.squeeze1, (reshape86,), out_sinfo=R.Tensor((seq_len, 6, 64), dtype="float32"))
            lv52: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_3_self_attn_v_cache3, squeeze15, sinfo_args=(R.Object,))
            lv53: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv51, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape87 = R.call_tir(cls.reshape6, (lv53,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            lv54: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv52, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape88 = R.call_tir(cls.reshape6, (lv54,), out_sinfo=R.Tensor((1, total_seq_len, 6, 64), dtype="float32"))
            permute_dims183 = R.call_tir(cls.transpose10, (reshape84,), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            permute_dims184 = R.call_tir(cls.transpose10, (reshape87,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            permute_dims185 = R.call_tir(cls.transpose10, (reshape88,), out_sinfo=R.Tensor((1, 6, total_seq_len, 64), dtype="float32"))
            lv158 = R.call_tir(cls.fused_NT_matmul11_maximum4_minimum4, (permute_dims183, permute_dims184), out_sinfo=R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32"))
            softmax18 = R.call_tir(cls.softmax3, (lv158,), out_sinfo=R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32"))
            matmul129 = R.call_tir(cls.matmul19, (softmax18, permute_dims185), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            permute_dims187 = R.call_tir(cls.transpose12, (matmul129,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            reshape89 = R.call_tir(cls.reshape11, (permute_dims187,), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv159 = R.call_tir(cls.fused_NT_matmul10_add10_add9, (reshape89, model_decoder_layers_3_self_attn_out_proj_weight2, model_decoder_layers_3_self_attn_out_proj_bias2, lv155), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            layer_norm32 = R.call_tir(cls.layer_norm2, (lv159, model_decoder_layers_3_encoder_attn_layer_norm_weight2, model_decoder_layers_3_encoder_attn_layer_norm_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv160 = R.call_tir(cls.fused_NT_matmul10_add10_multiply2, (layer_norm32, model_decoder_layers_3_encoder_attn_q_proj_weight2, model_decoder_layers_3_encoder_attn_q_proj_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape90 = R.call_tir(cls.reshape10, (lv160,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            permute_dims190 = R.call_tir(cls.transpose10, (reshape90,), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            lv161 = R.call_tir(cls.fused_NT_matmul12_maximum5_minimum5, (permute_dims190, lv126), out_sinfo=R.Tensor((1, 6, seq_len, 1500), dtype="float32"))
            softmax19 = R.call_tir(cls.softmax4, (lv161,), out_sinfo=R.Tensor((1, 6, seq_len, 1500), dtype="float32"))
            matmul133 = R.call_tir(cls.matmul20, (softmax19, lv128), out_sinfo=R.Tensor((1, 6, seq_len, 64), dtype="float32"))
            permute_dims194 = R.call_tir(cls.transpose12, (matmul133,), out_sinfo=R.Tensor((1, seq_len, 6, 64), dtype="float32"))
            reshape91 = R.call_tir(cls.reshape11, (permute_dims194,), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv162 = R.call_tir(cls.fused_NT_matmul10_add10_add9, (reshape91, model_decoder_layers_3_encoder_attn_out_proj_weight2, model_decoder_layers_3_encoder_attn_out_proj_bias2, lv159), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            layer_norm33 = R.call_tir(cls.layer_norm2, (lv162, model_decoder_layers_3_final_layer_norm_weight2, model_decoder_layers_3_final_layer_norm_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv163 = R.call_tir(cls.fused_NT_matmul13_add11_gelu4, (layer_norm33, model_decoder_layers_3_fc1_weight2, model_decoder_layers_3_fc1_bias2), out_sinfo=R.Tensor((1, seq_len, 1536), dtype="float32"))
            lv164 = R.call_tir(cls.fused_NT_matmul14_add10_add9, (lv163, model_decoder_layers_3_fc2_weight2, model_decoder_layers_3_fc2_bias2, lv162), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            layer_norm34 = R.call_tir(cls.layer_norm2, (lv164, model_decoder_layer_norm_weight2, model_decoder_layer_norm_bias2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            lv117_1 = R.call_tir(cls.NT_matmul15, (layer_norm34, proj_out_weight2), out_sinfo=R.Tensor((1, seq_len, 51865), dtype="float32"))
            gv3: R.Tuple(R.Tensor((1, seq_len, 51865), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)) = lv117_1, (lv39, lv40, model_decoder_layers_0_encoder_attn_k_cache3, model_decoder_layers_0_encoder_attn_v_cache3, lv43, lv44, model_decoder_layers_1_encoder_attn_k_cache3, model_decoder_layers_1_encoder_attn_v_cache3, lv47, lv48, model_decoder_layers_2_encoder_attn_k_cache3, model_decoder_layers_2_encoder_attn_v_cache3, lv51, lv52, model_decoder_layers_3_encoder_attn_k_cache3, model_decoder_layers_3_encoder_attn_v_cache3)
            R.output(gv3)
        return gv3

# Metadata omitted. Use show_meta=True in script() method to show it.