# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def NT_matmul(A: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), B: T.Buffer((T.int64(384), T.int64(384)), "float32"), NT_matmul: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
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
    def NT_matmul1(A: T.Buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(64)), "float32"), var_B: T.handle, var_NT_matmul: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        total_seq_len = T.int64()
        B = T.match_buffer(var_B, (T.int64(1), T.int64(6), total_seq_len, T.int64(64)))
        NT_matmul = T.match_buffer(var_NT_matmul, (T.int64(1), T.int64(6), T.int64(1), total_seq_len))
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(6), T.int64(1), total_seq_len, T.int64(64)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_i3, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2, v_i3] = NT_matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_i3, v_k]

    @T.prim_func(private=True)
    def NT_matmul10(var_A: T.handle, B: T.Buffer((T.int64(384), T.int64(384)), "float32"), var_NT_matmul: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
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
    def NT_matmul11(var_A: T.handle, var_B: T.handle, var_NT_matmul: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(6), seq_len, T.int64(64)))
        total_seq_len = T.int64()
        B = T.match_buffer(var_B, (T.int64(1), T.int64(6), total_seq_len, T.int64(64)))
        NT_matmul = T.match_buffer(var_NT_matmul, (T.int64(1), T.int64(6), seq_len, total_seq_len))
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(6), seq_len, total_seq_len, T.int64(64)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_i3, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2, v_i3] = NT_matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_i3, v_k]

    @T.prim_func(private=True)
    def NT_matmul12(var_A: T.handle, B: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32"), var_NT_matmul: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(6), seq_len, T.int64(64)))
        NT_matmul = T.match_buffer(var_NT_matmul, (T.int64(1), T.int64(6), seq_len, T.int64(1500)))
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(6), seq_len, T.int64(1500), T.int64(64)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_i3, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2, v_i3] = NT_matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_i3, v_k]

    @T.prim_func(private=True)
    def NT_matmul13(var_A: T.handle, B: T.Buffer((T.int64(1536), T.int64(384)), "float32"), var_NT_matmul: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), seq_len, T.int64(384)))
        NT_matmul = T.match_buffer(var_NT_matmul, (T.int64(1), seq_len, T.int64(1536)))
        # with T.block("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(1536), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]

    @T.prim_func(private=True)
    def NT_matmul14(var_A: T.handle, B: T.Buffer((T.int64(384), T.int64(1536)), "float32"), var_NT_matmul: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), seq_len, T.int64(1536)))
        NT_matmul = T.match_buffer(var_NT_matmul, (T.int64(1), seq_len, T.int64(384)))
        # with T.block("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(384), T.int64(1536)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]

    @T.prim_func(private=True)
    def NT_matmul15(var_A: T.handle, B: T.Buffer((T.int64(51865), T.int64(384)), "float32"), var_NT_matmul: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
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
        T.func_attr({"tir.noalias": T.bool(True)})
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
    def NT_matmul3(A: T.Buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(64)), "float32"), B: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32"), NT_matmul: T.Buffer((T.int64(1), T.int64(6), T.int64(1), T.int64(1500)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(6), T.int64(1), T.int64(1500), T.int64(64)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_i3, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2, v_i3] = NT_matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_i3, v_k]

    @T.prim_func(private=True)
    def NT_matmul4(A: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), B: T.Buffer((T.int64(1536), T.int64(384)), "float32"), NT_matmul: T.Buffer((T.int64(1), T.int64(1), T.int64(1536)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(1536), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]

    @T.prim_func(private=True)
    def NT_matmul5(A: T.Buffer((T.int64(1), T.int64(1), T.int64(1536)), "float32"), B: T.Buffer((T.int64(384), T.int64(1536)), "float32"), NT_matmul: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(384), T.int64(1536)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]

    @T.prim_func(private=True)
    def NT_matmul6(A: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), B: T.Buffer((T.int64(51865), T.int64(384)), "float32"), NT_matmul: T.Buffer((T.int64(1), T.int64(1), T.int64(51865)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
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
    def NT_matmul7(A: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32"), B: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(64)), "float32"), NT_matmul: T.Buffer((T.int64(1), T.int64(6), T.int64(1500), T.int64(1500)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(6), T.int64(1500), T.int64(1500), T.int64(64)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_i3, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2, v_i3] = NT_matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_i3, v_k]

    @T.prim_func(private=True)
    def NT_matmul8(A: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32"), B: T.Buffer((T.int64(1536), T.int64(384)), "float32"), NT_matmul: T.Buffer((T.int64(1), T.int64(1500), T.int64(1536)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1500), T.int64(1536), T.int64(384)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]

    @T.prim_func(private=True)
    def NT_matmul9(A: T.Buffer((T.int64(1), T.int64(1500), T.int64(1536)), "float32"), B: T.Buffer((T.int64(384), T.int64(1536)), "float32"), NT_matmul: T.Buffer((T.int64(1), T.int64(1500), T.int64(384)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1500), T.int64(384), T.int64(1536)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                T.writes(NT_matmul[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul[v_i0, v_i1, v_i2] = T.float32(0)
                NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]

    @T.prim_func(private=True)
    def position_embedding(A: T.Buffer((T.int64(1), T.int64(1)), "int32"), B: T.Buffer((T.int64(448), T.int64(384)), "float32"), position_embedding: T.Buffer((T.int64(1), T.int64(1), T.int64(384)), "float32"), total_seq_len: T.int64):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j, k in T.grid(T.int64(1), T.int64(1), T.int64(384)):
            with T.block("position_embedding"):
                v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
                T.reads(B[total_seq_len + v_j - T.int64(1), v_k])
                T.writes(position_embedding[v_i, v_j, v_k])
                position_embedding[v_i, v_j, v_k] = B[total_seq_len + v_j - T.int64(1), v_k]

    @T.prim_func(private=True)
    def position_embedding1(var_A: T.handle, B: T.Buffer((T.int64(448), T.int64(384)), "float32"), var_position_embedding: T.handle, total_seq_len: T.int64):
        T.func_attr({"tir.noalias": T.bool(True)})
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

    @R.function
    def _initialize_effect() -> R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object):
        R.func_attr({"tir_var_upper_bound": {"seq_len": 448, "total_seq_len": 448}})
        with R.dataflow():
            lv: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_0_self_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv1: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_0_self_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv1, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv2: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_0_encoder_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv2, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv3: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_0_encoder_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv3, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv4: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_1_self_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv4, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv5: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_1_self_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv5, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv6: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_1_encoder_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv6, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv7: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_1_encoder_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv7, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv8: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_2_self_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv8, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv9: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_2_self_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv9, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv10: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_2_encoder_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv10, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv11: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_2_encoder_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv11, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv12: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_3_self_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv12, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv13: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_3_self_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv13, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv14: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_3_encoder_attn_k_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv14, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
            lv15: R.Tensor((448, 6, 64), dtype="float32") = R.zeros(R.shape([448, 6, 64]), dtype="float32")
            model_decoder_layers_3_encoder_attn_v_cache: R.Object = R.call_packed("vm.builtin.attention_kv_cache_create", lv15, R.shape([448, 6, 64]), R.prim_value(0), sinfo_args=(R.Object,))
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
            reshape16: R.Tensor((1,), dtype="int32") = R.reshape(input_ids, R.shape([1]))
            take: R.Tensor((1, 384), dtype="float32") = R.take(model_decoder_embed_tokens_weight1, reshape16, axis=0)
            reshape17: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(take, R.shape([1, 1, 384]))
            lv21 = R.call_tir(cls.position_embedding, (input_ids, model_decoder_embed_positions_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"), tir_vars=R.shape([total_seq_len]))
            add29: R.Tensor((1, 1, 384), dtype="float32") = R.add(reshape17, lv21)
            layer_norm9: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add29, model_decoder_layers_0_self_attn_layer_norm_weight1, model_decoder_layers_0_self_attn_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv28 = R.call_tir(cls.NT_matmul, (layer_norm9, model_decoder_layers_0_self_attn_q_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add30: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv28, model_decoder_layers_0_self_attn_q_proj_bias1)
            mul4: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add30, R.const(0.125, "float32"))
            reshape18: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul4, R.shape([1, 1, 6, 64]))
            lv29 = R.call_tir(cls.NT_matmul, (layer_norm9, model_decoder_layers_0_self_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            reshape19: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(lv29, R.shape([1, 1, 6, 64]))
            lv30 = R.call_tir(cls.NT_matmul, (layer_norm9, model_decoder_layers_0_self_attn_v_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add31: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv30, model_decoder_layers_0_self_attn_v_proj_bias1)
            reshape20: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(add31, R.shape([1, 1, 6, 64]))
            squeeze: R.Tensor((1, 6, 64), dtype="float32") = R.squeeze(reshape19, axis=[0])
            lv22: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_0_self_attn_k_cache2, squeeze, sinfo_args=(R.Object,))
            squeeze1: R.Tensor((1, 6, 64), dtype="float32") = R.squeeze(reshape20, axis=[0])
            lv23: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_0_self_attn_v_cache2, squeeze1, sinfo_args=(R.Object,))
            lv24: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv22, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape21: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv24, R.shape([1, total_seq_len, 6, 64]))
            lv25: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv23, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape22: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv25, R.shape([1, total_seq_len, 6, 64]))
            permute_dims48: R.Tensor((1, 6, 1, 64), dtype="float32") = R.permute_dims(reshape18, axes=[0, 2, 1, 3])
            permute_dims49: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape21, axes=[0, 2, 1, 3])
            permute_dims50: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape22, axes=[0, 2, 1, 3])
            lv31 = R.call_tir(cls.NT_matmul1, (permute_dims48, permute_dims49), out_sinfo=R.Tensor((1, 6, 1, total_seq_len), dtype="float32"))
            maximum8: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.maximum(lv31, R.const(-3.4028234663852886e+38, "float32"))
            minimum8: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.minimum(maximum8, R.const(3.4028234663852886e+38, "float32"))
            softmax4: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.nn.softmax(minimum8, axis=-1)
            matmul36: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax4, permute_dims50, out_dtype="void")
            permute_dims52: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul36, axes=[0, 2, 1, 3])
            reshape23: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims52, R.shape([1, 1, 384]))
            lv32 = R.call_tir(cls.NT_matmul, (reshape23, model_decoder_layers_0_self_attn_out_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add32: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv32, model_decoder_layers_0_self_attn_out_proj_bias1)
            add33: R.Tensor((1, 1, 384), dtype="float32") = R.add(add29, add32)
            layer_norm10: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add33, model_decoder_layers_0_encoder_attn_layer_norm_weight1, model_decoder_layers_0_encoder_attn_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv33 = R.call_tir(cls.NT_matmul, (layer_norm10, model_decoder_layers_0_encoder_attn_q_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add34: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv33, model_decoder_layers_0_encoder_attn_q_proj_bias1)
            mul5: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add34, R.const(0.125, "float32"))
            reshape24: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul5, R.shape([1, 1, 6, 64]))
            lv34 = R.call_tir(cls.NT_matmul2, (encoder_hidden_states, model_decoder_layers_0_encoder_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape25: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(lv34, R.shape([1, 1500, 6, 64]))
            lv35 = R.call_tir(cls.NT_matmul2, (encoder_hidden_states, model_decoder_layers_0_encoder_attn_v_proj_weight1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add35: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv35, model_decoder_layers_0_encoder_attn_v_proj_bias1)
            reshape26: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add35, R.shape([1, 1500, 6, 64]))
            permute_dims57: R.Tensor((1, 6, 1, 64), dtype="float32") = R.permute_dims(reshape24, axes=[0, 2, 1, 3])
            permute_dims58: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape25, axes=[0, 2, 1, 3])
            permute_dims59: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape26, axes=[0, 2, 1, 3])
            lv36 = R.call_tir(cls.NT_matmul3, (permute_dims57, permute_dims58), out_sinfo=R.Tensor((1, 6, 1, 1500), dtype="float32"))
            maximum9: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.maximum(lv36, R.const(-3.4028234663852886e+38, "float32"))
            minimum9: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.minimum(maximum9, R.const(3.4028234663852886e+38, "float32"))
            softmax5: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.nn.softmax(minimum9, axis=-1)
            matmul42: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax5, permute_dims59, out_dtype="void")
            permute_dims61: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul42, axes=[0, 2, 1, 3])
            reshape27: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims61, R.shape([1, 1, 384]))
            lv37 = R.call_tir(cls.NT_matmul, (reshape27, model_decoder_layers_0_encoder_attn_out_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add36: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv37, model_decoder_layers_0_encoder_attn_out_proj_bias1)
            add37: R.Tensor((1, 1, 384), dtype="float32") = R.add(add33, add36)
            layer_norm11: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add37, model_decoder_layers_0_final_layer_norm_weight1, model_decoder_layers_0_final_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv38 = R.call_tir(cls.NT_matmul4, (layer_norm11, model_decoder_layers_0_fc1_weight1), out_sinfo=R.Tensor((1, 1, 1536), dtype="float32"))
            add38: R.Tensor((1, 1, 1536), dtype="float32") = R.add(lv38, model_decoder_layers_0_fc1_bias1)
            gelu6: R.Tensor((1, 1, 1536), dtype="float32") = R.nn.gelu(add38)
            lv39 = R.call_tir(cls.NT_matmul5, (gelu6, model_decoder_layers_0_fc2_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add39: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv39, model_decoder_layers_0_fc2_bias1)
            add40: R.Tensor((1, 1, 384), dtype="float32") = R.add(add37, add39)
            layer_norm12: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add40, model_decoder_layers_1_self_attn_layer_norm_weight1, model_decoder_layers_1_self_attn_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv40 = R.call_tir(cls.NT_matmul, (layer_norm12, model_decoder_layers_1_self_attn_q_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add41: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv40, model_decoder_layers_1_self_attn_q_proj_bias1)
            mul6: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add41, R.const(0.125, "float32"))
            reshape28: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul6, R.shape([1, 1, 6, 64]))
            lv41 = R.call_tir(cls.NT_matmul, (layer_norm12, model_decoder_layers_1_self_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            reshape29: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(lv41, R.shape([1, 1, 6, 64]))
            lv42 = R.call_tir(cls.NT_matmul, (layer_norm12, model_decoder_layers_1_self_attn_v_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add42: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv42, model_decoder_layers_1_self_attn_v_proj_bias1)
            reshape30: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(add42, R.shape([1, 1, 6, 64]))
            squeeze2: R.Tensor((1, 6, 64), dtype="float32") = R.squeeze(reshape29, axis=[0])
            lv26: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_1_self_attn_k_cache2, squeeze2, sinfo_args=(R.Object,))
            squeeze3: R.Tensor((1, 6, 64), dtype="float32") = R.squeeze(reshape30, axis=[0])
            lv27: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_1_self_attn_v_cache2, squeeze3, sinfo_args=(R.Object,))
            lv28_1: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv26, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape31: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv28_1, R.shape([1, total_seq_len, 6, 64]))
            lv29_1: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv27, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape32: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv29_1, R.shape([1, total_seq_len, 6, 64]))
            permute_dims68: R.Tensor((1, 6, 1, 64), dtype="float32") = R.permute_dims(reshape28, axes=[0, 2, 1, 3])
            permute_dims69: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape31, axes=[0, 2, 1, 3])
            permute_dims70: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape32, axes=[0, 2, 1, 3])
            lv43 = R.call_tir(cls.NT_matmul1, (permute_dims68, permute_dims69), out_sinfo=R.Tensor((1, 6, 1, total_seq_len), dtype="float32"))
            maximum10: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.maximum(lv43, R.const(-3.4028234663852886e+38, "float32"))
            minimum10: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.minimum(maximum10, R.const(3.4028234663852886e+38, "float32"))
            softmax6: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.nn.softmax(minimum10, axis=-1)
            matmul50: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax6, permute_dims70, out_dtype="void")
            permute_dims72: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul50, axes=[0, 2, 1, 3])
            reshape33: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims72, R.shape([1, 1, 384]))
            lv44 = R.call_tir(cls.NT_matmul, (reshape33, model_decoder_layers_1_self_attn_out_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add43: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv44, model_decoder_layers_1_self_attn_out_proj_bias1)
            add44: R.Tensor((1, 1, 384), dtype="float32") = R.add(add40, add43)
            layer_norm13: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add44, model_decoder_layers_1_encoder_attn_layer_norm_weight1, model_decoder_layers_1_encoder_attn_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv45 = R.call_tir(cls.NT_matmul, (layer_norm13, model_decoder_layers_1_encoder_attn_q_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add45: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv45, model_decoder_layers_1_encoder_attn_q_proj_bias1)
            mul7: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add45, R.const(0.125, "float32"))
            reshape34: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul7, R.shape([1, 1, 6, 64]))
            lv46 = R.call_tir(cls.NT_matmul2, (encoder_hidden_states, model_decoder_layers_1_encoder_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape35: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(lv46, R.shape([1, 1500, 6, 64]))
            lv47 = R.call_tir(cls.NT_matmul2, (encoder_hidden_states, model_decoder_layers_1_encoder_attn_v_proj_weight1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add46: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv47, model_decoder_layers_1_encoder_attn_v_proj_bias1)
            reshape36: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add46, R.shape([1, 1500, 6, 64]))
            permute_dims77: R.Tensor((1, 6, 1, 64), dtype="float32") = R.permute_dims(reshape34, axes=[0, 2, 1, 3])
            permute_dims78: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape35, axes=[0, 2, 1, 3])
            permute_dims79: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape36, axes=[0, 2, 1, 3])
            lv48 = R.call_tir(cls.NT_matmul3, (permute_dims77, permute_dims78), out_sinfo=R.Tensor((1, 6, 1, 1500), dtype="float32"))
            maximum11: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.maximum(lv48, R.const(-3.4028234663852886e+38, "float32"))
            minimum11: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.minimum(maximum11, R.const(3.4028234663852886e+38, "float32"))
            softmax7: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.nn.softmax(minimum11, axis=-1)
            matmul56: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax7, permute_dims79, out_dtype="void")
            permute_dims81: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul56, axes=[0, 2, 1, 3])
            reshape37: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims81, R.shape([1, 1, 384]))
            lv49 = R.call_tir(cls.NT_matmul, (reshape37, model_decoder_layers_1_encoder_attn_out_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add47: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv49, model_decoder_layers_1_encoder_attn_out_proj_bias1)
            add48: R.Tensor((1, 1, 384), dtype="float32") = R.add(add44, add47)
            layer_norm14: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add48, model_decoder_layers_1_final_layer_norm_weight1, model_decoder_layers_1_final_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv50 = R.call_tir(cls.NT_matmul4, (layer_norm14, model_decoder_layers_1_fc1_weight1), out_sinfo=R.Tensor((1, 1, 1536), dtype="float32"))
            add49: R.Tensor((1, 1, 1536), dtype="float32") = R.add(lv50, model_decoder_layers_1_fc1_bias1)
            gelu7: R.Tensor((1, 1, 1536), dtype="float32") = R.nn.gelu(add49)
            lv51 = R.call_tir(cls.NT_matmul5, (gelu7, model_decoder_layers_1_fc2_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add50: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv51, model_decoder_layers_1_fc2_bias1)
            add51: R.Tensor((1, 1, 384), dtype="float32") = R.add(add48, add50)
            layer_norm15: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add51, model_decoder_layers_2_self_attn_layer_norm_weight1, model_decoder_layers_2_self_attn_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv52 = R.call_tir(cls.NT_matmul, (layer_norm15, model_decoder_layers_2_self_attn_q_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add52: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv52, model_decoder_layers_2_self_attn_q_proj_bias1)
            mul8: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add52, R.const(0.125, "float32"))
            reshape38: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul8, R.shape([1, 1, 6, 64]))
            lv53 = R.call_tir(cls.NT_matmul, (layer_norm15, model_decoder_layers_2_self_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            reshape39: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(lv53, R.shape([1, 1, 6, 64]))
            lv54 = R.call_tir(cls.NT_matmul, (layer_norm15, model_decoder_layers_2_self_attn_v_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add53: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv54, model_decoder_layers_2_self_attn_v_proj_bias1)
            reshape40: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(add53, R.shape([1, 1, 6, 64]))
            squeeze4: R.Tensor((1, 6, 64), dtype="float32") = R.squeeze(reshape39, axis=[0])
            lv30_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_2_self_attn_k_cache2, squeeze4, sinfo_args=(R.Object,))
            squeeze5: R.Tensor((1, 6, 64), dtype="float32") = R.squeeze(reshape40, axis=[0])
            lv31_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_2_self_attn_v_cache2, squeeze5, sinfo_args=(R.Object,))
            lv32_1: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv30_1, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape41: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv32_1, R.shape([1, total_seq_len, 6, 64]))
            lv33_1: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv31_1, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape42: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv33_1, R.shape([1, total_seq_len, 6, 64]))
            permute_dims88: R.Tensor((1, 6, 1, 64), dtype="float32") = R.permute_dims(reshape38, axes=[0, 2, 1, 3])
            permute_dims89: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape41, axes=[0, 2, 1, 3])
            permute_dims90: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape42, axes=[0, 2, 1, 3])
            lv55 = R.call_tir(cls.NT_matmul1, (permute_dims88, permute_dims89), out_sinfo=R.Tensor((1, 6, 1, total_seq_len), dtype="float32"))
            maximum12: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.maximum(lv55, R.const(-3.4028234663852886e+38, "float32"))
            minimum12: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.minimum(maximum12, R.const(3.4028234663852886e+38, "float32"))
            softmax8: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.nn.softmax(minimum12, axis=-1)
            matmul64: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax8, permute_dims90, out_dtype="void")
            permute_dims92: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul64, axes=[0, 2, 1, 3])
            reshape43: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims92, R.shape([1, 1, 384]))
            lv56 = R.call_tir(cls.NT_matmul, (reshape43, model_decoder_layers_2_self_attn_out_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add54: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv56, model_decoder_layers_2_self_attn_out_proj_bias1)
            add55: R.Tensor((1, 1, 384), dtype="float32") = R.add(add51, add54)
            layer_norm16: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add55, model_decoder_layers_2_encoder_attn_layer_norm_weight1, model_decoder_layers_2_encoder_attn_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv57 = R.call_tir(cls.NT_matmul, (layer_norm16, model_decoder_layers_2_encoder_attn_q_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add56: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv57, model_decoder_layers_2_encoder_attn_q_proj_bias1)
            mul9: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add56, R.const(0.125, "float32"))
            reshape44: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul9, R.shape([1, 1, 6, 64]))
            lv58 = R.call_tir(cls.NT_matmul2, (encoder_hidden_states, model_decoder_layers_2_encoder_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape45: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(lv58, R.shape([1, 1500, 6, 64]))
            lv59 = R.call_tir(cls.NT_matmul2, (encoder_hidden_states, model_decoder_layers_2_encoder_attn_v_proj_weight1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add57: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv59, model_decoder_layers_2_encoder_attn_v_proj_bias1)
            reshape46: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add57, R.shape([1, 1500, 6, 64]))
            permute_dims97: R.Tensor((1, 6, 1, 64), dtype="float32") = R.permute_dims(reshape44, axes=[0, 2, 1, 3])
            permute_dims98: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape45, axes=[0, 2, 1, 3])
            permute_dims99: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape46, axes=[0, 2, 1, 3])
            lv60 = R.call_tir(cls.NT_matmul3, (permute_dims97, permute_dims98), out_sinfo=R.Tensor((1, 6, 1, 1500), dtype="float32"))
            maximum13: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.maximum(lv60, R.const(-3.4028234663852886e+38, "float32"))
            minimum13: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.minimum(maximum13, R.const(3.4028234663852886e+38, "float32"))
            softmax9: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.nn.softmax(minimum13, axis=-1)
            matmul70: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax9, permute_dims99, out_dtype="void")
            permute_dims101: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul70, axes=[0, 2, 1, 3])
            reshape47: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims101, R.shape([1, 1, 384]))
            lv61 = R.call_tir(cls.NT_matmul, (reshape47, model_decoder_layers_2_encoder_attn_out_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add58: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv61, model_decoder_layers_2_encoder_attn_out_proj_bias1)
            add59: R.Tensor((1, 1, 384), dtype="float32") = R.add(add55, add58)
            layer_norm17: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add59, model_decoder_layers_2_final_layer_norm_weight1, model_decoder_layers_2_final_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv62 = R.call_tir(cls.NT_matmul4, (layer_norm17, model_decoder_layers_2_fc1_weight1), out_sinfo=R.Tensor((1, 1, 1536), dtype="float32"))
            add60: R.Tensor((1, 1, 1536), dtype="float32") = R.add(lv62, model_decoder_layers_2_fc1_bias1)
            gelu8: R.Tensor((1, 1, 1536), dtype="float32") = R.nn.gelu(add60)
            lv63 = R.call_tir(cls.NT_matmul5, (gelu8, model_decoder_layers_2_fc2_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add61: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv63, model_decoder_layers_2_fc2_bias1)
            add62: R.Tensor((1, 1, 384), dtype="float32") = R.add(add59, add61)
            layer_norm18: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add62, model_decoder_layers_3_self_attn_layer_norm_weight1, model_decoder_layers_3_self_attn_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv64 = R.call_tir(cls.NT_matmul, (layer_norm18, model_decoder_layers_3_self_attn_q_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add63: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv64, model_decoder_layers_3_self_attn_q_proj_bias1)
            mul10: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add63, R.const(0.125, "float32"))
            reshape48: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul10, R.shape([1, 1, 6, 64]))
            lv65 = R.call_tir(cls.NT_matmul, (layer_norm18, model_decoder_layers_3_self_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            reshape49: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(lv65, R.shape([1, 1, 6, 64]))
            lv66 = R.call_tir(cls.NT_matmul, (layer_norm18, model_decoder_layers_3_self_attn_v_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add64: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv66, model_decoder_layers_3_self_attn_v_proj_bias1)
            reshape50: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(add64, R.shape([1, 1, 6, 64]))
            squeeze6: R.Tensor((1, 6, 64), dtype="float32") = R.squeeze(reshape49, axis=[0])
            lv34_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_3_self_attn_k_cache2, squeeze6, sinfo_args=(R.Object,))
            squeeze7: R.Tensor((1, 6, 64), dtype="float32") = R.squeeze(reshape50, axis=[0])
            lv35_1: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_3_self_attn_v_cache2, squeeze7, sinfo_args=(R.Object,))
            lv36_1: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv34_1, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape51: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv36_1, R.shape([1, total_seq_len, 6, 64]))
            lv37_1: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv35_1, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape52: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv37_1, R.shape([1, total_seq_len, 6, 64]))
            permute_dims108: R.Tensor((1, 6, 1, 64), dtype="float32") = R.permute_dims(reshape48, axes=[0, 2, 1, 3])
            permute_dims109: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape51, axes=[0, 2, 1, 3])
            permute_dims110: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape52, axes=[0, 2, 1, 3])
            lv67 = R.call_tir(cls.NT_matmul1, (permute_dims108, permute_dims109), out_sinfo=R.Tensor((1, 6, 1, total_seq_len), dtype="float32"))
            maximum14: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.maximum(lv67, R.const(-3.4028234663852886e+38, "float32"))
            minimum14: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.minimum(maximum14, R.const(3.4028234663852886e+38, "float32"))
            softmax10: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.nn.softmax(minimum14, axis=-1)
            matmul78: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax10, permute_dims110, out_dtype="void")
            permute_dims112: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul78, axes=[0, 2, 1, 3])
            reshape53: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims112, R.shape([1, 1, 384]))
            lv68 = R.call_tir(cls.NT_matmul, (reshape53, model_decoder_layers_3_self_attn_out_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add65: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv68, model_decoder_layers_3_self_attn_out_proj_bias1)
            add66: R.Tensor((1, 1, 384), dtype="float32") = R.add(add62, add65)
            layer_norm19: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add66, model_decoder_layers_3_encoder_attn_layer_norm_weight1, model_decoder_layers_3_encoder_attn_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv69 = R.call_tir(cls.NT_matmul, (layer_norm19, model_decoder_layers_3_encoder_attn_q_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add67: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv69, model_decoder_layers_3_encoder_attn_q_proj_bias1)
            mul11: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add67, R.const(0.125, "float32"))
            reshape54: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul11, R.shape([1, 1, 6, 64]))
            lv70 = R.call_tir(cls.NT_matmul2, (encoder_hidden_states, model_decoder_layers_3_encoder_attn_k_proj_weight1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape55: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(lv70, R.shape([1, 1500, 6, 64]))
            lv71 = R.call_tir(cls.NT_matmul2, (encoder_hidden_states, model_decoder_layers_3_encoder_attn_v_proj_weight1), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add68: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv71, model_decoder_layers_3_encoder_attn_v_proj_bias1)
            reshape56: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add68, R.shape([1, 1500, 6, 64]))
            permute_dims117: R.Tensor((1, 6, 1, 64), dtype="float32") = R.permute_dims(reshape54, axes=[0, 2, 1, 3])
            permute_dims118: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape55, axes=[0, 2, 1, 3])
            permute_dims119: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape56, axes=[0, 2, 1, 3])
            lv72 = R.call_tir(cls.NT_matmul3, (permute_dims117, permute_dims118), out_sinfo=R.Tensor((1, 6, 1, 1500), dtype="float32"))
            maximum15: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.maximum(lv72, R.const(-3.4028234663852886e+38, "float32"))
            minimum15: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.minimum(maximum15, R.const(3.4028234663852886e+38, "float32"))
            softmax11: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.nn.softmax(minimum15, axis=-1)
            matmul84: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax11, permute_dims119, out_dtype="void")
            permute_dims121: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul84, axes=[0, 2, 1, 3])
            reshape57: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims121, R.shape([1, 1, 384]))
            lv73 = R.call_tir(cls.NT_matmul, (reshape57, model_decoder_layers_3_encoder_attn_out_proj_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add69: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv73, model_decoder_layers_3_encoder_attn_out_proj_bias1)
            add70: R.Tensor((1, 1, 384), dtype="float32") = R.add(add66, add69)
            layer_norm20: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add70, model_decoder_layers_3_final_layer_norm_weight1, model_decoder_layers_3_final_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv74 = R.call_tir(cls.NT_matmul4, (layer_norm20, model_decoder_layers_3_fc1_weight1), out_sinfo=R.Tensor((1, 1, 1536), dtype="float32"))
            add71: R.Tensor((1, 1, 1536), dtype="float32") = R.add(lv74, model_decoder_layers_3_fc1_bias1)
            gelu9: R.Tensor((1, 1, 1536), dtype="float32") = R.nn.gelu(add71)
            lv75 = R.call_tir(cls.NT_matmul5, (gelu9, model_decoder_layers_3_fc2_weight1), out_sinfo=R.Tensor((1, 1, 384), dtype="float32"))
            add72: R.Tensor((1, 1, 384), dtype="float32") = R.add(lv75, model_decoder_layers_3_fc2_bias1)
            add73: R.Tensor((1, 1, 384), dtype="float32") = R.add(add70, add72)
            layer_norm21: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add73, model_decoder_layer_norm_weight1, model_decoder_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv76 = R.call_tir(cls.NT_matmul6, (layer_norm21, proj_out_weight1), out_sinfo=R.Tensor((1, 1, 51865), dtype="float32"))
            gv2: R.Tuple(R.Tuple(R.Tensor((1, 1, 51865), dtype="float32"), R.Tuple(R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")), R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")), R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")), R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")))), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)) = (lv76, ((reshape25, reshape26), (reshape35, reshape36), (reshape45, reshape46), (reshape55, reshape56))), (lv22, lv23, model_decoder_layers_0_encoder_attn_k_cache2, model_decoder_layers_0_encoder_attn_v_cache2, lv26, lv27, model_decoder_layers_1_encoder_attn_k_cache2, model_decoder_layers_1_encoder_attn_v_cache2, lv30_1, lv31_1, model_decoder_layers_2_encoder_attn_k_cache2, model_decoder_layers_2_encoder_attn_v_cache2, lv34_1, lv35_1, model_decoder_layers_3_encoder_attn_k_cache2, model_decoder_layers_3_encoder_attn_v_cache2)
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
            model_encoder_conv1_bias: R.Tensor((384,), dtype="float32") = packed_params[1]
            model_encoder_conv2_weight: R.Tensor((384, 384, 3), dtype="float32") = packed_params[2]
            model_encoder_conv2_bias: R.Tensor((384,), dtype="float32") = packed_params[3]
            model_encoder_embed_positions_weight: R.Tensor((1500, 384), dtype="float32") = packed_params[4]
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
            lv17: R.Tensor((1, 384, 3000), dtype="float32") = R.nn.conv1d(input_ids, model_encoder_conv1_weight, strides=[1], padding=[1, 1], dilation=[1], groups=1, data_layout="NCW", kernel_layout="OIW", out_layout="NCW", out_dtype="void")
            lv18: R.Tensor((1, 384, 1), dtype="float32") = R.reshape(model_encoder_conv1_bias, R.shape([1, 384, 1]))
            conv1d: R.Tensor((1, 384, 3000), dtype="float32") = R.add(lv17, lv18)
            gelu: R.Tensor((1, 384, 3000), dtype="float32") = R.nn.gelu(conv1d)
            lv19: R.Tensor((1, 384, 1500), dtype="float32") = R.nn.conv1d(gelu, model_encoder_conv2_weight, strides=[2], padding=[1, 1], dilation=[1], groups=1, data_layout="NCW", kernel_layout="OIW", out_layout="NCW", out_dtype="void")
            lv20: R.Tensor((1, 384, 1), dtype="float32") = R.reshape(model_encoder_conv2_bias, R.shape([1, 384, 1]))
            conv1d1: R.Tensor((1, 384, 1500), dtype="float32") = R.add(lv19, lv20)
            gelu1: R.Tensor((1, 384, 1500), dtype="float32") = R.nn.gelu(conv1d1)
            permute_dims: R.Tensor((1, 1500, 384), dtype="float32") = R.permute_dims(gelu1, axes=[0, 2, 1])
            add: R.Tensor((1, 1500, 384), dtype="float32") = R.add(permute_dims, model_encoder_embed_positions_weight)
            layer_norm: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(add, model_encoder_layers_0_self_attn_layer_norm_weight, model_encoder_layers_0_self_attn_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv = R.call_tir(cls.NT_matmul2, (layer_norm, model_encoder_layers_0_self_attn_q_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add1: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv, model_encoder_layers_0_self_attn_q_proj_bias)
            mul: R.Tensor((1, 1500, 384), dtype="float32") = R.multiply(add1, R.const(0.125, "float32"))
            reshape: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(mul, R.shape([1, 1500, 6, 64]))
            lv1 = R.call_tir(cls.NT_matmul2, (layer_norm, model_encoder_layers_0_self_attn_k_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape1: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(lv1, R.shape([1, 1500, 6, 64]))
            lv2 = R.call_tir(cls.NT_matmul2, (layer_norm, model_encoder_layers_0_self_attn_v_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add2: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv2, model_encoder_layers_0_self_attn_v_proj_bias)
            reshape2: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add2, R.shape([1, 1500, 6, 64]))
            permute_dims4: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape, axes=[0, 2, 1, 3])
            permute_dims5: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape1, axes=[0, 2, 1, 3])
            permute_dims6: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape2, axes=[0, 2, 1, 3])
            lv3 = R.call_tir(cls.NT_matmul7, (permute_dims4, permute_dims5), out_sinfo=R.Tensor((1, 6, 1500, 1500), dtype="float32"))
            maximum: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.maximum(lv3, R.const(-3.4028234663852886e+38, "float32"))
            minimum: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.minimum(maximum, R.const(3.4028234663852886e+38, "float32"))
            softmax: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.nn.softmax(minimum, axis=-1)
            matmul4: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.matmul(softmax, permute_dims6, out_dtype="void")
            permute_dims8: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.permute_dims(matmul4, axes=[0, 2, 1, 3])
            reshape3: R.Tensor((1, 1500, 384), dtype="float32") = R.reshape(permute_dims8, R.shape([1, 1500, 384]))
            lv4 = R.call_tir(cls.NT_matmul2, (reshape3, model_encoder_layers_0_self_attn_out_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add3: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv4, model_encoder_layers_0_self_attn_out_proj_bias)
            add4: R.Tensor((1, 1500, 384), dtype="float32") = R.add(add, add3)
            layer_norm1: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(add4, model_encoder_layers_0_final_layer_norm_weight, model_encoder_layers_0_final_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv5 = R.call_tir(cls.NT_matmul8, (layer_norm1, model_encoder_layers_0_fc1_weight), out_sinfo=R.Tensor((1, 1500, 1536), dtype="float32"))
            add5: R.Tensor((1, 1500, 1536), dtype="float32") = R.add(lv5, model_encoder_layers_0_fc1_bias)
            gelu2: R.Tensor((1, 1500, 1536), dtype="float32") = R.nn.gelu(add5)
            lv6 = R.call_tir(cls.NT_matmul9, (gelu2, model_encoder_layers_0_fc2_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add6: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv6, model_encoder_layers_0_fc2_bias)
            add7: R.Tensor((1, 1500, 384), dtype="float32") = R.add(add4, add6)
            maximum1: R.Tensor((1, 1500, 384), dtype="float32") = R.maximum(add7, R.const(-3.4028234663852886e+38, "float32"))
            minimum1: R.Tensor((1, 1500, 384), dtype="float32") = R.minimum(maximum1, add7)
            layer_norm2: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(minimum1, model_encoder_layers_1_self_attn_layer_norm_weight, model_encoder_layers_1_self_attn_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv7 = R.call_tir(cls.NT_matmul2, (layer_norm2, model_encoder_layers_1_self_attn_q_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add8: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv7, model_encoder_layers_1_self_attn_q_proj_bias)
            mul1: R.Tensor((1, 1500, 384), dtype="float32") = R.multiply(add8, R.const(0.125, "float32"))
            reshape4: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(mul1, R.shape([1, 1500, 6, 64]))
            lv8 = R.call_tir(cls.NT_matmul2, (layer_norm2, model_encoder_layers_1_self_attn_k_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape5: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(lv8, R.shape([1, 1500, 6, 64]))
            lv9 = R.call_tir(cls.NT_matmul2, (layer_norm2, model_encoder_layers_1_self_attn_v_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add9: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv9, model_encoder_layers_1_self_attn_v_proj_bias)
            reshape6: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add9, R.shape([1, 1500, 6, 64]))
            permute_dims15: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape4, axes=[0, 2, 1, 3])
            permute_dims16: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape5, axes=[0, 2, 1, 3])
            permute_dims17: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape6, axes=[0, 2, 1, 3])
            lv10 = R.call_tir(cls.NT_matmul7, (permute_dims15, permute_dims16), out_sinfo=R.Tensor((1, 6, 1500, 1500), dtype="float32"))
            maximum2: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.maximum(lv10, R.const(-3.4028234663852886e+38, "float32"))
            minimum2: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.minimum(maximum2, R.const(3.4028234663852886e+38, "float32"))
            softmax1: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.nn.softmax(minimum2, axis=-1)
            matmul12: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.matmul(softmax1, permute_dims17, out_dtype="void")
            permute_dims19: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.permute_dims(matmul12, axes=[0, 2, 1, 3])
            reshape7: R.Tensor((1, 1500, 384), dtype="float32") = R.reshape(permute_dims19, R.shape([1, 1500, 384]))
            lv11 = R.call_tir(cls.NT_matmul2, (reshape7, model_encoder_layers_1_self_attn_out_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add10: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv11, model_encoder_layers_1_self_attn_out_proj_bias)
            add11: R.Tensor((1, 1500, 384), dtype="float32") = R.add(minimum1, add10)
            layer_norm3: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(add11, model_encoder_layers_1_final_layer_norm_weight, model_encoder_layers_1_final_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv12 = R.call_tir(cls.NT_matmul8, (layer_norm3, model_encoder_layers_1_fc1_weight), out_sinfo=R.Tensor((1, 1500, 1536), dtype="float32"))
            add12: R.Tensor((1, 1500, 1536), dtype="float32") = R.add(lv12, model_encoder_layers_1_fc1_bias)
            gelu3: R.Tensor((1, 1500, 1536), dtype="float32") = R.nn.gelu(add12)
            lv13 = R.call_tir(cls.NT_matmul9, (gelu3, model_encoder_layers_1_fc2_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add13: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv13, model_encoder_layers_1_fc2_bias)
            add14: R.Tensor((1, 1500, 384), dtype="float32") = R.add(add11, add13)
            maximum3: R.Tensor((1, 1500, 384), dtype="float32") = R.maximum(add14, R.const(-3.4028234663852886e+38, "float32"))
            minimum3: R.Tensor((1, 1500, 384), dtype="float32") = R.minimum(maximum3, add14)
            layer_norm4: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(minimum3, model_encoder_layers_2_self_attn_layer_norm_weight, model_encoder_layers_2_self_attn_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv14 = R.call_tir(cls.NT_matmul2, (layer_norm4, model_encoder_layers_2_self_attn_q_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add15: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv14, model_encoder_layers_2_self_attn_q_proj_bias)
            mul2: R.Tensor((1, 1500, 384), dtype="float32") = R.multiply(add15, R.const(0.125, "float32"))
            reshape8: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(mul2, R.shape([1, 1500, 6, 64]))
            lv15 = R.call_tir(cls.NT_matmul2, (layer_norm4, model_encoder_layers_2_self_attn_k_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape9: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(lv15, R.shape([1, 1500, 6, 64]))
            lv16 = R.call_tir(cls.NT_matmul2, (layer_norm4, model_encoder_layers_2_self_attn_v_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add16: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv16, model_encoder_layers_2_self_attn_v_proj_bias)
            reshape10: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add16, R.shape([1, 1500, 6, 64]))
            permute_dims26: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape8, axes=[0, 2, 1, 3])
            permute_dims27: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape9, axes=[0, 2, 1, 3])
            permute_dims28: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape10, axes=[0, 2, 1, 3])
            lv17_1 = R.call_tir(cls.NT_matmul7, (permute_dims26, permute_dims27), out_sinfo=R.Tensor((1, 6, 1500, 1500), dtype="float32"))
            maximum4: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.maximum(lv17_1, R.const(-3.4028234663852886e+38, "float32"))
            minimum4: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.minimum(maximum4, R.const(3.4028234663852886e+38, "float32"))
            softmax2: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.nn.softmax(minimum4, axis=-1)
            matmul20: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.matmul(softmax2, permute_dims28, out_dtype="void")
            permute_dims30: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.permute_dims(matmul20, axes=[0, 2, 1, 3])
            reshape11: R.Tensor((1, 1500, 384), dtype="float32") = R.reshape(permute_dims30, R.shape([1, 1500, 384]))
            lv18_1 = R.call_tir(cls.NT_matmul2, (reshape11, model_encoder_layers_2_self_attn_out_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add17: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv18_1, model_encoder_layers_2_self_attn_out_proj_bias)
            add18: R.Tensor((1, 1500, 384), dtype="float32") = R.add(minimum3, add17)
            layer_norm5: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(add18, model_encoder_layers_2_final_layer_norm_weight, model_encoder_layers_2_final_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv19_1 = R.call_tir(cls.NT_matmul8, (layer_norm5, model_encoder_layers_2_fc1_weight), out_sinfo=R.Tensor((1, 1500, 1536), dtype="float32"))
            add19: R.Tensor((1, 1500, 1536), dtype="float32") = R.add(lv19_1, model_encoder_layers_2_fc1_bias)
            gelu4: R.Tensor((1, 1500, 1536), dtype="float32") = R.nn.gelu(add19)
            lv20_1 = R.call_tir(cls.NT_matmul9, (gelu4, model_encoder_layers_2_fc2_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add20: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv20_1, model_encoder_layers_2_fc2_bias)
            add21: R.Tensor((1, 1500, 384), dtype="float32") = R.add(add18, add20)
            maximum5: R.Tensor((1, 1500, 384), dtype="float32") = R.maximum(add21, R.const(-3.4028234663852886e+38, "float32"))
            minimum5: R.Tensor((1, 1500, 384), dtype="float32") = R.minimum(maximum5, add21)
            layer_norm6: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(minimum5, model_encoder_layers_3_self_attn_layer_norm_weight, model_encoder_layers_3_self_attn_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv21 = R.call_tir(cls.NT_matmul2, (layer_norm6, model_encoder_layers_3_self_attn_q_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add22: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv21, model_encoder_layers_3_self_attn_q_proj_bias)
            mul3: R.Tensor((1, 1500, 384), dtype="float32") = R.multiply(add22, R.const(0.125, "float32"))
            reshape12: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(mul3, R.shape([1, 1500, 6, 64]))
            lv22 = R.call_tir(cls.NT_matmul2, (layer_norm6, model_encoder_layers_3_self_attn_k_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            reshape13: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(lv22, R.shape([1, 1500, 6, 64]))
            lv23 = R.call_tir(cls.NT_matmul2, (layer_norm6, model_encoder_layers_3_self_attn_v_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add23: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv23, model_encoder_layers_3_self_attn_v_proj_bias)
            reshape14: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add23, R.shape([1, 1500, 6, 64]))
            permute_dims37: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape12, axes=[0, 2, 1, 3])
            permute_dims38: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape13, axes=[0, 2, 1, 3])
            permute_dims39: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape14, axes=[0, 2, 1, 3])
            lv24 = R.call_tir(cls.NT_matmul7, (permute_dims37, permute_dims38), out_sinfo=R.Tensor((1, 6, 1500, 1500), dtype="float32"))
            maximum6: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.maximum(lv24, R.const(-3.4028234663852886e+38, "float32"))
            minimum6: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.minimum(maximum6, R.const(3.4028234663852886e+38, "float32"))
            softmax3: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.nn.softmax(minimum6, axis=-1)
            matmul28: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.matmul(softmax3, permute_dims39, out_dtype="void")
            permute_dims41: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.permute_dims(matmul28, axes=[0, 2, 1, 3])
            reshape15: R.Tensor((1, 1500, 384), dtype="float32") = R.reshape(permute_dims41, R.shape([1, 1500, 384]))
            lv25 = R.call_tir(cls.NT_matmul2, (reshape15, model_encoder_layers_3_self_attn_out_proj_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add24: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv25, model_encoder_layers_3_self_attn_out_proj_bias)
            add25: R.Tensor((1, 1500, 384), dtype="float32") = R.add(minimum5, add24)
            layer_norm7: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(add25, model_encoder_layers_3_final_layer_norm_weight, model_encoder_layers_3_final_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv26 = R.call_tir(cls.NT_matmul8, (layer_norm7, model_encoder_layers_3_fc1_weight), out_sinfo=R.Tensor((1, 1500, 1536), dtype="float32"))
            add26: R.Tensor((1, 1500, 1536), dtype="float32") = R.add(lv26, model_encoder_layers_3_fc1_bias)
            gelu5: R.Tensor((1, 1500, 1536), dtype="float32") = R.nn.gelu(add26)
            lv27 = R.call_tir(cls.NT_matmul9, (gelu5, model_encoder_layers_3_fc2_weight), out_sinfo=R.Tensor((1, 1500, 384), dtype="float32"))
            add27: R.Tensor((1, 1500, 384), dtype="float32") = R.add(lv27, model_encoder_layers_3_fc2_bias)
            add28: R.Tensor((1, 1500, 384), dtype="float32") = R.add(add25, add27)
            maximum7: R.Tensor((1, 1500, 384), dtype="float32") = R.maximum(add28, R.const(-3.4028234663852886e+38, "float32"))
            minimum7: R.Tensor((1, 1500, 384), dtype="float32") = R.minimum(maximum7, add28)
            layer_norm8: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(minimum7, model_encoder_layer_norm_weight, model_encoder_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            gv1: R.Tuple(R.Tensor((1, 1500, 384), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)) = layer_norm8, (model_decoder_layers_0_self_attn_k_cache1, model_decoder_layers_0_self_attn_v_cache1, model_decoder_layers_0_encoder_attn_k_cache1, model_decoder_layers_0_encoder_attn_v_cache1, model_decoder_layers_1_self_attn_k_cache1, model_decoder_layers_1_self_attn_v_cache1, model_decoder_layers_1_encoder_attn_k_cache1, model_decoder_layers_1_encoder_attn_v_cache1, model_decoder_layers_2_self_attn_k_cache1, model_decoder_layers_2_self_attn_v_cache1, model_decoder_layers_2_encoder_attn_k_cache1, model_decoder_layers_2_encoder_attn_v_cache1, model_decoder_layers_3_self_attn_k_cache1, model_decoder_layers_3_self_attn_v_cache1, model_decoder_layers_3_encoder_attn_k_cache1, model_decoder_layers_3_encoder_attn_v_cache1)
            R.output(gv1)
        return gv1

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul(model_encoder_layers_0_self_attn_q_proj_weight: R.Tensor((384, 384), dtype="float32"), layer_norm: R.Tensor((1, 1500, 384), dtype="float32")) -> R.Tensor((1, 1500, 384), dtype="float32"):
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims1: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_0_self_attn_q_proj_weight, axes=None)
            gv: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(layer_norm, permute_dims1, out_dtype="void")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul1(permute_dims5: R.Tensor((1, 6, 1500, 64), dtype="float32"), permute_dims4: R.Tensor((1, 6, 1500, 64), dtype="float32")) -> R.Tensor((1, 6, 1500, 1500), dtype="float32"):
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims7: R.Tensor((1, 6, 64, 1500), dtype="float32") = R.permute_dims(permute_dims5, axes=[0, 1, 3, 2])
            gv: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.matmul(permute_dims4, permute_dims7, out_dtype="void")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul10(model_decoder_layers_0_self_attn_q_proj_weight2: R.Tensor((384, 384), dtype="float32"), layer_norm22: R.Tensor((1, "seq_len", 384), dtype="float32")) -> R.Tensor((1, "seq_len", 384), dtype="float32"):
        seq_len = T.int64()
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims126: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_self_attn_q_proj_weight2, axes=None)
            gv: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm22, permute_dims126, out_dtype="void")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul11(permute_dims130: R.Tensor((1, 6, "total_seq_len", 64), dtype="float32"), permute_dims129: R.Tensor((1, 6, "seq_len", 64), dtype="float32")) -> R.Tensor((1, 6, "seq_len", "total_seq_len"), dtype="float32"):
        seq_len = T.int64()
        total_seq_len = T.int64()
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims132: R.Tensor((1, 6, 64, total_seq_len), dtype="float32") = R.permute_dims(permute_dims130, axes=[0, 1, 3, 2])
            gv: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.matmul(permute_dims129, permute_dims132, out_dtype="void")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul12(permute_dims137: R.Tensor((1, 6, 1500, 64), dtype="float32"), permute_dims136: R.Tensor((1, 6, "seq_len", 64), dtype="float32")) -> R.Tensor((1, 6, "seq_len", 1500), dtype="float32"):
        seq_len = T.int64()
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims139: R.Tensor((1, 6, 64, 1500), dtype="float32") = R.permute_dims(permute_dims137, axes=[0, 1, 3, 2])
            gv: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.matmul(permute_dims136, permute_dims139, out_dtype="void")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul13(model_decoder_layers_0_fc1_weight2: R.Tensor((1536, 384), dtype="float32"), layer_norm24: R.Tensor((1, "seq_len", 384), dtype="float32")) -> R.Tensor((1, "seq_len", 1536), dtype="float32"):
        seq_len = T.int64()
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims142: R.Tensor((384, 1536), dtype="float32") = R.permute_dims(model_decoder_layers_0_fc1_weight2, axes=None)
            gv: R.Tensor((1, seq_len, 1536), dtype="float32") = R.matmul(layer_norm24, permute_dims142, out_dtype="void")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul14(model_decoder_layers_0_fc2_weight2: R.Tensor((384, 1536), dtype="float32"), gelu10: R.Tensor((1, "seq_len", 1536), dtype="float32")) -> R.Tensor((1, "seq_len", 384), dtype="float32"):
        seq_len = T.int64()
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims143: R.Tensor((1536, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_fc2_weight2, axes=None)
            gv: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(gelu10, permute_dims143, out_dtype="void")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul15(proj_out_weight2: R.Tensor((51865, 384), dtype="float32"), layer_norm34: R.Tensor((1, "seq_len", 384), dtype="float32")) -> R.Tensor((1, "seq_len", 51865), dtype="float32"):
        seq_len = T.int64()
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims198: R.Tensor((384, 51865), dtype="float32") = R.permute_dims(proj_out_weight2, axes=None)
            gv: R.Tensor((1, seq_len, 51865), dtype="float32") = R.matmul(layer_norm34, permute_dims198, out_dtype="void")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul2(model_encoder_layers_0_fc1_weight: R.Tensor((1536, 384), dtype="float32"), layer_norm1: R.Tensor((1, 1500, 384), dtype="float32")) -> R.Tensor((1, 1500, 1536), dtype="float32"):
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims10: R.Tensor((384, 1536), dtype="float32") = R.permute_dims(model_encoder_layers_0_fc1_weight, axes=None)
            gv: R.Tensor((1, 1500, 1536), dtype="float32") = R.matmul(layer_norm1, permute_dims10, out_dtype="void")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul3(model_encoder_layers_0_fc2_weight: R.Tensor((384, 1536), dtype="float32"), gelu2: R.Tensor((1, 1500, 1536), dtype="float32")) -> R.Tensor((1, 1500, 384), dtype="float32"):
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims11: R.Tensor((1536, 384), dtype="float32") = R.permute_dims(model_encoder_layers_0_fc2_weight, axes=None)
            gv: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(gelu2, permute_dims11, out_dtype="void")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul4(model_decoder_layers_0_self_attn_q_proj_weight1: R.Tensor((384, 384), dtype="float32"), layer_norm9: R.Tensor((1, 1, 384), dtype="float32")) -> R.Tensor((1, 1, 384), dtype="float32"):
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims45: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_self_attn_q_proj_weight1, axes=None)
            gv: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm9, permute_dims45, out_dtype="void")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul5(permute_dims49: R.Tensor((1, 6, "total_seq_len", 64), dtype="float32"), permute_dims48: R.Tensor((1, 6, 1, 64), dtype="float32")) -> R.Tensor((1, 6, 1, "total_seq_len"), dtype="float32"):
        total_seq_len = T.int64()
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims51: R.Tensor((1, 6, 64, total_seq_len), dtype="float32") = R.permute_dims(permute_dims49, axes=[0, 1, 3, 2])
            gv: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.matmul(permute_dims48, permute_dims51, out_dtype="void")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul6(permute_dims58: R.Tensor((1, 6, 1500, 64), dtype="float32"), permute_dims57: R.Tensor((1, 6, 1, 64), dtype="float32")) -> R.Tensor((1, 6, 1, 1500), dtype="float32"):
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims60: R.Tensor((1, 6, 64, 1500), dtype="float32") = R.permute_dims(permute_dims58, axes=[0, 1, 3, 2])
            gv: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.matmul(permute_dims57, permute_dims60, out_dtype="void")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul7(model_decoder_layers_0_fc1_weight1: R.Tensor((1536, 384), dtype="float32"), layer_norm11: R.Tensor((1, 1, 384), dtype="float32")) -> R.Tensor((1, 1, 1536), dtype="float32"):
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims63: R.Tensor((384, 1536), dtype="float32") = R.permute_dims(model_decoder_layers_0_fc1_weight1, axes=None)
            gv: R.Tensor((1, 1, 1536), dtype="float32") = R.matmul(layer_norm11, permute_dims63, out_dtype="void")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul8(model_decoder_layers_0_fc2_weight1: R.Tensor((384, 1536), dtype="float32"), gelu6: R.Tensor((1, 1, 1536), dtype="float32")) -> R.Tensor((1, 1, 384), dtype="float32"):
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims64: R.Tensor((1536, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_fc2_weight1, axes=None)
            gv: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(gelu6, permute_dims64, out_dtype="void")
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_permute_dims_relax_matmul9(proj_out_weight1: R.Tensor((51865, 384), dtype="float32"), layer_norm21: R.Tensor((1, 1, 384), dtype="float32")) -> R.Tensor((1, 1, 51865), dtype="float32"):
        R.func_attr({"Composite": "transpose_matmul_fuse", "Primitive": 1})
        with R.dataflow():
            permute_dims125: R.Tensor((384, 51865), dtype="float32") = R.permute_dims(proj_out_weight1, axes=None)
            gv: R.Tensor((1, 1, 51865), dtype="float32") = R.matmul(layer_norm21, permute_dims125, out_dtype="void")
            R.output(gv)
        return gv

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
            cached_encoder_key_value_0_0: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_0[0]
            cached_encoder_key_value_0_1: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_0[1]
            cached_encoder_key_value_1: R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")) = cached_encoder_key_value[1]
            cached_encoder_key_value_1_0: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_1[0]
            cached_encoder_key_value_1_1: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_1[1]
            cached_encoder_key_value_2: R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")) = cached_encoder_key_value[2]
            cached_encoder_key_value_2_0: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_2[0]
            cached_encoder_key_value_2_1: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_2[1]
            cached_encoder_key_value_3: R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")) = cached_encoder_key_value[3]
            cached_encoder_key_value_3_0: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_3[0]
            cached_encoder_key_value_3_1: R.Tensor((1, 1500, 6, 64), dtype="float32") = cached_encoder_key_value_3[1]
            reshape58: R.Tensor((seq_len,), dtype="int32") = R.reshape(input_ids, R.shape([seq_len]))
            take1: R.Tensor((seq_len, 384), dtype="float32") = R.take(model_decoder_embed_tokens_weight2, reshape58, axis=0)
            reshape59: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(take1, R.shape([1, seq_len, 384]))
            lv38 = R.call_tir(cls.position_embedding1, (input_ids, model_decoder_embed_positions_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"), tir_vars=R.shape([total_seq_len]))
            add74: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(reshape59, lv38)
            layer_norm22: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add74, model_decoder_layers_0_self_attn_layer_norm_weight2, model_decoder_layers_0_self_attn_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv77 = R.call_tir(cls.NT_matmul10, (layer_norm22, model_decoder_layers_0_self_attn_q_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add75: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv77, model_decoder_layers_0_self_attn_q_proj_bias2)
            mul12: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add75, R.const(0.125, "float32"))
            reshape60: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul12, R.shape([1, seq_len, 6, 64]))
            lv78 = R.call_tir(cls.NT_matmul10, (layer_norm22, model_decoder_layers_0_self_attn_k_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape61: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(lv78, R.shape([1, seq_len, 6, 64]))
            lv79 = R.call_tir(cls.NT_matmul10, (layer_norm22, model_decoder_layers_0_self_attn_v_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add76: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv79, model_decoder_layers_0_self_attn_v_proj_bias2)
            reshape62: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(add76, R.shape([1, seq_len, 6, 64]))
            squeeze8: R.Tensor((seq_len, 6, 64), dtype="float32") = R.squeeze(reshape61, axis=[0])
            lv39: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_0_self_attn_k_cache3, squeeze8, sinfo_args=(R.Object,))
            squeeze9: R.Tensor((seq_len, 6, 64), dtype="float32") = R.squeeze(reshape62, axis=[0])
            lv40: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_0_self_attn_v_cache3, squeeze9, sinfo_args=(R.Object,))
            lv41: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv39, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape63: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv41, R.shape([1, total_seq_len, 6, 64]))
            lv42: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv40, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape64: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv42, R.shape([1, total_seq_len, 6, 64]))
            permute_dims129: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.permute_dims(reshape60, axes=[0, 2, 1, 3])
            permute_dims130: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape63, axes=[0, 2, 1, 3])
            permute_dims131: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape64, axes=[0, 2, 1, 3])
            lv80 = R.call_tir(cls.NT_matmul11, (permute_dims129, permute_dims130), out_sinfo=R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32"))
            maximum16: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.maximum(lv80, R.const(-3.4028234663852886e+38, "float32"))
            minimum16: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.minimum(maximum16, R.const(3.4028234663852886e+38, "float32"))
            softmax12: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.nn.softmax(minimum16, axis=-1)
            matmul93: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax12, permute_dims131, out_dtype="void")
            permute_dims133: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul93, axes=[0, 2, 1, 3])
            reshape65: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims133, R.shape([1, seq_len, 384]))
            lv81 = R.call_tir(cls.NT_matmul10, (reshape65, model_decoder_layers_0_self_attn_out_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add77: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv81, model_decoder_layers_0_self_attn_out_proj_bias2)
            add78: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add74, add77)
            layer_norm23: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add78, model_decoder_layers_0_encoder_attn_layer_norm_weight2, model_decoder_layers_0_encoder_attn_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv82 = R.call_tir(cls.NT_matmul10, (layer_norm23, model_decoder_layers_0_encoder_attn_q_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add79: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv82, model_decoder_layers_0_encoder_attn_q_proj_bias2)
            mul13: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add79, R.const(0.125, "float32"))
            reshape66: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul13, R.shape([1, seq_len, 6, 64]))
            permute_dims136: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.permute_dims(reshape66, axes=[0, 2, 1, 3])
            permute_dims137: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_0_0, axes=[0, 2, 1, 3])
            permute_dims138: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_0_1, axes=[0, 2, 1, 3])
            lv83 = R.call_tir(cls.NT_matmul12, (permute_dims136, permute_dims137), out_sinfo=R.Tensor((1, 6, seq_len, 1500), dtype="float32"))
            maximum17: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.maximum(lv83, R.const(-3.4028234663852886e+38, "float32"))
            minimum17: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.minimum(maximum17, R.const(3.4028234663852886e+38, "float32"))
            softmax13: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.nn.softmax(minimum17, axis=-1)
            matmul97: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax13, permute_dims138, out_dtype="void")
            permute_dims140: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul97, axes=[0, 2, 1, 3])
            reshape67: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims140, R.shape([1, seq_len, 384]))
            lv84 = R.call_tir(cls.NT_matmul10, (reshape67, model_decoder_layers_0_encoder_attn_out_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add80: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv84, model_decoder_layers_0_encoder_attn_out_proj_bias2)
            add81: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add78, add80)
            layer_norm24: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add81, model_decoder_layers_0_final_layer_norm_weight2, model_decoder_layers_0_final_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv85 = R.call_tir(cls.NT_matmul13, (layer_norm24, model_decoder_layers_0_fc1_weight2), out_sinfo=R.Tensor((1, seq_len, 1536), dtype="float32"))
            add82: R.Tensor((1, seq_len, 1536), dtype="float32") = R.add(lv85, model_decoder_layers_0_fc1_bias2)
            gelu10: R.Tensor((1, seq_len, 1536), dtype="float32") = R.nn.gelu(add82)
            lv86 = R.call_tir(cls.NT_matmul14, (gelu10, model_decoder_layers_0_fc2_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add83: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv86, model_decoder_layers_0_fc2_bias2)
            add84: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add81, add83)
            layer_norm25: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add84, model_decoder_layers_1_self_attn_layer_norm_weight2, model_decoder_layers_1_self_attn_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv87 = R.call_tir(cls.NT_matmul10, (layer_norm25, model_decoder_layers_1_self_attn_q_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add85: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv87, model_decoder_layers_1_self_attn_q_proj_bias2)
            mul14: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add85, R.const(0.125, "float32"))
            reshape68: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul14, R.shape([1, seq_len, 6, 64]))
            lv88 = R.call_tir(cls.NT_matmul10, (layer_norm25, model_decoder_layers_1_self_attn_k_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape69: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(lv88, R.shape([1, seq_len, 6, 64]))
            lv89 = R.call_tir(cls.NT_matmul10, (layer_norm25, model_decoder_layers_1_self_attn_v_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add86: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv89, model_decoder_layers_1_self_attn_v_proj_bias2)
            reshape70: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(add86, R.shape([1, seq_len, 6, 64]))
            squeeze10: R.Tensor((seq_len, 6, 64), dtype="float32") = R.squeeze(reshape69, axis=[0])
            lv43: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_1_self_attn_k_cache3, squeeze10, sinfo_args=(R.Object,))
            squeeze11: R.Tensor((seq_len, 6, 64), dtype="float32") = R.squeeze(reshape70, axis=[0])
            lv44: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_1_self_attn_v_cache3, squeeze11, sinfo_args=(R.Object,))
            lv45: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv43, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape71: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv45, R.shape([1, total_seq_len, 6, 64]))
            lv46: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv44, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape72: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv46, R.shape([1, total_seq_len, 6, 64]))
            permute_dims147: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.permute_dims(reshape68, axes=[0, 2, 1, 3])
            permute_dims148: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape71, axes=[0, 2, 1, 3])
            permute_dims149: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape72, axes=[0, 2, 1, 3])
            lv90 = R.call_tir(cls.NT_matmul11, (permute_dims147, permute_dims148), out_sinfo=R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32"))
            maximum18: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.maximum(lv90, R.const(-3.4028234663852886e+38, "float32"))
            minimum18: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.minimum(maximum18, R.const(3.4028234663852886e+38, "float32"))
            softmax14: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.nn.softmax(minimum18, axis=-1)
            matmul105: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax14, permute_dims149, out_dtype="void")
            permute_dims151: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul105, axes=[0, 2, 1, 3])
            reshape73: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims151, R.shape([1, seq_len, 384]))
            lv91 = R.call_tir(cls.NT_matmul10, (reshape73, model_decoder_layers_1_self_attn_out_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add87: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv91, model_decoder_layers_1_self_attn_out_proj_bias2)
            add88: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add84, add87)
            layer_norm26: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add88, model_decoder_layers_1_encoder_attn_layer_norm_weight2, model_decoder_layers_1_encoder_attn_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv92 = R.call_tir(cls.NT_matmul10, (layer_norm26, model_decoder_layers_1_encoder_attn_q_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add89: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv92, model_decoder_layers_1_encoder_attn_q_proj_bias2)
            mul15: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add89, R.const(0.125, "float32"))
            reshape74: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul15, R.shape([1, seq_len, 6, 64]))
            permute_dims154: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.permute_dims(reshape74, axes=[0, 2, 1, 3])
            permute_dims155: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_1_0, axes=[0, 2, 1, 3])
            permute_dims156: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_1_1, axes=[0, 2, 1, 3])
            lv93 = R.call_tir(cls.NT_matmul12, (permute_dims154, permute_dims155), out_sinfo=R.Tensor((1, 6, seq_len, 1500), dtype="float32"))
            maximum19: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.maximum(lv93, R.const(-3.4028234663852886e+38, "float32"))
            minimum19: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.minimum(maximum19, R.const(3.4028234663852886e+38, "float32"))
            softmax15: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.nn.softmax(minimum19, axis=-1)
            matmul109: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax15, permute_dims156, out_dtype="void")
            permute_dims158: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul109, axes=[0, 2, 1, 3])
            reshape75: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims158, R.shape([1, seq_len, 384]))
            lv94 = R.call_tir(cls.NT_matmul10, (reshape75, model_decoder_layers_1_encoder_attn_out_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add90: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv94, model_decoder_layers_1_encoder_attn_out_proj_bias2)
            add91: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add88, add90)
            layer_norm27: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add91, model_decoder_layers_1_final_layer_norm_weight2, model_decoder_layers_1_final_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv95 = R.call_tir(cls.NT_matmul13, (layer_norm27, model_decoder_layers_1_fc1_weight2), out_sinfo=R.Tensor((1, seq_len, 1536), dtype="float32"))
            add92: R.Tensor((1, seq_len, 1536), dtype="float32") = R.add(lv95, model_decoder_layers_1_fc1_bias2)
            gelu11: R.Tensor((1, seq_len, 1536), dtype="float32") = R.nn.gelu(add92)
            lv96 = R.call_tir(cls.NT_matmul14, (gelu11, model_decoder_layers_1_fc2_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add93: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv96, model_decoder_layers_1_fc2_bias2)
            add94: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add91, add93)
            layer_norm28: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add94, model_decoder_layers_2_self_attn_layer_norm_weight2, model_decoder_layers_2_self_attn_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv97 = R.call_tir(cls.NT_matmul10, (layer_norm28, model_decoder_layers_2_self_attn_q_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add95: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv97, model_decoder_layers_2_self_attn_q_proj_bias2)
            mul16: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add95, R.const(0.125, "float32"))
            reshape76: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul16, R.shape([1, seq_len, 6, 64]))
            lv98 = R.call_tir(cls.NT_matmul10, (layer_norm28, model_decoder_layers_2_self_attn_k_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape77: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(lv98, R.shape([1, seq_len, 6, 64]))
            lv99 = R.call_tir(cls.NT_matmul10, (layer_norm28, model_decoder_layers_2_self_attn_v_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add96: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv99, model_decoder_layers_2_self_attn_v_proj_bias2)
            reshape78: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(add96, R.shape([1, seq_len, 6, 64]))
            squeeze12: R.Tensor((seq_len, 6, 64), dtype="float32") = R.squeeze(reshape77, axis=[0])
            lv47: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_2_self_attn_k_cache3, squeeze12, sinfo_args=(R.Object,))
            squeeze13: R.Tensor((seq_len, 6, 64), dtype="float32") = R.squeeze(reshape78, axis=[0])
            lv48: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_2_self_attn_v_cache3, squeeze13, sinfo_args=(R.Object,))
            lv49: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv47, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape79: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv49, R.shape([1, total_seq_len, 6, 64]))
            lv50: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv48, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape80: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv50, R.shape([1, total_seq_len, 6, 64]))
            permute_dims165: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.permute_dims(reshape76, axes=[0, 2, 1, 3])
            permute_dims166: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape79, axes=[0, 2, 1, 3])
            permute_dims167: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape80, axes=[0, 2, 1, 3])
            lv100 = R.call_tir(cls.NT_matmul11, (permute_dims165, permute_dims166), out_sinfo=R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32"))
            maximum20: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.maximum(lv100, R.const(-3.4028234663852886e+38, "float32"))
            minimum20: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.minimum(maximum20, R.const(3.4028234663852886e+38, "float32"))
            softmax16: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.nn.softmax(minimum20, axis=-1)
            matmul117: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax16, permute_dims167, out_dtype="void")
            permute_dims169: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul117, axes=[0, 2, 1, 3])
            reshape81: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims169, R.shape([1, seq_len, 384]))
            lv101 = R.call_tir(cls.NT_matmul10, (reshape81, model_decoder_layers_2_self_attn_out_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add97: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv101, model_decoder_layers_2_self_attn_out_proj_bias2)
            add98: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add94, add97)
            layer_norm29: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add98, model_decoder_layers_2_encoder_attn_layer_norm_weight2, model_decoder_layers_2_encoder_attn_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv102 = R.call_tir(cls.NT_matmul10, (layer_norm29, model_decoder_layers_2_encoder_attn_q_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add99: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv102, model_decoder_layers_2_encoder_attn_q_proj_bias2)
            mul17: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add99, R.const(0.125, "float32"))
            reshape82: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul17, R.shape([1, seq_len, 6, 64]))
            permute_dims172: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.permute_dims(reshape82, axes=[0, 2, 1, 3])
            permute_dims173: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_2_0, axes=[0, 2, 1, 3])
            permute_dims174: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_2_1, axes=[0, 2, 1, 3])
            lv103 = R.call_tir(cls.NT_matmul12, (permute_dims172, permute_dims173), out_sinfo=R.Tensor((1, 6, seq_len, 1500), dtype="float32"))
            maximum21: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.maximum(lv103, R.const(-3.4028234663852886e+38, "float32"))
            minimum21: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.minimum(maximum21, R.const(3.4028234663852886e+38, "float32"))
            softmax17: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.nn.softmax(minimum21, axis=-1)
            matmul121: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax17, permute_dims174, out_dtype="void")
            permute_dims176: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul121, axes=[0, 2, 1, 3])
            reshape83: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims176, R.shape([1, seq_len, 384]))
            lv104 = R.call_tir(cls.NT_matmul10, (reshape83, model_decoder_layers_2_encoder_attn_out_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add100: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv104, model_decoder_layers_2_encoder_attn_out_proj_bias2)
            add101: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add98, add100)
            layer_norm30: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add101, model_decoder_layers_2_final_layer_norm_weight2, model_decoder_layers_2_final_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv105 = R.call_tir(cls.NT_matmul13, (layer_norm30, model_decoder_layers_2_fc1_weight2), out_sinfo=R.Tensor((1, seq_len, 1536), dtype="float32"))
            add102: R.Tensor((1, seq_len, 1536), dtype="float32") = R.add(lv105, model_decoder_layers_2_fc1_bias2)
            gelu12: R.Tensor((1, seq_len, 1536), dtype="float32") = R.nn.gelu(add102)
            lv106 = R.call_tir(cls.NT_matmul14, (gelu12, model_decoder_layers_2_fc2_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add103: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv106, model_decoder_layers_2_fc2_bias2)
            add104: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add101, add103)
            layer_norm31: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add104, model_decoder_layers_3_self_attn_layer_norm_weight2, model_decoder_layers_3_self_attn_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv107 = R.call_tir(cls.NT_matmul10, (layer_norm31, model_decoder_layers_3_self_attn_q_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add105: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv107, model_decoder_layers_3_self_attn_q_proj_bias2)
            mul18: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add105, R.const(0.125, "float32"))
            reshape84: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul18, R.shape([1, seq_len, 6, 64]))
            lv108 = R.call_tir(cls.NT_matmul10, (layer_norm31, model_decoder_layers_3_self_attn_k_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            reshape85: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(lv108, R.shape([1, seq_len, 6, 64]))
            lv109 = R.call_tir(cls.NT_matmul10, (layer_norm31, model_decoder_layers_3_self_attn_v_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add106: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv109, model_decoder_layers_3_self_attn_v_proj_bias2)
            reshape86: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(add106, R.shape([1, seq_len, 6, 64]))
            squeeze14: R.Tensor((seq_len, 6, 64), dtype="float32") = R.squeeze(reshape85, axis=[0])
            lv51: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_3_self_attn_k_cache3, squeeze14, sinfo_args=(R.Object,))
            squeeze15: R.Tensor((seq_len, 6, 64), dtype="float32") = R.squeeze(reshape86, axis=[0])
            lv52: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_3_self_attn_v_cache3, squeeze15, sinfo_args=(R.Object,))
            lv53: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv51, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape87: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv53, R.shape([1, total_seq_len, 6, 64]))
            lv54: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv52, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape88: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv54, R.shape([1, total_seq_len, 6, 64]))
            permute_dims183: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.permute_dims(reshape84, axes=[0, 2, 1, 3])
            permute_dims184: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape87, axes=[0, 2, 1, 3])
            permute_dims185: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape88, axes=[0, 2, 1, 3])
            lv110 = R.call_tir(cls.NT_matmul11, (permute_dims183, permute_dims184), out_sinfo=R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32"))
            maximum22: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.maximum(lv110, R.const(-3.4028234663852886e+38, "float32"))
            minimum22: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.minimum(maximum22, R.const(3.4028234663852886e+38, "float32"))
            softmax18: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.nn.softmax(minimum22, axis=-1)
            matmul129: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax18, permute_dims185, out_dtype="void")
            permute_dims187: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul129, axes=[0, 2, 1, 3])
            reshape89: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims187, R.shape([1, seq_len, 384]))
            lv111 = R.call_tir(cls.NT_matmul10, (reshape89, model_decoder_layers_3_self_attn_out_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add107: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv111, model_decoder_layers_3_self_attn_out_proj_bias2)
            add108: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add104, add107)
            layer_norm32: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add108, model_decoder_layers_3_encoder_attn_layer_norm_weight2, model_decoder_layers_3_encoder_attn_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv112 = R.call_tir(cls.NT_matmul10, (layer_norm32, model_decoder_layers_3_encoder_attn_q_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add109: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv112, model_decoder_layers_3_encoder_attn_q_proj_bias2)
            mul19: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add109, R.const(0.125, "float32"))
            reshape90: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul19, R.shape([1, seq_len, 6, 64]))
            permute_dims190: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.permute_dims(reshape90, axes=[0, 2, 1, 3])
            permute_dims191: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_3_0, axes=[0, 2, 1, 3])
            permute_dims192: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_3_1, axes=[0, 2, 1, 3])
            lv113 = R.call_tir(cls.NT_matmul12, (permute_dims190, permute_dims191), out_sinfo=R.Tensor((1, 6, seq_len, 1500), dtype="float32"))
            maximum23: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.maximum(lv113, R.const(-3.4028234663852886e+38, "float32"))
            minimum23: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.minimum(maximum23, R.const(3.4028234663852886e+38, "float32"))
            softmax19: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.nn.softmax(minimum23, axis=-1)
            matmul133: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax19, permute_dims192, out_dtype="void")
            permute_dims194: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul133, axes=[0, 2, 1, 3])
            reshape91: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims194, R.shape([1, seq_len, 384]))
            lv114 = R.call_tir(cls.NT_matmul10, (reshape91, model_decoder_layers_3_encoder_attn_out_proj_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add110: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv114, model_decoder_layers_3_encoder_attn_out_proj_bias2)
            add111: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add108, add110)
            layer_norm33: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add111, model_decoder_layers_3_final_layer_norm_weight2, model_decoder_layers_3_final_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv115 = R.call_tir(cls.NT_matmul13, (layer_norm33, model_decoder_layers_3_fc1_weight2), out_sinfo=R.Tensor((1, seq_len, 1536), dtype="float32"))
            add112: R.Tensor((1, seq_len, 1536), dtype="float32") = R.add(lv115, model_decoder_layers_3_fc1_bias2)
            gelu13: R.Tensor((1, seq_len, 1536), dtype="float32") = R.nn.gelu(add112)
            lv116 = R.call_tir(cls.NT_matmul14, (gelu13, model_decoder_layers_3_fc2_weight2), out_sinfo=R.Tensor((1, seq_len, 384), dtype="float32"))
            add113: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(lv116, model_decoder_layers_3_fc2_bias2)
            add114: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add111, add113)
            layer_norm34: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add114, model_decoder_layer_norm_weight2, model_decoder_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            lv117 = R.call_tir(cls.NT_matmul15, (layer_norm34, proj_out_weight2), out_sinfo=R.Tensor((1, seq_len, 51865), dtype="float32"))
            gv3: R.Tuple(R.Tensor((1, seq_len, 51865), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)) = lv117, (lv39, lv40, model_decoder_layers_0_encoder_attn_k_cache3, model_decoder_layers_0_encoder_attn_v_cache3, lv43, lv44, model_decoder_layers_1_encoder_attn_k_cache3, model_decoder_layers_1_encoder_attn_v_cache3, lv47, lv48, model_decoder_layers_2_encoder_attn_k_cache3, model_decoder_layers_2_encoder_attn_v_cache3, lv51, lv52, model_decoder_layers_3_encoder_attn_k_cache3, model_decoder_layers_3_encoder_attn_v_cache3)
            R.output(gv3)
        return gv3