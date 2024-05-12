# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
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
            model_encoder_conv1_weight1: R.Tensor((384, 80, 3), dtype="float32") = packed_params[0]
            model_encoder_conv1_bias1: R.Tensor((384,), dtype="float32") = packed_params[1]
            model_encoder_conv2_weight1: R.Tensor((384, 384, 3), dtype="float32") = packed_params[2]
            model_encoder_conv2_bias1: R.Tensor((384,), dtype="float32") = packed_params[3]
            model_encoder_embed_positions_weight1: R.Tensor((1500, 384), dtype="float32") = packed_params[4]
            model_encoder_layers_0_self_attn_k_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[5]
            model_encoder_layers_0_self_attn_v_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[6]
            model_encoder_layers_0_self_attn_v_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[7]
            model_encoder_layers_0_self_attn_q_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[8]
            model_encoder_layers_0_self_attn_q_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[9]
            model_encoder_layers_0_self_attn_out_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[10]
            model_encoder_layers_0_self_attn_out_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[11]
            model_encoder_layers_0_self_attn_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[12]
            model_encoder_layers_0_self_attn_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[13]
            model_encoder_layers_0_fc1_weight1: R.Tensor((1536, 384), dtype="float32") = packed_params[14]
            model_encoder_layers_0_fc1_bias1: R.Tensor((1536,), dtype="float32") = packed_params[15]
            model_encoder_layers_0_fc2_weight1: R.Tensor((384, 1536), dtype="float32") = packed_params[16]
            model_encoder_layers_0_fc2_bias1: R.Tensor((384,), dtype="float32") = packed_params[17]
            model_encoder_layers_0_final_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[18]
            model_encoder_layers_0_final_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[19]
            model_encoder_layers_1_self_attn_k_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[20]
            model_encoder_layers_1_self_attn_v_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[21]
            model_encoder_layers_1_self_attn_v_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[22]
            model_encoder_layers_1_self_attn_q_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[23]
            model_encoder_layers_1_self_attn_q_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[24]
            model_encoder_layers_1_self_attn_out_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[25]
            model_encoder_layers_1_self_attn_out_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[26]
            model_encoder_layers_1_self_attn_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[27]
            model_encoder_layers_1_self_attn_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[28]
            model_encoder_layers_1_fc1_weight1: R.Tensor((1536, 384), dtype="float32") = packed_params[29]
            model_encoder_layers_1_fc1_bias1: R.Tensor((1536,), dtype="float32") = packed_params[30]
            model_encoder_layers_1_fc2_weight1: R.Tensor((384, 1536), dtype="float32") = packed_params[31]
            model_encoder_layers_1_fc2_bias1: R.Tensor((384,), dtype="float32") = packed_params[32]
            model_encoder_layers_1_final_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[33]
            model_encoder_layers_1_final_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[34]
            model_encoder_layers_2_self_attn_k_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[35]
            model_encoder_layers_2_self_attn_v_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[36]
            model_encoder_layers_2_self_attn_v_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[37]
            model_encoder_layers_2_self_attn_q_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[38]
            model_encoder_layers_2_self_attn_q_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[39]
            model_encoder_layers_2_self_attn_out_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[40]
            model_encoder_layers_2_self_attn_out_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[41]
            model_encoder_layers_2_self_attn_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[42]
            model_encoder_layers_2_self_attn_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[43]
            model_encoder_layers_2_fc1_weight1: R.Tensor((1536, 384), dtype="float32") = packed_params[44]
            model_encoder_layers_2_fc1_bias1: R.Tensor((1536,), dtype="float32") = packed_params[45]
            model_encoder_layers_2_fc2_weight1: R.Tensor((384, 1536), dtype="float32") = packed_params[46]
            model_encoder_layers_2_fc2_bias1: R.Tensor((384,), dtype="float32") = packed_params[47]
            model_encoder_layers_2_final_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[48]
            model_encoder_layers_2_final_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[49]
            model_encoder_layers_3_self_attn_k_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[50]
            model_encoder_layers_3_self_attn_v_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[51]
            model_encoder_layers_3_self_attn_v_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[52]
            model_encoder_layers_3_self_attn_q_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[53]
            model_encoder_layers_3_self_attn_q_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[54]
            model_encoder_layers_3_self_attn_out_proj_weight1: R.Tensor((384, 384), dtype="float32") = packed_params[55]
            model_encoder_layers_3_self_attn_out_proj_bias1: R.Tensor((384,), dtype="float32") = packed_params[56]
            model_encoder_layers_3_self_attn_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[57]
            model_encoder_layers_3_self_attn_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[58]
            model_encoder_layers_3_fc1_weight1: R.Tensor((1536, 384), dtype="float32") = packed_params[59]
            model_encoder_layers_3_fc1_bias1: R.Tensor((1536,), dtype="float32") = packed_params[60]
            model_encoder_layers_3_fc2_weight1: R.Tensor((384, 1536), dtype="float32") = packed_params[61]
            model_encoder_layers_3_fc2_bias1: R.Tensor((384,), dtype="float32") = packed_params[62]
            model_encoder_layers_3_final_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[63]
            model_encoder_layers_3_final_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[64]
            model_encoder_layer_norm_weight1: R.Tensor((384,), dtype="float32") = packed_params[65]
            model_encoder_layer_norm_bias1: R.Tensor((384,), dtype="float32") = packed_params[66]
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
            permute_dims45: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_self_attn_q_proj_weight1, axes=None)
            matmul32: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm9, permute_dims45, out_dtype="void")
            add30: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul32, model_decoder_layers_0_self_attn_q_proj_bias1)
            mul4: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add30, R.const(0.125, "float32"))
            reshape18: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul4, R.shape([1, 1, 6, 64]))
            permute_dims46: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_self_attn_k_proj_weight1, axes=None)
            matmul33: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm9, permute_dims46, out_dtype="void")
            reshape19: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(matmul33, R.shape([1, 1, 6, 64]))
            permute_dims47: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_self_attn_v_proj_weight1, axes=None)
            matmul34: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm9, permute_dims47, out_dtype="void")
            add31: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul34, model_decoder_layers_0_self_attn_v_proj_bias1)
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
            permute_dims51: R.Tensor((1, 6, 64, total_seq_len), dtype="float32") = R.permute_dims(permute_dims49, axes=[0, 1, 3, 2])
            matmul35: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.matmul(permute_dims48, permute_dims51, out_dtype="void")
            maximum8: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.maximum(matmul35, R.const(-3.4028234663852886e+38, "float32"))
            minimum8: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.minimum(maximum8, R.const(3.4028234663852886e+38, "float32"))
            softmax4: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.nn.softmax(minimum8, axis=-1)
            matmul36: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax4, permute_dims50, out_dtype="void")
            permute_dims52: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul36, axes=[0, 2, 1, 3])
            reshape23: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims52, R.shape([1, 1, 384]))
            permute_dims53: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_self_attn_out_proj_weight1, axes=None)
            matmul37: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(reshape23, permute_dims53, out_dtype="void")
            add32: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul37, model_decoder_layers_0_self_attn_out_proj_bias1)
            add33: R.Tensor((1, 1, 384), dtype="float32") = R.add(add29, add32)
            layer_norm10: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add33, model_decoder_layers_0_encoder_attn_layer_norm_weight1, model_decoder_layers_0_encoder_attn_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims54: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_encoder_attn_q_proj_weight1, axes=None)
            matmul38: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm10, permute_dims54, out_dtype="void")
            add34: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul38, model_decoder_layers_0_encoder_attn_q_proj_bias1)
            mul5: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add34, R.const(0.125, "float32"))
            reshape24: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul5, R.shape([1, 1, 6, 64]))
            permute_dims55: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_encoder_attn_k_proj_weight1, axes=None)
            matmul39: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(encoder_hidden_states, permute_dims55, out_dtype="void")
            reshape25: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(matmul39, R.shape([1, 1500, 6, 64]))
            permute_dims56: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_encoder_attn_v_proj_weight1, axes=None)
            matmul40: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(encoder_hidden_states, permute_dims56, out_dtype="void")
            add35: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul40, model_decoder_layers_0_encoder_attn_v_proj_bias1)
            reshape26: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add35, R.shape([1, 1500, 6, 64]))
            permute_dims57: R.Tensor((1, 6, 1, 64), dtype="float32") = R.permute_dims(reshape24, axes=[0, 2, 1, 3])
            permute_dims58: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape25, axes=[0, 2, 1, 3])
            permute_dims59: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape26, axes=[0, 2, 1, 3])
            permute_dims60: R.Tensor((1, 6, 64, 1500), dtype="float32") = R.permute_dims(permute_dims58, axes=[0, 1, 3, 2])
            matmul41: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.matmul(permute_dims57, permute_dims60, out_dtype="void")
            maximum9: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.maximum(matmul41, R.const(-3.4028234663852886e+38, "float32"))
            minimum9: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.minimum(maximum9, R.const(3.4028234663852886e+38, "float32"))
            softmax5: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.nn.softmax(minimum9, axis=-1)
            matmul42: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax5, permute_dims59, out_dtype="void")
            permute_dims61: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul42, axes=[0, 2, 1, 3])
            reshape27: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims61, R.shape([1, 1, 384]))
            permute_dims62: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_encoder_attn_out_proj_weight1, axes=None)
            matmul43: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(reshape27, permute_dims62, out_dtype="void")
            add36: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul43, model_decoder_layers_0_encoder_attn_out_proj_bias1)
            add37: R.Tensor((1, 1, 384), dtype="float32") = R.add(add33, add36)
            layer_norm11: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add37, model_decoder_layers_0_final_layer_norm_weight1, model_decoder_layers_0_final_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims63: R.Tensor((384, 1536), dtype="float32") = R.permute_dims(model_decoder_layers_0_fc1_weight1, axes=None)
            matmul44: R.Tensor((1, 1, 1536), dtype="float32") = R.matmul(layer_norm11, permute_dims63, out_dtype="void")
            add38: R.Tensor((1, 1, 1536), dtype="float32") = R.add(matmul44, model_decoder_layers_0_fc1_bias1)
            gelu6: R.Tensor((1, 1, 1536), dtype="float32") = R.nn.gelu(add38)
            permute_dims64: R.Tensor((1536, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_fc2_weight1, axes=None)
            matmul45: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(gelu6, permute_dims64, out_dtype="void")
            add39: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul45, model_decoder_layers_0_fc2_bias1)
            add40: R.Tensor((1, 1, 384), dtype="float32") = R.add(add37, add39)
            layer_norm12: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add40, model_decoder_layers_1_self_attn_layer_norm_weight1, model_decoder_layers_1_self_attn_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims65: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_self_attn_q_proj_weight1, axes=None)
            matmul46: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm12, permute_dims65, out_dtype="void")
            add41: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul46, model_decoder_layers_1_self_attn_q_proj_bias1)
            mul6: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add41, R.const(0.125, "float32"))
            reshape28: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul6, R.shape([1, 1, 6, 64]))
            permute_dims66: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_self_attn_k_proj_weight1, axes=None)
            matmul47: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm12, permute_dims66, out_dtype="void")
            reshape29: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(matmul47, R.shape([1, 1, 6, 64]))
            permute_dims67: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_self_attn_v_proj_weight1, axes=None)
            matmul48: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm12, permute_dims67, out_dtype="void")
            add42: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul48, model_decoder_layers_1_self_attn_v_proj_bias1)
            reshape30: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(add42, R.shape([1, 1, 6, 64]))
            squeeze2: R.Tensor((1, 6, 64), dtype="float32") = R.squeeze(reshape29, axis=[0])
            lv26: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_1_self_attn_k_cache2, squeeze2, sinfo_args=(R.Object,))
            squeeze3: R.Tensor((1, 6, 64), dtype="float32") = R.squeeze(reshape30, axis=[0])
            lv27: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_1_self_attn_v_cache2, squeeze3, sinfo_args=(R.Object,))
            lv28: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv26, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape31: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv28, R.shape([1, total_seq_len, 6, 64]))
            lv29: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv27, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape32: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv29, R.shape([1, total_seq_len, 6, 64]))
            permute_dims68: R.Tensor((1, 6, 1, 64), dtype="float32") = R.permute_dims(reshape28, axes=[0, 2, 1, 3])
            permute_dims69: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape31, axes=[0, 2, 1, 3])
            permute_dims70: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape32, axes=[0, 2, 1, 3])
            permute_dims71: R.Tensor((1, 6, 64, total_seq_len), dtype="float32") = R.permute_dims(permute_dims69, axes=[0, 1, 3, 2])
            matmul49: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.matmul(permute_dims68, permute_dims71, out_dtype="void")
            maximum10: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.maximum(matmul49, R.const(-3.4028234663852886e+38, "float32"))
            minimum10: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.minimum(maximum10, R.const(3.4028234663852886e+38, "float32"))
            softmax6: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.nn.softmax(minimum10, axis=-1)
            matmul50: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax6, permute_dims70, out_dtype="void")
            permute_dims72: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul50, axes=[0, 2, 1, 3])
            reshape33: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims72, R.shape([1, 1, 384]))
            permute_dims73: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_self_attn_out_proj_weight1, axes=None)
            matmul51: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(reshape33, permute_dims73, out_dtype="void")
            add43: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul51, model_decoder_layers_1_self_attn_out_proj_bias1)
            add44: R.Tensor((1, 1, 384), dtype="float32") = R.add(add40, add43)
            layer_norm13: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add44, model_decoder_layers_1_encoder_attn_layer_norm_weight1, model_decoder_layers_1_encoder_attn_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims74: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_encoder_attn_q_proj_weight1, axes=None)
            matmul52: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm13, permute_dims74, out_dtype="void")
            add45: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul52, model_decoder_layers_1_encoder_attn_q_proj_bias1)
            mul7: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add45, R.const(0.125, "float32"))
            reshape34: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul7, R.shape([1, 1, 6, 64]))
            permute_dims75: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_encoder_attn_k_proj_weight1, axes=None)
            matmul53: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(encoder_hidden_states, permute_dims75, out_dtype="void")
            reshape35: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(matmul53, R.shape([1, 1500, 6, 64]))
            permute_dims76: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_encoder_attn_v_proj_weight1, axes=None)
            matmul54: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(encoder_hidden_states, permute_dims76, out_dtype="void")
            add46: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul54, model_decoder_layers_1_encoder_attn_v_proj_bias1)
            reshape36: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add46, R.shape([1, 1500, 6, 64]))
            permute_dims77: R.Tensor((1, 6, 1, 64), dtype="float32") = R.permute_dims(reshape34, axes=[0, 2, 1, 3])
            permute_dims78: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape35, axes=[0, 2, 1, 3])
            permute_dims79: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape36, axes=[0, 2, 1, 3])
            permute_dims80: R.Tensor((1, 6, 64, 1500), dtype="float32") = R.permute_dims(permute_dims78, axes=[0, 1, 3, 2])
            matmul55: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.matmul(permute_dims77, permute_dims80, out_dtype="void")
            maximum11: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.maximum(matmul55, R.const(-3.4028234663852886e+38, "float32"))
            minimum11: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.minimum(maximum11, R.const(3.4028234663852886e+38, "float32"))
            softmax7: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.nn.softmax(minimum11, axis=-1)
            matmul56: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax7, permute_dims79, out_dtype="void")
            permute_dims81: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul56, axes=[0, 2, 1, 3])
            reshape37: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims81, R.shape([1, 1, 384]))
            permute_dims82: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_encoder_attn_out_proj_weight1, axes=None)
            matmul57: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(reshape37, permute_dims82, out_dtype="void")
            add47: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul57, model_decoder_layers_1_encoder_attn_out_proj_bias1)
            add48: R.Tensor((1, 1, 384), dtype="float32") = R.add(add44, add47)
            layer_norm14: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add48, model_decoder_layers_1_final_layer_norm_weight1, model_decoder_layers_1_final_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims83: R.Tensor((384, 1536), dtype="float32") = R.permute_dims(model_decoder_layers_1_fc1_weight1, axes=None)
            matmul58: R.Tensor((1, 1, 1536), dtype="float32") = R.matmul(layer_norm14, permute_dims83, out_dtype="void")
            add49: R.Tensor((1, 1, 1536), dtype="float32") = R.add(matmul58, model_decoder_layers_1_fc1_bias1)
            gelu7: R.Tensor((1, 1, 1536), dtype="float32") = R.nn.gelu(add49)
            permute_dims84: R.Tensor((1536, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_fc2_weight1, axes=None)
            matmul59: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(gelu7, permute_dims84, out_dtype="void")
            add50: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul59, model_decoder_layers_1_fc2_bias1)
            add51: R.Tensor((1, 1, 384), dtype="float32") = R.add(add48, add50)
            layer_norm15: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add51, model_decoder_layers_2_self_attn_layer_norm_weight1, model_decoder_layers_2_self_attn_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims85: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_self_attn_q_proj_weight1, axes=None)
            matmul60: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm15, permute_dims85, out_dtype="void")
            add52: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul60, model_decoder_layers_2_self_attn_q_proj_bias1)
            mul8: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add52, R.const(0.125, "float32"))
            reshape38: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul8, R.shape([1, 1, 6, 64]))
            permute_dims86: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_self_attn_k_proj_weight1, axes=None)
            matmul61: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm15, permute_dims86, out_dtype="void")
            reshape39: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(matmul61, R.shape([1, 1, 6, 64]))
            permute_dims87: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_self_attn_v_proj_weight1, axes=None)
            matmul62: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm15, permute_dims87, out_dtype="void")
            add53: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul62, model_decoder_layers_2_self_attn_v_proj_bias1)
            reshape40: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(add53, R.shape([1, 1, 6, 64]))
            squeeze4: R.Tensor((1, 6, 64), dtype="float32") = R.squeeze(reshape39, axis=[0])
            lv30: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_2_self_attn_k_cache2, squeeze4, sinfo_args=(R.Object,))
            squeeze5: R.Tensor((1, 6, 64), dtype="float32") = R.squeeze(reshape40, axis=[0])
            lv31: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_2_self_attn_v_cache2, squeeze5, sinfo_args=(R.Object,))
            lv32: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv30, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape41: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv32, R.shape([1, total_seq_len, 6, 64]))
            lv33: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv31, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape42: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv33, R.shape([1, total_seq_len, 6, 64]))
            permute_dims88: R.Tensor((1, 6, 1, 64), dtype="float32") = R.permute_dims(reshape38, axes=[0, 2, 1, 3])
            permute_dims89: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape41, axes=[0, 2, 1, 3])
            permute_dims90: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape42, axes=[0, 2, 1, 3])
            permute_dims91: R.Tensor((1, 6, 64, total_seq_len), dtype="float32") = R.permute_dims(permute_dims89, axes=[0, 1, 3, 2])
            matmul63: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.matmul(permute_dims88, permute_dims91, out_dtype="void")
            maximum12: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.maximum(matmul63, R.const(-3.4028234663852886e+38, "float32"))
            minimum12: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.minimum(maximum12, R.const(3.4028234663852886e+38, "float32"))
            softmax8: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.nn.softmax(minimum12, axis=-1)
            matmul64: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax8, permute_dims90, out_dtype="void")
            permute_dims92: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul64, axes=[0, 2, 1, 3])
            reshape43: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims92, R.shape([1, 1, 384]))
            permute_dims93: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_self_attn_out_proj_weight1, axes=None)
            matmul65: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(reshape43, permute_dims93, out_dtype="void")
            add54: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul65, model_decoder_layers_2_self_attn_out_proj_bias1)
            add55: R.Tensor((1, 1, 384), dtype="float32") = R.add(add51, add54)
            layer_norm16: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add55, model_decoder_layers_2_encoder_attn_layer_norm_weight1, model_decoder_layers_2_encoder_attn_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims94: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_encoder_attn_q_proj_weight1, axes=None)
            matmul66: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm16, permute_dims94, out_dtype="void")
            add56: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul66, model_decoder_layers_2_encoder_attn_q_proj_bias1)
            mul9: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add56, R.const(0.125, "float32"))
            reshape44: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul9, R.shape([1, 1, 6, 64]))
            permute_dims95: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_encoder_attn_k_proj_weight1, axes=None)
            matmul67: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(encoder_hidden_states, permute_dims95, out_dtype="void")
            reshape45: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(matmul67, R.shape([1, 1500, 6, 64]))
            permute_dims96: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_encoder_attn_v_proj_weight1, axes=None)
            matmul68: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(encoder_hidden_states, permute_dims96, out_dtype="void")
            add57: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul68, model_decoder_layers_2_encoder_attn_v_proj_bias1)
            reshape46: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add57, R.shape([1, 1500, 6, 64]))
            permute_dims97: R.Tensor((1, 6, 1, 64), dtype="float32") = R.permute_dims(reshape44, axes=[0, 2, 1, 3])
            permute_dims98: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape45, axes=[0, 2, 1, 3])
            permute_dims99: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape46, axes=[0, 2, 1, 3])
            permute_dims100: R.Tensor((1, 6, 64, 1500), dtype="float32") = R.permute_dims(permute_dims98, axes=[0, 1, 3, 2])
            matmul69: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.matmul(permute_dims97, permute_dims100, out_dtype="void")
            maximum13: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.maximum(matmul69, R.const(-3.4028234663852886e+38, "float32"))
            minimum13: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.minimum(maximum13, R.const(3.4028234663852886e+38, "float32"))
            softmax9: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.nn.softmax(minimum13, axis=-1)
            matmul70: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax9, permute_dims99, out_dtype="void")
            permute_dims101: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul70, axes=[0, 2, 1, 3])
            reshape47: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims101, R.shape([1, 1, 384]))
            permute_dims102: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_encoder_attn_out_proj_weight1, axes=None)
            matmul71: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(reshape47, permute_dims102, out_dtype="void")
            add58: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul71, model_decoder_layers_2_encoder_attn_out_proj_bias1)
            add59: R.Tensor((1, 1, 384), dtype="float32") = R.add(add55, add58)
            layer_norm17: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add59, model_decoder_layers_2_final_layer_norm_weight1, model_decoder_layers_2_final_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims103: R.Tensor((384, 1536), dtype="float32") = R.permute_dims(model_decoder_layers_2_fc1_weight1, axes=None)
            matmul72: R.Tensor((1, 1, 1536), dtype="float32") = R.matmul(layer_norm17, permute_dims103, out_dtype="void")
            add60: R.Tensor((1, 1, 1536), dtype="float32") = R.add(matmul72, model_decoder_layers_2_fc1_bias1)
            gelu8: R.Tensor((1, 1, 1536), dtype="float32") = R.nn.gelu(add60)
            permute_dims104: R.Tensor((1536, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_fc2_weight1, axes=None)
            matmul73: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(gelu8, permute_dims104, out_dtype="void")
            add61: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul73, model_decoder_layers_2_fc2_bias1)
            add62: R.Tensor((1, 1, 384), dtype="float32") = R.add(add59, add61)
            layer_norm18: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add62, model_decoder_layers_3_self_attn_layer_norm_weight1, model_decoder_layers_3_self_attn_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims105: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_self_attn_q_proj_weight1, axes=None)
            matmul74: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm18, permute_dims105, out_dtype="void")
            add63: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul74, model_decoder_layers_3_self_attn_q_proj_bias1)
            mul10: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add63, R.const(0.125, "float32"))
            reshape48: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul10, R.shape([1, 1, 6, 64]))
            permute_dims106: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_self_attn_k_proj_weight1, axes=None)
            matmul75: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm18, permute_dims106, out_dtype="void")
            reshape49: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(matmul75, R.shape([1, 1, 6, 64]))
            permute_dims107: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_self_attn_v_proj_weight1, axes=None)
            matmul76: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm18, permute_dims107, out_dtype="void")
            add64: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul76, model_decoder_layers_3_self_attn_v_proj_bias1)
            reshape50: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(add64, R.shape([1, 1, 6, 64]))
            squeeze6: R.Tensor((1, 6, 64), dtype="float32") = R.squeeze(reshape49, axis=[0])
            lv34: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_3_self_attn_k_cache2, squeeze6, sinfo_args=(R.Object,))
            squeeze7: R.Tensor((1, 6, 64), dtype="float32") = R.squeeze(reshape50, axis=[0])
            lv35: R.Object = R.call_packed("vm.builtin.attention_kv_cache_append", model_decoder_layers_3_self_attn_v_cache2, squeeze7, sinfo_args=(R.Object,))
            lv36: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv34, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape51: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv36, R.shape([1, total_seq_len, 6, 64]))
            lv37: R.Tensor((total_seq_len, 6, 64), dtype="float32") = R.call_packed("vm.builtin.attention_kv_cache_view", lv35, R.shape([total_seq_len, 6, 64]), sinfo_args=(R.Tensor((total_seq_len, 6, 64), dtype="float32"),))
            reshape52: R.Tensor((1, total_seq_len, 6, 64), dtype="float32") = R.reshape(lv37, R.shape([1, total_seq_len, 6, 64]))
            permute_dims108: R.Tensor((1, 6, 1, 64), dtype="float32") = R.permute_dims(reshape48, axes=[0, 2, 1, 3])
            permute_dims109: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape51, axes=[0, 2, 1, 3])
            permute_dims110: R.Tensor((1, 6, total_seq_len, 64), dtype="float32") = R.permute_dims(reshape52, axes=[0, 2, 1, 3])
            permute_dims111: R.Tensor((1, 6, 64, total_seq_len), dtype="float32") = R.permute_dims(permute_dims109, axes=[0, 1, 3, 2])
            matmul77: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.matmul(permute_dims108, permute_dims111, out_dtype="void")
            maximum14: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.maximum(matmul77, R.const(-3.4028234663852886e+38, "float32"))
            minimum14: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.minimum(maximum14, R.const(3.4028234663852886e+38, "float32"))
            softmax10: R.Tensor((1, 6, 1, total_seq_len), dtype="float32") = R.nn.softmax(minimum14, axis=-1)
            matmul78: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax10, permute_dims110, out_dtype="void")
            permute_dims112: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul78, axes=[0, 2, 1, 3])
            reshape53: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims112, R.shape([1, 1, 384]))
            permute_dims113: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_self_attn_out_proj_weight1, axes=None)
            matmul79: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(reshape53, permute_dims113, out_dtype="void")
            add65: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul79, model_decoder_layers_3_self_attn_out_proj_bias1)
            add66: R.Tensor((1, 1, 384), dtype="float32") = R.add(add62, add65)
            layer_norm19: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add66, model_decoder_layers_3_encoder_attn_layer_norm_weight1, model_decoder_layers_3_encoder_attn_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims114: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_encoder_attn_q_proj_weight1, axes=None)
            matmul80: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(layer_norm19, permute_dims114, out_dtype="void")
            add67: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul80, model_decoder_layers_3_encoder_attn_q_proj_bias1)
            mul11: R.Tensor((1, 1, 384), dtype="float32") = R.multiply(add67, R.const(0.125, "float32"))
            reshape54: R.Tensor((1, 1, 6, 64), dtype="float32") = R.reshape(mul11, R.shape([1, 1, 6, 64]))
            permute_dims115: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_encoder_attn_k_proj_weight1, axes=None)
            matmul81: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(encoder_hidden_states, permute_dims115, out_dtype="void")
            reshape55: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(matmul81, R.shape([1, 1500, 6, 64]))
            permute_dims116: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_encoder_attn_v_proj_weight1, axes=None)
            matmul82: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(encoder_hidden_states, permute_dims116, out_dtype="void")
            add68: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul82, model_decoder_layers_3_encoder_attn_v_proj_bias1)
            reshape56: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add68, R.shape([1, 1500, 6, 64]))
            permute_dims117: R.Tensor((1, 6, 1, 64), dtype="float32") = R.permute_dims(reshape54, axes=[0, 2, 1, 3])
            permute_dims118: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape55, axes=[0, 2, 1, 3])
            permute_dims119: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape56, axes=[0, 2, 1, 3])
            permute_dims120: R.Tensor((1, 6, 64, 1500), dtype="float32") = R.permute_dims(permute_dims118, axes=[0, 1, 3, 2])
            matmul83: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.matmul(permute_dims117, permute_dims120, out_dtype="void")
            maximum15: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.maximum(matmul83, R.const(-3.4028234663852886e+38, "float32"))
            minimum15: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.minimum(maximum15, R.const(3.4028234663852886e+38, "float32"))
            softmax11: R.Tensor((1, 6, 1, 1500), dtype="float32") = R.nn.softmax(minimum15, axis=-1)
            matmul84: R.Tensor((1, 6, 1, 64), dtype="float32") = R.matmul(softmax11, permute_dims119, out_dtype="void")
            permute_dims121: R.Tensor((1, 1, 6, 64), dtype="float32") = R.permute_dims(matmul84, axes=[0, 2, 1, 3])
            reshape57: R.Tensor((1, 1, 384), dtype="float32") = R.reshape(permute_dims121, R.shape([1, 1, 384]))
            permute_dims122: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_encoder_attn_out_proj_weight1, axes=None)
            matmul85: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(reshape57, permute_dims122, out_dtype="void")
            add69: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul85, model_decoder_layers_3_encoder_attn_out_proj_bias1)
            add70: R.Tensor((1, 1, 384), dtype="float32") = R.add(add66, add69)
            layer_norm20: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add70, model_decoder_layers_3_final_layer_norm_weight1, model_decoder_layers_3_final_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims123: R.Tensor((384, 1536), dtype="float32") = R.permute_dims(model_decoder_layers_3_fc1_weight1, axes=None)
            matmul86: R.Tensor((1, 1, 1536), dtype="float32") = R.matmul(layer_norm20, permute_dims123, out_dtype="void")
            add71: R.Tensor((1, 1, 1536), dtype="float32") = R.add(matmul86, model_decoder_layers_3_fc1_bias1)
            gelu9: R.Tensor((1, 1, 1536), dtype="float32") = R.nn.gelu(add71)
            permute_dims124: R.Tensor((1536, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_fc2_weight1, axes=None)
            matmul87: R.Tensor((1, 1, 384), dtype="float32") = R.matmul(gelu9, permute_dims124, out_dtype="void")
            add72: R.Tensor((1, 1, 384), dtype="float32") = R.add(matmul87, model_decoder_layers_3_fc2_bias1)
            add73: R.Tensor((1, 1, 384), dtype="float32") = R.add(add70, add72)
            layer_norm21: R.Tensor((1, 1, 384), dtype="float32") = R.nn.layer_norm(add73, model_decoder_layer_norm_weight1, model_decoder_layer_norm_bias1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims125: R.Tensor((384, 51865), dtype="float32") = R.permute_dims(proj_out_weight1, axes=None)
            matmul88: R.Tensor((1, 1, 51865), dtype="float32") = R.matmul(layer_norm21, permute_dims125, out_dtype="void")
            gv2: R.Tuple(R.Tuple(R.Tensor((1, 1, 51865), dtype="float32"), R.Tuple(R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")), R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")), R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")), R.Tuple(R.Tensor((1, 1500, 6, 64), dtype="float32"), R.Tensor((1, 1500, 6, 64), dtype="float32")))), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)) = (matmul88, ((reshape25, reshape26), (reshape35, reshape36), (reshape45, reshape46), (reshape55, reshape56))), (lv22, lv23, model_decoder_layers_0_encoder_attn_k_cache2, model_decoder_layers_0_encoder_attn_v_cache2, lv26, lv27, model_decoder_layers_1_encoder_attn_k_cache2, model_decoder_layers_1_encoder_attn_v_cache2, lv30, lv31, model_decoder_layers_2_encoder_attn_k_cache2, model_decoder_layers_2_encoder_attn_v_cache2, lv34, lv35, model_decoder_layers_3_encoder_attn_k_cache2, model_decoder_layers_3_encoder_attn_v_cache2)
            R.output(gv2)
        return gv2

    @R.function
    def encode(input_ids: R.Tensor((1, 80, 3000), dtype="float32"), packed_effects: R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object), packed_params: R.Tuple(R.Tensor((384, 80, 3), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384, 3), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1500, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((51865, 384), dtype="float32"), R.Tensor((448, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384, 384), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((1536, 384), dtype="float32"), R.Tensor((1536,), dtype="float32"), R.Tensor((384, 1536), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((384,), dtype="float32"), R.Tensor((51865, 384), dtype="float32"))) -> R.Tuple(R.Tensor((1, 1500, 384), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)):
        R.func_attr({"num_input": 2, "tir_var_upper_bound": {"seq_len": 448, "total_seq_len": 448}})
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
            model_decoder_embed_tokens_weight: R.Tensor((51865, 384), dtype="float32") = packed_params[67]
            model_decoder_embed_positions_weight: R.Tensor((448, 384), dtype="float32") = packed_params[68]
            model_decoder_layers_0_self_attn_k_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[69]
            model_decoder_layers_0_self_attn_v_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[70]
            model_decoder_layers_0_self_attn_v_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[71]
            model_decoder_layers_0_self_attn_q_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[72]
            model_decoder_layers_0_self_attn_q_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[73]
            model_decoder_layers_0_self_attn_out_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[74]
            model_decoder_layers_0_self_attn_out_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[75]
            model_decoder_layers_0_self_attn_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[76]
            model_decoder_layers_0_self_attn_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[77]
            model_decoder_layers_0_encoder_attn_k_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[78]
            model_decoder_layers_0_encoder_attn_v_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[79]
            model_decoder_layers_0_encoder_attn_v_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[80]
            model_decoder_layers_0_encoder_attn_q_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[81]
            model_decoder_layers_0_encoder_attn_q_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[82]
            model_decoder_layers_0_encoder_attn_out_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[83]
            model_decoder_layers_0_encoder_attn_out_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[84]
            model_decoder_layers_0_encoder_attn_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[85]
            model_decoder_layers_0_encoder_attn_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[86]
            model_decoder_layers_0_fc1_weight: R.Tensor((1536, 384), dtype="float32") = packed_params[87]
            model_decoder_layers_0_fc1_bias: R.Tensor((1536,), dtype="float32") = packed_params[88]
            model_decoder_layers_0_fc2_weight: R.Tensor((384, 1536), dtype="float32") = packed_params[89]
            model_decoder_layers_0_fc2_bias: R.Tensor((384,), dtype="float32") = packed_params[90]
            model_decoder_layers_0_final_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[91]
            model_decoder_layers_0_final_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[92]
            model_decoder_layers_1_self_attn_k_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[93]
            model_decoder_layers_1_self_attn_v_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[94]
            model_decoder_layers_1_self_attn_v_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[95]
            model_decoder_layers_1_self_attn_q_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[96]
            model_decoder_layers_1_self_attn_q_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[97]
            model_decoder_layers_1_self_attn_out_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[98]
            model_decoder_layers_1_self_attn_out_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[99]
            model_decoder_layers_1_self_attn_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[100]
            model_decoder_layers_1_self_attn_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[101]
            model_decoder_layers_1_encoder_attn_k_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[102]
            model_decoder_layers_1_encoder_attn_v_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[103]
            model_decoder_layers_1_encoder_attn_v_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[104]
            model_decoder_layers_1_encoder_attn_q_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[105]
            model_decoder_layers_1_encoder_attn_q_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[106]
            model_decoder_layers_1_encoder_attn_out_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[107]
            model_decoder_layers_1_encoder_attn_out_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[108]
            model_decoder_layers_1_encoder_attn_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[109]
            model_decoder_layers_1_encoder_attn_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[110]
            model_decoder_layers_1_fc1_weight: R.Tensor((1536, 384), dtype="float32") = packed_params[111]
            model_decoder_layers_1_fc1_bias: R.Tensor((1536,), dtype="float32") = packed_params[112]
            model_decoder_layers_1_fc2_weight: R.Tensor((384, 1536), dtype="float32") = packed_params[113]
            model_decoder_layers_1_fc2_bias: R.Tensor((384,), dtype="float32") = packed_params[114]
            model_decoder_layers_1_final_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[115]
            model_decoder_layers_1_final_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[116]
            model_decoder_layers_2_self_attn_k_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[117]
            model_decoder_layers_2_self_attn_v_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[118]
            model_decoder_layers_2_self_attn_v_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[119]
            model_decoder_layers_2_self_attn_q_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[120]
            model_decoder_layers_2_self_attn_q_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[121]
            model_decoder_layers_2_self_attn_out_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[122]
            model_decoder_layers_2_self_attn_out_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[123]
            model_decoder_layers_2_self_attn_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[124]
            model_decoder_layers_2_self_attn_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[125]
            model_decoder_layers_2_encoder_attn_k_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[126]
            model_decoder_layers_2_encoder_attn_v_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[127]
            model_decoder_layers_2_encoder_attn_v_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[128]
            model_decoder_layers_2_encoder_attn_q_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[129]
            model_decoder_layers_2_encoder_attn_q_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[130]
            model_decoder_layers_2_encoder_attn_out_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[131]
            model_decoder_layers_2_encoder_attn_out_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[132]
            model_decoder_layers_2_encoder_attn_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[133]
            model_decoder_layers_2_encoder_attn_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[134]
            model_decoder_layers_2_fc1_weight: R.Tensor((1536, 384), dtype="float32") = packed_params[135]
            model_decoder_layers_2_fc1_bias: R.Tensor((1536,), dtype="float32") = packed_params[136]
            model_decoder_layers_2_fc2_weight: R.Tensor((384, 1536), dtype="float32") = packed_params[137]
            model_decoder_layers_2_fc2_bias: R.Tensor((384,), dtype="float32") = packed_params[138]
            model_decoder_layers_2_final_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[139]
            model_decoder_layers_2_final_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[140]
            model_decoder_layers_3_self_attn_k_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[141]
            model_decoder_layers_3_self_attn_v_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[142]
            model_decoder_layers_3_self_attn_v_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[143]
            model_decoder_layers_3_self_attn_q_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[144]
            model_decoder_layers_3_self_attn_q_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[145]
            model_decoder_layers_3_self_attn_out_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[146]
            model_decoder_layers_3_self_attn_out_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[147]
            model_decoder_layers_3_self_attn_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[148]
            model_decoder_layers_3_self_attn_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[149]
            model_decoder_layers_3_encoder_attn_k_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[150]
            model_decoder_layers_3_encoder_attn_v_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[151]
            model_decoder_layers_3_encoder_attn_v_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[152]
            model_decoder_layers_3_encoder_attn_q_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[153]
            model_decoder_layers_3_encoder_attn_q_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[154]
            model_decoder_layers_3_encoder_attn_out_proj_weight: R.Tensor((384, 384), dtype="float32") = packed_params[155]
            model_decoder_layers_3_encoder_attn_out_proj_bias: R.Tensor((384,), dtype="float32") = packed_params[156]
            model_decoder_layers_3_encoder_attn_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[157]
            model_decoder_layers_3_encoder_attn_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[158]
            model_decoder_layers_3_fc1_weight: R.Tensor((1536, 384), dtype="float32") = packed_params[159]
            model_decoder_layers_3_fc1_bias: R.Tensor((1536,), dtype="float32") = packed_params[160]
            model_decoder_layers_3_fc2_weight: R.Tensor((384, 1536), dtype="float32") = packed_params[161]
            model_decoder_layers_3_fc2_bias: R.Tensor((384,), dtype="float32") = packed_params[162]
            model_decoder_layers_3_final_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[163]
            model_decoder_layers_3_final_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[164]
            model_decoder_layer_norm_weight: R.Tensor((384,), dtype="float32") = packed_params[165]
            model_decoder_layer_norm_bias: R.Tensor((384,), dtype="float32") = packed_params[166]
            proj_out_weight: R.Tensor((51865, 384), dtype="float32") = packed_params[167]
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
            permute_dims1: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_0_self_attn_q_proj_weight, axes=None)
            matmul: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(layer_norm, permute_dims1, out_dtype="void")
            add1: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul, model_encoder_layers_0_self_attn_q_proj_bias)
            mul: R.Tensor((1, 1500, 384), dtype="float32") = R.multiply(add1, R.const(0.125, "float32"))
            reshape: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(mul, R.shape([1, 1500, 6, 64]))
            permute_dims2: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_0_self_attn_k_proj_weight, axes=None)
            matmul1: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(layer_norm, permute_dims2, out_dtype="void")
            reshape1: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(matmul1, R.shape([1, 1500, 6, 64]))
            permute_dims3: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_0_self_attn_v_proj_weight, axes=None)
            matmul2: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(layer_norm, permute_dims3, out_dtype="void")
            add2: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul2, model_encoder_layers_0_self_attn_v_proj_bias)
            reshape2: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add2, R.shape([1, 1500, 6, 64]))
            permute_dims4: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape, axes=[0, 2, 1, 3])
            permute_dims5: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape1, axes=[0, 2, 1, 3])
            permute_dims6: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape2, axes=[0, 2, 1, 3])
            permute_dims7: R.Tensor((1, 6, 64, 1500), dtype="float32") = R.permute_dims(permute_dims5, axes=[0, 1, 3, 2])
            matmul3: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.matmul(permute_dims4, permute_dims7, out_dtype="void")
            maximum: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.maximum(matmul3, R.const(-3.4028234663852886e+38, "float32"))
            minimum: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.minimum(maximum, R.const(3.4028234663852886e+38, "float32"))
            softmax: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.nn.softmax(minimum, axis=-1)
            matmul4: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.matmul(softmax, permute_dims6, out_dtype="void")
            permute_dims8: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.permute_dims(matmul4, axes=[0, 2, 1, 3])
            reshape3: R.Tensor((1, 1500, 384), dtype="float32") = R.reshape(permute_dims8, R.shape([1, 1500, 384]))
            permute_dims9: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_0_self_attn_out_proj_weight, axes=None)
            matmul5: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(reshape3, permute_dims9, out_dtype="void")
            add3: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul5, model_encoder_layers_0_self_attn_out_proj_bias)
            add4: R.Tensor((1, 1500, 384), dtype="float32") = R.add(add, add3)
            layer_norm1: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(add4, model_encoder_layers_0_final_layer_norm_weight, model_encoder_layers_0_final_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims10: R.Tensor((384, 1536), dtype="float32") = R.permute_dims(model_encoder_layers_0_fc1_weight, axes=None)
            matmul6: R.Tensor((1, 1500, 1536), dtype="float32") = R.matmul(layer_norm1, permute_dims10, out_dtype="void")
            add5: R.Tensor((1, 1500, 1536), dtype="float32") = R.add(matmul6, model_encoder_layers_0_fc1_bias)
            gelu2: R.Tensor((1, 1500, 1536), dtype="float32") = R.nn.gelu(add5)
            permute_dims11: R.Tensor((1536, 384), dtype="float32") = R.permute_dims(model_encoder_layers_0_fc2_weight, axes=None)
            matmul7: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(gelu2, permute_dims11, out_dtype="void")
            add6: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul7, model_encoder_layers_0_fc2_bias)
            add7: R.Tensor((1, 1500, 384), dtype="float32") = R.add(add4, add6)
            maximum1: R.Tensor((1, 1500, 384), dtype="float32") = R.maximum(add7, R.const(-3.4028234663852886e+38, "float32"))
            minimum1: R.Tensor((1, 1500, 384), dtype="float32") = R.minimum(maximum1, add7)
            layer_norm2: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(minimum1, model_encoder_layers_1_self_attn_layer_norm_weight, model_encoder_layers_1_self_attn_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims12: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_1_self_attn_q_proj_weight, axes=None)
            matmul8: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(layer_norm2, permute_dims12, out_dtype="void")
            add8: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul8, model_encoder_layers_1_self_attn_q_proj_bias)
            mul1: R.Tensor((1, 1500, 384), dtype="float32") = R.multiply(add8, R.const(0.125, "float32"))
            reshape4: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(mul1, R.shape([1, 1500, 6, 64]))
            permute_dims13: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_1_self_attn_k_proj_weight, axes=None)
            matmul9: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(layer_norm2, permute_dims13, out_dtype="void")
            reshape5: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(matmul9, R.shape([1, 1500, 6, 64]))
            permute_dims14: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_1_self_attn_v_proj_weight, axes=None)
            matmul10: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(layer_norm2, permute_dims14, out_dtype="void")
            add9: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul10, model_encoder_layers_1_self_attn_v_proj_bias)
            reshape6: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add9, R.shape([1, 1500, 6, 64]))
            permute_dims15: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape4, axes=[0, 2, 1, 3])
            permute_dims16: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape5, axes=[0, 2, 1, 3])
            permute_dims17: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape6, axes=[0, 2, 1, 3])
            permute_dims18: R.Tensor((1, 6, 64, 1500), dtype="float32") = R.permute_dims(permute_dims16, axes=[0, 1, 3, 2])
            matmul11: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.matmul(permute_dims15, permute_dims18, out_dtype="void")
            maximum2: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.maximum(matmul11, R.const(-3.4028234663852886e+38, "float32"))
            minimum2: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.minimum(maximum2, R.const(3.4028234663852886e+38, "float32"))
            softmax1: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.nn.softmax(minimum2, axis=-1)
            matmul12: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.matmul(softmax1, permute_dims17, out_dtype="void")
            permute_dims19: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.permute_dims(matmul12, axes=[0, 2, 1, 3])
            reshape7: R.Tensor((1, 1500, 384), dtype="float32") = R.reshape(permute_dims19, R.shape([1, 1500, 384]))
            permute_dims20: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_1_self_attn_out_proj_weight, axes=None)
            matmul13: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(reshape7, permute_dims20, out_dtype="void")
            add10: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul13, model_encoder_layers_1_self_attn_out_proj_bias)
            add11: R.Tensor((1, 1500, 384), dtype="float32") = R.add(minimum1, add10)
            layer_norm3: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(add11, model_encoder_layers_1_final_layer_norm_weight, model_encoder_layers_1_final_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims21: R.Tensor((384, 1536), dtype="float32") = R.permute_dims(model_encoder_layers_1_fc1_weight, axes=None)
            matmul14: R.Tensor((1, 1500, 1536), dtype="float32") = R.matmul(layer_norm3, permute_dims21, out_dtype="void")
            add12: R.Tensor((1, 1500, 1536), dtype="float32") = R.add(matmul14, model_encoder_layers_1_fc1_bias)
            gelu3: R.Tensor((1, 1500, 1536), dtype="float32") = R.nn.gelu(add12)
            permute_dims22: R.Tensor((1536, 384), dtype="float32") = R.permute_dims(model_encoder_layers_1_fc2_weight, axes=None)
            matmul15: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(gelu3, permute_dims22, out_dtype="void")
            add13: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul15, model_encoder_layers_1_fc2_bias)
            add14: R.Tensor((1, 1500, 384), dtype="float32") = R.add(add11, add13)
            maximum3: R.Tensor((1, 1500, 384), dtype="float32") = R.maximum(add14, R.const(-3.4028234663852886e+38, "float32"))
            minimum3: R.Tensor((1, 1500, 384), dtype="float32") = R.minimum(maximum3, add14)
            layer_norm4: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(minimum3, model_encoder_layers_2_self_attn_layer_norm_weight, model_encoder_layers_2_self_attn_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims23: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_2_self_attn_q_proj_weight, axes=None)
            matmul16: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(layer_norm4, permute_dims23, out_dtype="void")
            add15: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul16, model_encoder_layers_2_self_attn_q_proj_bias)
            mul2: R.Tensor((1, 1500, 384), dtype="float32") = R.multiply(add15, R.const(0.125, "float32"))
            reshape8: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(mul2, R.shape([1, 1500, 6, 64]))
            permute_dims24: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_2_self_attn_k_proj_weight, axes=None)
            matmul17: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(layer_norm4, permute_dims24, out_dtype="void")
            reshape9: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(matmul17, R.shape([1, 1500, 6, 64]))
            permute_dims25: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_2_self_attn_v_proj_weight, axes=None)
            matmul18: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(layer_norm4, permute_dims25, out_dtype="void")
            add16: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul18, model_encoder_layers_2_self_attn_v_proj_bias)
            reshape10: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add16, R.shape([1, 1500, 6, 64]))
            permute_dims26: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape8, axes=[0, 2, 1, 3])
            permute_dims27: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape9, axes=[0, 2, 1, 3])
            permute_dims28: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape10, axes=[0, 2, 1, 3])
            permute_dims29: R.Tensor((1, 6, 64, 1500), dtype="float32") = R.permute_dims(permute_dims27, axes=[0, 1, 3, 2])
            matmul19: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.matmul(permute_dims26, permute_dims29, out_dtype="void")
            maximum4: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.maximum(matmul19, R.const(-3.4028234663852886e+38, "float32"))
            minimum4: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.minimum(maximum4, R.const(3.4028234663852886e+38, "float32"))
            softmax2: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.nn.softmax(minimum4, axis=-1)
            matmul20: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.matmul(softmax2, permute_dims28, out_dtype="void")
            permute_dims30: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.permute_dims(matmul20, axes=[0, 2, 1, 3])
            reshape11: R.Tensor((1, 1500, 384), dtype="float32") = R.reshape(permute_dims30, R.shape([1, 1500, 384]))
            permute_dims31: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_2_self_attn_out_proj_weight, axes=None)
            matmul21: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(reshape11, permute_dims31, out_dtype="void")
            add17: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul21, model_encoder_layers_2_self_attn_out_proj_bias)
            add18: R.Tensor((1, 1500, 384), dtype="float32") = R.add(minimum3, add17)
            layer_norm5: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(add18, model_encoder_layers_2_final_layer_norm_weight, model_encoder_layers_2_final_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims32: R.Tensor((384, 1536), dtype="float32") = R.permute_dims(model_encoder_layers_2_fc1_weight, axes=None)
            matmul22: R.Tensor((1, 1500, 1536), dtype="float32") = R.matmul(layer_norm5, permute_dims32, out_dtype="void")
            add19: R.Tensor((1, 1500, 1536), dtype="float32") = R.add(matmul22, model_encoder_layers_2_fc1_bias)
            gelu4: R.Tensor((1, 1500, 1536), dtype="float32") = R.nn.gelu(add19)
            permute_dims33: R.Tensor((1536, 384), dtype="float32") = R.permute_dims(model_encoder_layers_2_fc2_weight, axes=None)
            matmul23: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(gelu4, permute_dims33, out_dtype="void")
            add20: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul23, model_encoder_layers_2_fc2_bias)
            add21: R.Tensor((1, 1500, 384), dtype="float32") = R.add(add18, add20)
            maximum5: R.Tensor((1, 1500, 384), dtype="float32") = R.maximum(add21, R.const(-3.4028234663852886e+38, "float32"))
            minimum5: R.Tensor((1, 1500, 384), dtype="float32") = R.minimum(maximum5, add21)
            layer_norm6: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(minimum5, model_encoder_layers_3_self_attn_layer_norm_weight, model_encoder_layers_3_self_attn_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims34: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_3_self_attn_q_proj_weight, axes=None)
            matmul24: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(layer_norm6, permute_dims34, out_dtype="void")
            add22: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul24, model_encoder_layers_3_self_attn_q_proj_bias)
            mul3: R.Tensor((1, 1500, 384), dtype="float32") = R.multiply(add22, R.const(0.125, "float32"))
            reshape12: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(mul3, R.shape([1, 1500, 6, 64]))
            permute_dims35: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_3_self_attn_k_proj_weight, axes=None)
            matmul25: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(layer_norm6, permute_dims35, out_dtype="void")
            reshape13: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(matmul25, R.shape([1, 1500, 6, 64]))
            permute_dims36: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_3_self_attn_v_proj_weight, axes=None)
            matmul26: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(layer_norm6, permute_dims36, out_dtype="void")
            add23: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul26, model_encoder_layers_3_self_attn_v_proj_bias)
            reshape14: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.reshape(add23, R.shape([1, 1500, 6, 64]))
            permute_dims37: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape12, axes=[0, 2, 1, 3])
            permute_dims38: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape13, axes=[0, 2, 1, 3])
            permute_dims39: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(reshape14, axes=[0, 2, 1, 3])
            permute_dims40: R.Tensor((1, 6, 64, 1500), dtype="float32") = R.permute_dims(permute_dims38, axes=[0, 1, 3, 2])
            matmul27: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.matmul(permute_dims37, permute_dims40, out_dtype="void")
            maximum6: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.maximum(matmul27, R.const(-3.4028234663852886e+38, "float32"))
            minimum6: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.minimum(maximum6, R.const(3.4028234663852886e+38, "float32"))
            softmax3: R.Tensor((1, 6, 1500, 1500), dtype="float32") = R.nn.softmax(minimum6, axis=-1)
            matmul28: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.matmul(softmax3, permute_dims39, out_dtype="void")
            permute_dims41: R.Tensor((1, 1500, 6, 64), dtype="float32") = R.permute_dims(matmul28, axes=[0, 2, 1, 3])
            reshape15: R.Tensor((1, 1500, 384), dtype="float32") = R.reshape(permute_dims41, R.shape([1, 1500, 384]))
            permute_dims42: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_encoder_layers_3_self_attn_out_proj_weight, axes=None)
            matmul29: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(reshape15, permute_dims42, out_dtype="void")
            add24: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul29, model_encoder_layers_3_self_attn_out_proj_bias)
            add25: R.Tensor((1, 1500, 384), dtype="float32") = R.add(minimum5, add24)
            layer_norm7: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(add25, model_encoder_layers_3_final_layer_norm_weight, model_encoder_layers_3_final_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims43: R.Tensor((384, 1536), dtype="float32") = R.permute_dims(model_encoder_layers_3_fc1_weight, axes=None)
            matmul30: R.Tensor((1, 1500, 1536), dtype="float32") = R.matmul(layer_norm7, permute_dims43, out_dtype="void")
            add26: R.Tensor((1, 1500, 1536), dtype="float32") = R.add(matmul30, model_encoder_layers_3_fc1_bias)
            gelu5: R.Tensor((1, 1500, 1536), dtype="float32") = R.nn.gelu(add26)
            permute_dims44: R.Tensor((1536, 384), dtype="float32") = R.permute_dims(model_encoder_layers_3_fc2_weight, axes=None)
            matmul31: R.Tensor((1, 1500, 384), dtype="float32") = R.matmul(gelu5, permute_dims44, out_dtype="void")
            add27: R.Tensor((1, 1500, 384), dtype="float32") = R.add(matmul31, model_encoder_layers_3_fc2_bias)
            add28: R.Tensor((1, 1500, 384), dtype="float32") = R.add(add25, add27)
            maximum7: R.Tensor((1, 1500, 384), dtype="float32") = R.maximum(add28, R.const(-3.4028234663852886e+38, "float32"))
            minimum7: R.Tensor((1, 1500, 384), dtype="float32") = R.minimum(maximum7, add28)
            layer_norm8: R.Tensor((1, 1500, 384), dtype="float32") = R.nn.layer_norm(minimum7, model_encoder_layer_norm_weight, model_encoder_layer_norm_bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
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
            model_encoder_conv1_weight2: R.Tensor((384, 80, 3), dtype="float32") = packed_params[0]
            model_encoder_conv1_bias2: R.Tensor((384,), dtype="float32") = packed_params[1]
            model_encoder_conv2_weight2: R.Tensor((384, 384, 3), dtype="float32") = packed_params[2]
            model_encoder_conv2_bias2: R.Tensor((384,), dtype="float32") = packed_params[3]
            model_encoder_embed_positions_weight2: R.Tensor((1500, 384), dtype="float32") = packed_params[4]
            model_encoder_layers_0_self_attn_k_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[5]
            model_encoder_layers_0_self_attn_v_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[6]
            model_encoder_layers_0_self_attn_v_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[7]
            model_encoder_layers_0_self_attn_q_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[8]
            model_encoder_layers_0_self_attn_q_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[9]
            model_encoder_layers_0_self_attn_out_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[10]
            model_encoder_layers_0_self_attn_out_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[11]
            model_encoder_layers_0_self_attn_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[12]
            model_encoder_layers_0_self_attn_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[13]
            model_encoder_layers_0_fc1_weight2: R.Tensor((1536, 384), dtype="float32") = packed_params[14]
            model_encoder_layers_0_fc1_bias2: R.Tensor((1536,), dtype="float32") = packed_params[15]
            model_encoder_layers_0_fc2_weight2: R.Tensor((384, 1536), dtype="float32") = packed_params[16]
            model_encoder_layers_0_fc2_bias2: R.Tensor((384,), dtype="float32") = packed_params[17]
            model_encoder_layers_0_final_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[18]
            model_encoder_layers_0_final_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[19]
            model_encoder_layers_1_self_attn_k_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[20]
            model_encoder_layers_1_self_attn_v_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[21]
            model_encoder_layers_1_self_attn_v_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[22]
            model_encoder_layers_1_self_attn_q_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[23]
            model_encoder_layers_1_self_attn_q_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[24]
            model_encoder_layers_1_self_attn_out_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[25]
            model_encoder_layers_1_self_attn_out_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[26]
            model_encoder_layers_1_self_attn_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[27]
            model_encoder_layers_1_self_attn_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[28]
            model_encoder_layers_1_fc1_weight2: R.Tensor((1536, 384), dtype="float32") = packed_params[29]
            model_encoder_layers_1_fc1_bias2: R.Tensor((1536,), dtype="float32") = packed_params[30]
            model_encoder_layers_1_fc2_weight2: R.Tensor((384, 1536), dtype="float32") = packed_params[31]
            model_encoder_layers_1_fc2_bias2: R.Tensor((384,), dtype="float32") = packed_params[32]
            model_encoder_layers_1_final_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[33]
            model_encoder_layers_1_final_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[34]
            model_encoder_layers_2_self_attn_k_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[35]
            model_encoder_layers_2_self_attn_v_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[36]
            model_encoder_layers_2_self_attn_v_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[37]
            model_encoder_layers_2_self_attn_q_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[38]
            model_encoder_layers_2_self_attn_q_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[39]
            model_encoder_layers_2_self_attn_out_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[40]
            model_encoder_layers_2_self_attn_out_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[41]
            model_encoder_layers_2_self_attn_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[42]
            model_encoder_layers_2_self_attn_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[43]
            model_encoder_layers_2_fc1_weight2: R.Tensor((1536, 384), dtype="float32") = packed_params[44]
            model_encoder_layers_2_fc1_bias2: R.Tensor((1536,), dtype="float32") = packed_params[45]
            model_encoder_layers_2_fc2_weight2: R.Tensor((384, 1536), dtype="float32") = packed_params[46]
            model_encoder_layers_2_fc2_bias2: R.Tensor((384,), dtype="float32") = packed_params[47]
            model_encoder_layers_2_final_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[48]
            model_encoder_layers_2_final_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[49]
            model_encoder_layers_3_self_attn_k_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[50]
            model_encoder_layers_3_self_attn_v_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[51]
            model_encoder_layers_3_self_attn_v_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[52]
            model_encoder_layers_3_self_attn_q_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[53]
            model_encoder_layers_3_self_attn_q_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[54]
            model_encoder_layers_3_self_attn_out_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[55]
            model_encoder_layers_3_self_attn_out_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[56]
            model_encoder_layers_3_self_attn_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[57]
            model_encoder_layers_3_self_attn_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[58]
            model_encoder_layers_3_fc1_weight2: R.Tensor((1536, 384), dtype="float32") = packed_params[59]
            model_encoder_layers_3_fc1_bias2: R.Tensor((1536,), dtype="float32") = packed_params[60]
            model_encoder_layers_3_fc2_weight2: R.Tensor((384, 1536), dtype="float32") = packed_params[61]
            model_encoder_layers_3_fc2_bias2: R.Tensor((384,), dtype="float32") = packed_params[62]
            model_encoder_layers_3_final_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[63]
            model_encoder_layers_3_final_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[64]
            model_encoder_layer_norm_weight2: R.Tensor((384,), dtype="float32") = packed_params[65]
            model_encoder_layer_norm_bias2: R.Tensor((384,), dtype="float32") = packed_params[66]
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
            model_decoder_layers_0_encoder_attn_k_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[78]
            model_decoder_layers_0_encoder_attn_v_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[79]
            model_decoder_layers_0_encoder_attn_v_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[80]
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
            model_decoder_layers_1_encoder_attn_k_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[102]
            model_decoder_layers_1_encoder_attn_v_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[103]
            model_decoder_layers_1_encoder_attn_v_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[104]
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
            model_decoder_layers_2_encoder_attn_k_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[126]
            model_decoder_layers_2_encoder_attn_v_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[127]
            model_decoder_layers_2_encoder_attn_v_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[128]
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
            model_decoder_layers_3_encoder_attn_k_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[150]
            model_decoder_layers_3_encoder_attn_v_proj_weight2: R.Tensor((384, 384), dtype="float32") = packed_params[151]
            model_decoder_layers_3_encoder_attn_v_proj_bias2: R.Tensor((384,), dtype="float32") = packed_params[152]
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
            permute_dims126: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_self_attn_q_proj_weight2, axes=None)
            matmul89: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm22, permute_dims126, out_dtype="void")
            add75: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul89, model_decoder_layers_0_self_attn_q_proj_bias2)
            mul12: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add75, R.const(0.125, "float32"))
            reshape60: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul12, R.shape([1, seq_len, 6, 64]))
            permute_dims127: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_self_attn_k_proj_weight2, axes=None)
            matmul90: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm22, permute_dims127, out_dtype="void")
            reshape61: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(matmul90, R.shape([1, seq_len, 6, 64]))
            permute_dims128: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_self_attn_v_proj_weight2, axes=None)
            matmul91: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm22, permute_dims128, out_dtype="void")
            add76: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul91, model_decoder_layers_0_self_attn_v_proj_bias2)
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
            permute_dims132: R.Tensor((1, 6, 64, total_seq_len), dtype="float32") = R.permute_dims(permute_dims130, axes=[0, 1, 3, 2])
            matmul92: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.matmul(permute_dims129, permute_dims132, out_dtype="void")
            maximum16: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.maximum(matmul92, R.const(-3.4028234663852886e+38, "float32"))
            minimum16: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.minimum(maximum16, R.const(3.4028234663852886e+38, "float32"))
            softmax12: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.nn.softmax(minimum16, axis=-1)
            matmul93: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax12, permute_dims131, out_dtype="void")
            permute_dims133: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul93, axes=[0, 2, 1, 3])
            reshape65: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims133, R.shape([1, seq_len, 384]))
            permute_dims134: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_self_attn_out_proj_weight2, axes=None)
            matmul94: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(reshape65, permute_dims134, out_dtype="void")
            add77: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul94, model_decoder_layers_0_self_attn_out_proj_bias2)
            add78: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add74, add77)
            layer_norm23: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add78, model_decoder_layers_0_encoder_attn_layer_norm_weight2, model_decoder_layers_0_encoder_attn_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims135: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_encoder_attn_q_proj_weight2, axes=None)
            matmul95: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm23, permute_dims135, out_dtype="void")
            add79: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul95, model_decoder_layers_0_encoder_attn_q_proj_bias2)
            mul13: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add79, R.const(0.125, "float32"))
            reshape66: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul13, R.shape([1, seq_len, 6, 64]))
            permute_dims136: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.permute_dims(reshape66, axes=[0, 2, 1, 3])
            permute_dims137: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_0_0, axes=[0, 2, 1, 3])
            permute_dims138: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_0_1, axes=[0, 2, 1, 3])
            permute_dims139: R.Tensor((1, 6, 64, 1500), dtype="float32") = R.permute_dims(permute_dims137, axes=[0, 1, 3, 2])
            matmul96: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.matmul(permute_dims136, permute_dims139, out_dtype="void")
            maximum17: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.maximum(matmul96, R.const(-3.4028234663852886e+38, "float32"))
            minimum17: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.minimum(maximum17, R.const(3.4028234663852886e+38, "float32"))
            softmax13: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.nn.softmax(minimum17, axis=-1)
            matmul97: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax13, permute_dims138, out_dtype="void")
            permute_dims140: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul97, axes=[0, 2, 1, 3])
            reshape67: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims140, R.shape([1, seq_len, 384]))
            permute_dims141: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_encoder_attn_out_proj_weight2, axes=None)
            matmul98: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(reshape67, permute_dims141, out_dtype="void")
            add80: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul98, model_decoder_layers_0_encoder_attn_out_proj_bias2)
            add81: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add78, add80)
            layer_norm24: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add81, model_decoder_layers_0_final_layer_norm_weight2, model_decoder_layers_0_final_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims142: R.Tensor((384, 1536), dtype="float32") = R.permute_dims(model_decoder_layers_0_fc1_weight2, axes=None)
            matmul99: R.Tensor((1, seq_len, 1536), dtype="float32") = R.matmul(layer_norm24, permute_dims142, out_dtype="void")
            add82: R.Tensor((1, seq_len, 1536), dtype="float32") = R.add(matmul99, model_decoder_layers_0_fc1_bias2)
            gelu10: R.Tensor((1, seq_len, 1536), dtype="float32") = R.nn.gelu(add82)
            permute_dims143: R.Tensor((1536, 384), dtype="float32") = R.permute_dims(model_decoder_layers_0_fc2_weight2, axes=None)
            matmul100: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(gelu10, permute_dims143, out_dtype="void")
            add83: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul100, model_decoder_layers_0_fc2_bias2)
            add84: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add81, add83)
            layer_norm25: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add84, model_decoder_layers_1_self_attn_layer_norm_weight2, model_decoder_layers_1_self_attn_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims144: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_self_attn_q_proj_weight2, axes=None)
            matmul101: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm25, permute_dims144, out_dtype="void")
            add85: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul101, model_decoder_layers_1_self_attn_q_proj_bias2)
            mul14: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add85, R.const(0.125, "float32"))
            reshape68: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul14, R.shape([1, seq_len, 6, 64]))
            permute_dims145: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_self_attn_k_proj_weight2, axes=None)
            matmul102: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm25, permute_dims145, out_dtype="void")
            reshape69: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(matmul102, R.shape([1, seq_len, 6, 64]))
            permute_dims146: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_self_attn_v_proj_weight2, axes=None)
            matmul103: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm25, permute_dims146, out_dtype="void")
            add86: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul103, model_decoder_layers_1_self_attn_v_proj_bias2)
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
            permute_dims150: R.Tensor((1, 6, 64, total_seq_len), dtype="float32") = R.permute_dims(permute_dims148, axes=[0, 1, 3, 2])
            matmul104: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.matmul(permute_dims147, permute_dims150, out_dtype="void")
            maximum18: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.maximum(matmul104, R.const(-3.4028234663852886e+38, "float32"))
            minimum18: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.minimum(maximum18, R.const(3.4028234663852886e+38, "float32"))
            softmax14: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.nn.softmax(minimum18, axis=-1)
            matmul105: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax14, permute_dims149, out_dtype="void")
            permute_dims151: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul105, axes=[0, 2, 1, 3])
            reshape73: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims151, R.shape([1, seq_len, 384]))
            permute_dims152: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_self_attn_out_proj_weight2, axes=None)
            matmul106: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(reshape73, permute_dims152, out_dtype="void")
            add87: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul106, model_decoder_layers_1_self_attn_out_proj_bias2)
            add88: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add84, add87)
            layer_norm26: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add88, model_decoder_layers_1_encoder_attn_layer_norm_weight2, model_decoder_layers_1_encoder_attn_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims153: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_encoder_attn_q_proj_weight2, axes=None)
            matmul107: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm26, permute_dims153, out_dtype="void")
            add89: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul107, model_decoder_layers_1_encoder_attn_q_proj_bias2)
            mul15: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add89, R.const(0.125, "float32"))
            reshape74: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul15, R.shape([1, seq_len, 6, 64]))
            permute_dims154: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.permute_dims(reshape74, axes=[0, 2, 1, 3])
            permute_dims155: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_1_0, axes=[0, 2, 1, 3])
            permute_dims156: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_1_1, axes=[0, 2, 1, 3])
            permute_dims157: R.Tensor((1, 6, 64, 1500), dtype="float32") = R.permute_dims(permute_dims155, axes=[0, 1, 3, 2])
            matmul108: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.matmul(permute_dims154, permute_dims157, out_dtype="void")
            maximum19: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.maximum(matmul108, R.const(-3.4028234663852886e+38, "float32"))
            minimum19: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.minimum(maximum19, R.const(3.4028234663852886e+38, "float32"))
            softmax15: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.nn.softmax(minimum19, axis=-1)
            matmul109: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax15, permute_dims156, out_dtype="void")
            permute_dims158: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul109, axes=[0, 2, 1, 3])
            reshape75: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims158, R.shape([1, seq_len, 384]))
            permute_dims159: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_encoder_attn_out_proj_weight2, axes=None)
            matmul110: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(reshape75, permute_dims159, out_dtype="void")
            add90: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul110, model_decoder_layers_1_encoder_attn_out_proj_bias2)
            add91: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add88, add90)
            layer_norm27: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add91, model_decoder_layers_1_final_layer_norm_weight2, model_decoder_layers_1_final_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims160: R.Tensor((384, 1536), dtype="float32") = R.permute_dims(model_decoder_layers_1_fc1_weight2, axes=None)
            matmul111: R.Tensor((1, seq_len, 1536), dtype="float32") = R.matmul(layer_norm27, permute_dims160, out_dtype="void")
            add92: R.Tensor((1, seq_len, 1536), dtype="float32") = R.add(matmul111, model_decoder_layers_1_fc1_bias2)
            gelu11: R.Tensor((1, seq_len, 1536), dtype="float32") = R.nn.gelu(add92)
            permute_dims161: R.Tensor((1536, 384), dtype="float32") = R.permute_dims(model_decoder_layers_1_fc2_weight2, axes=None)
            matmul112: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(gelu11, permute_dims161, out_dtype="void")
            add93: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul112, model_decoder_layers_1_fc2_bias2)
            add94: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add91, add93)
            layer_norm28: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add94, model_decoder_layers_2_self_attn_layer_norm_weight2, model_decoder_layers_2_self_attn_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims162: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_self_attn_q_proj_weight2, axes=None)
            matmul113: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm28, permute_dims162, out_dtype="void")
            add95: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul113, model_decoder_layers_2_self_attn_q_proj_bias2)
            mul16: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add95, R.const(0.125, "float32"))
            reshape76: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul16, R.shape([1, seq_len, 6, 64]))
            permute_dims163: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_self_attn_k_proj_weight2, axes=None)
            matmul114: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm28, permute_dims163, out_dtype="void")
            reshape77: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(matmul114, R.shape([1, seq_len, 6, 64]))
            permute_dims164: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_self_attn_v_proj_weight2, axes=None)
            matmul115: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm28, permute_dims164, out_dtype="void")
            add96: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul115, model_decoder_layers_2_self_attn_v_proj_bias2)
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
            permute_dims168: R.Tensor((1, 6, 64, total_seq_len), dtype="float32") = R.permute_dims(permute_dims166, axes=[0, 1, 3, 2])
            matmul116: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.matmul(permute_dims165, permute_dims168, out_dtype="void")
            maximum20: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.maximum(matmul116, R.const(-3.4028234663852886e+38, "float32"))
            minimum20: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.minimum(maximum20, R.const(3.4028234663852886e+38, "float32"))
            softmax16: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.nn.softmax(minimum20, axis=-1)
            matmul117: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax16, permute_dims167, out_dtype="void")
            permute_dims169: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul117, axes=[0, 2, 1, 3])
            reshape81: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims169, R.shape([1, seq_len, 384]))
            permute_dims170: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_self_attn_out_proj_weight2, axes=None)
            matmul118: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(reshape81, permute_dims170, out_dtype="void")
            add97: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul118, model_decoder_layers_2_self_attn_out_proj_bias2)
            add98: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add94, add97)
            layer_norm29: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add98, model_decoder_layers_2_encoder_attn_layer_norm_weight2, model_decoder_layers_2_encoder_attn_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims171: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_encoder_attn_q_proj_weight2, axes=None)
            matmul119: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm29, permute_dims171, out_dtype="void")
            add99: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul119, model_decoder_layers_2_encoder_attn_q_proj_bias2)
            mul17: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add99, R.const(0.125, "float32"))
            reshape82: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul17, R.shape([1, seq_len, 6, 64]))
            permute_dims172: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.permute_dims(reshape82, axes=[0, 2, 1, 3])
            permute_dims173: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_2_0, axes=[0, 2, 1, 3])
            permute_dims174: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_2_1, axes=[0, 2, 1, 3])
            permute_dims175: R.Tensor((1, 6, 64, 1500), dtype="float32") = R.permute_dims(permute_dims173, axes=[0, 1, 3, 2])
            matmul120: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.matmul(permute_dims172, permute_dims175, out_dtype="void")
            maximum21: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.maximum(matmul120, R.const(-3.4028234663852886e+38, "float32"))
            minimum21: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.minimum(maximum21, R.const(3.4028234663852886e+38, "float32"))
            softmax17: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.nn.softmax(minimum21, axis=-1)
            matmul121: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax17, permute_dims174, out_dtype="void")
            permute_dims176: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul121, axes=[0, 2, 1, 3])
            reshape83: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims176, R.shape([1, seq_len, 384]))
            permute_dims177: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_encoder_attn_out_proj_weight2, axes=None)
            matmul122: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(reshape83, permute_dims177, out_dtype="void")
            add100: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul122, model_decoder_layers_2_encoder_attn_out_proj_bias2)
            add101: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add98, add100)
            layer_norm30: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add101, model_decoder_layers_2_final_layer_norm_weight2, model_decoder_layers_2_final_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims178: R.Tensor((384, 1536), dtype="float32") = R.permute_dims(model_decoder_layers_2_fc1_weight2, axes=None)
            matmul123: R.Tensor((1, seq_len, 1536), dtype="float32") = R.matmul(layer_norm30, permute_dims178, out_dtype="void")
            add102: R.Tensor((1, seq_len, 1536), dtype="float32") = R.add(matmul123, model_decoder_layers_2_fc1_bias2)
            gelu12: R.Tensor((1, seq_len, 1536), dtype="float32") = R.nn.gelu(add102)
            permute_dims179: R.Tensor((1536, 384), dtype="float32") = R.permute_dims(model_decoder_layers_2_fc2_weight2, axes=None)
            matmul124: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(gelu12, permute_dims179, out_dtype="void")
            add103: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul124, model_decoder_layers_2_fc2_bias2)
            add104: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add101, add103)
            layer_norm31: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add104, model_decoder_layers_3_self_attn_layer_norm_weight2, model_decoder_layers_3_self_attn_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims180: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_self_attn_q_proj_weight2, axes=None)
            matmul125: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm31, permute_dims180, out_dtype="void")
            add105: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul125, model_decoder_layers_3_self_attn_q_proj_bias2)
            mul18: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add105, R.const(0.125, "float32"))
            reshape84: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul18, R.shape([1, seq_len, 6, 64]))
            permute_dims181: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_self_attn_k_proj_weight2, axes=None)
            matmul126: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm31, permute_dims181, out_dtype="void")
            reshape85: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(matmul126, R.shape([1, seq_len, 6, 64]))
            permute_dims182: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_self_attn_v_proj_weight2, axes=None)
            matmul127: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm31, permute_dims182, out_dtype="void")
            add106: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul127, model_decoder_layers_3_self_attn_v_proj_bias2)
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
            permute_dims186: R.Tensor((1, 6, 64, total_seq_len), dtype="float32") = R.permute_dims(permute_dims184, axes=[0, 1, 3, 2])
            matmul128: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.matmul(permute_dims183, permute_dims186, out_dtype="void")
            maximum22: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.maximum(matmul128, R.const(-3.4028234663852886e+38, "float32"))
            minimum22: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.minimum(maximum22, R.const(3.4028234663852886e+38, "float32"))
            softmax18: R.Tensor((1, 6, seq_len, total_seq_len), dtype="float32") = R.nn.softmax(minimum22, axis=-1)
            matmul129: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax18, permute_dims185, out_dtype="void")
            permute_dims187: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul129, axes=[0, 2, 1, 3])
            reshape89: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims187, R.shape([1, seq_len, 384]))
            permute_dims188: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_self_attn_out_proj_weight2, axes=None)
            matmul130: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(reshape89, permute_dims188, out_dtype="void")
            add107: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul130, model_decoder_layers_3_self_attn_out_proj_bias2)
            add108: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add104, add107)
            layer_norm32: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add108, model_decoder_layers_3_encoder_attn_layer_norm_weight2, model_decoder_layers_3_encoder_attn_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims189: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_encoder_attn_q_proj_weight2, axes=None)
            matmul131: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(layer_norm32, permute_dims189, out_dtype="void")
            add109: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul131, model_decoder_layers_3_encoder_attn_q_proj_bias2)
            mul19: R.Tensor((1, seq_len, 384), dtype="float32") = R.multiply(add109, R.const(0.125, "float32"))
            reshape90: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.reshape(mul19, R.shape([1, seq_len, 6, 64]))
            permute_dims190: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.permute_dims(reshape90, axes=[0, 2, 1, 3])
            permute_dims191: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_3_0, axes=[0, 2, 1, 3])
            permute_dims192: R.Tensor((1, 6, 1500, 64), dtype="float32") = R.permute_dims(cached_encoder_key_value_3_1, axes=[0, 2, 1, 3])
            permute_dims193: R.Tensor((1, 6, 64, 1500), dtype="float32") = R.permute_dims(permute_dims191, axes=[0, 1, 3, 2])
            matmul132: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.matmul(permute_dims190, permute_dims193, out_dtype="void")
            maximum23: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.maximum(matmul132, R.const(-3.4028234663852886e+38, "float32"))
            minimum23: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.minimum(maximum23, R.const(3.4028234663852886e+38, "float32"))
            softmax19: R.Tensor((1, 6, seq_len, 1500), dtype="float32") = R.nn.softmax(minimum23, axis=-1)
            matmul133: R.Tensor((1, 6, seq_len, 64), dtype="float32") = R.matmul(softmax19, permute_dims192, out_dtype="void")
            permute_dims194: R.Tensor((1, seq_len, 6, 64), dtype="float32") = R.permute_dims(matmul133, axes=[0, 2, 1, 3])
            reshape91: R.Tensor((1, seq_len, 384), dtype="float32") = R.reshape(permute_dims194, R.shape([1, seq_len, 384]))
            permute_dims195: R.Tensor((384, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_encoder_attn_out_proj_weight2, axes=None)
            matmul134: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(reshape91, permute_dims195, out_dtype="void")
            add110: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul134, model_decoder_layers_3_encoder_attn_out_proj_bias2)
            add111: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add108, add110)
            layer_norm33: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add111, model_decoder_layers_3_final_layer_norm_weight2, model_decoder_layers_3_final_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims196: R.Tensor((384, 1536), dtype="float32") = R.permute_dims(model_decoder_layers_3_fc1_weight2, axes=None)
            matmul135: R.Tensor((1, seq_len, 1536), dtype="float32") = R.matmul(layer_norm33, permute_dims196, out_dtype="void")
            add112: R.Tensor((1, seq_len, 1536), dtype="float32") = R.add(matmul135, model_decoder_layers_3_fc1_bias2)
            gelu13: R.Tensor((1, seq_len, 1536), dtype="float32") = R.nn.gelu(add112)
            permute_dims197: R.Tensor((1536, 384), dtype="float32") = R.permute_dims(model_decoder_layers_3_fc2_weight2, axes=None)
            matmul136: R.Tensor((1, seq_len, 384), dtype="float32") = R.matmul(gelu13, permute_dims197, out_dtype="void")
            add113: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(matmul136, model_decoder_layers_3_fc2_bias2)
            add114: R.Tensor((1, seq_len, 384), dtype="float32") = R.add(add111, add113)
            layer_norm34: R.Tensor((1, seq_len, 384), dtype="float32") = R.nn.layer_norm(add114, model_decoder_layer_norm_weight2, model_decoder_layer_norm_bias2, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            permute_dims198: R.Tensor((384, 51865), dtype="float32") = R.permute_dims(proj_out_weight2, axes=None)
            matmul137: R.Tensor((1, seq_len, 51865), dtype="float32") = R.matmul(layer_norm34, permute_dims198, out_dtype="void")
            gv3: R.Tuple(R.Tensor((1, seq_len, 51865), dtype="float32"), R.Tuple(R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object, R.Object)) = matmul137, (lv39, lv40, model_decoder_layers_0_encoder_attn_k_cache3, model_decoder_layers_0_encoder_attn_v_cache3, lv43, lv44, model_decoder_layers_1_encoder_attn_k_cache3, model_decoder_layers_1_encoder_attn_v_cache3, lv47, lv48, model_decoder_layers_2_encoder_attn_k_cache3, model_decoder_layers_2_encoder_attn_v_cache3, lv51, lv52, model_decoder_layers_3_encoder_attn_k_cache3, model_decoder_layers_3_encoder_attn_v_cache3)
            R.output(gv3)
        return gv3