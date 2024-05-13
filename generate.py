def generate(model: Any, config: WhisperConfig, input_features) -> torch.Tensor:
    # encode
    print(model)

    encode_output = model["encode"](input_features)

  
    decoder_start_token_id = 0
    if hasattr(config, "decoder_start_token_id"):
        decoder_start_token_id = config.decoder_start_token_id
    elif "decoder_start_token_id" in config.kwargs:
        decoder_start_token_id = config.kwargs["decoder_start_token_id"]
    else:
        raise ValueError("decoder_start_token_id not found")        

    suppress_tokens = []
    if hasattr(config, "suppress_tokens"):
        suppress_tokens = config.suppress_tokens
    elif "suppress_tokens" in config.kwargs:
        suppress_tokens = config.kwargs["suppress_tokens"]
    else:
        raise ValueError("suppress_tokens not found")
    
    forced_decoder_ids = []
    if hasattr(config, "forced_decoder_ids"):
        forced_decoder_ids = config.forced_decoder_ids
    elif "forced_decoder_ids" in config.kwargs:
        forced_decoder_ids = config.kwargs["forced_decoder_ids"]
    else:
        raise ValueError("forced_decoder_ids not found")
    
    begin_suppress_tokens = []
    if hasattr(config, "begin_suppress_tokens"):
        begin_suppress_tokens = config.begin_suppress_tokens
    elif "begin_suppress_tokens" in config.kwargs:
        begin_suppress_tokens = config.kwargs["begin_suppress_tokens"]
    else:
        raise ValueError("begin_suppress_tokens not found")
    
    eos_token_id = []   
    if hasattr(config, "eos_token_id"):
        eos_token_id = config.eos_token_id
    elif "eos_token_id" in config.kwargs:
        eos_token_id = config.kwargs["eos_token_id"]
    else: 
        raise ValueError("eos_token_id not found") 

    
    print("eos_token_id", eos_token_id)
    print("decoder_start_token_id", decoder_start_token_id)
    print("suppress_tokens", suppress_tokens)
    print("begin_suppress_tokens", begin_suppress_tokens)
    print("forced_decoder_ids", forced_decoder_ids)    
    

    # decode start token
    input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.int32).to("cpu")
    generated_tokens = [decoder_start_token_id]
    print("input_ids", input_ids)
    print("generated_tokens", generated_tokens)
    

    while True:
        if len(generated_tokens) == 1:
            outputs, encode_kv_cache = model["decode"](
                input_ids, len(generated_tokens), encode_output)
        else:
            outputs = model["prefill"](input_ids, len(generated_tokens), encode_kv_cache)

        outputs_logits = outputs
        next_token_logits = outputs_logits[:, 0, :]
        func1 = lambda x: stable_softmax(x)
        func2 = lambda x: softmax(x)
        print_top5(func1, {0:outputs_logits.numpy()})
        print_top5(func2, {0:outputs_logits.numpy()})
        
    
        # suppress tokens
        next_tokens_scores = next_token_logits
        print(f"token 50285 {next_tokens_scores[0,50258]}")

        next_tokens_scores[:, suppress_tokens] = -float("inf")
        np.savez('/home/munusairam/Downloads/python_array_next_tokens_scores.npz', next_tokens_scores.numpy())

        print(f"token 50285 {next_tokens_scores[0,50258]}")

    
        # suppress tokens at begin
        if len(generated_tokens) == 1 + forced_decoder_ids[-1][0]:
            print(f"forced_decoder_ids is {forced_decoder_ids[-1][0]}")
            next_tokens_scores[:, begin_suppress_tokens] = -float("inf")

        print(f"token 50285 {next_tokens_scores[0,50258]}")


        # force tokens at sepcific position
        generation_idx = len(generated_tokens)
        current_token = dict(forced_decoder_ids).get(generation_idx, None)
        if current_token is not None:
            next_tokens_scores[:, :] = -float("inf")
            next_tokens_scores[:, current_token] = 0
        
        print(f"token 50285 {next_tokens_scores[0,50258]}")

        
        print_top5(func1, {0:next_tokens_scores.unsqueeze(dim=0).numpy()})
        print_top5(func2, {0:next_tokens_scores.unsqueeze(dim=0).numpy()})

        # argmax
        next_token = torch.argmax(next_tokens_scores, dim=-1)[0]
        input_ids[0][0] = next_token

        print(f"next_token {next_token}")
        print(model["softmax_with_temperature"](outputs_logits, torch.tensor(0.7))[0, 0, next_token])


        

        generated_tokens.append(next_token)

        print(generated_tokens)

        # stop when we meet eos_token_id or exceed the maximum length
        if (
            next_token == eos_token_id
            or len(generated_tokens) == config.max_target_positions
        ):
            break

    return generated_tokens
