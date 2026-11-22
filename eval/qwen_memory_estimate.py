"""GPU memory estimate for Qwen2.5-1.5B 4-bit LoRA at 4,096 tokens on a 15 GB T4 (D-029). Arithmetic from the model's shapes, not a measurement: there is no GPU here, and the
only measurements are the outcomes of the failed Kaggle run (which this estimate must reproduce) and the earlier Colab run (which it must not contradict).

    python3 eval/qwen_memory_estimate.py            # no dependencies; writes nothing, prints the table (saved in results/final_v3b/qwen_memory_estimate.txt)

Model (Qwen2.5-1.5B): 28 layers, hidden h = 1536, 12 query heads, 2 key/value heads (head dim 128, so k and v are 256 wide), MLP width I = 8960, vocabulary 151,936, 1.31e9
parameters in the linear layers of the 28 blocks, 233 M in the embedding. The notebook: bitsandbytes NF4 with double quantization for the linear layers, fp16 autocast
(Trainer fp16=True), `prepare_model_for_kbit_training` (casts everything that is not quantized, i.e. the embedding, the norms and the head, to fp32), LoRA r=16 on q_proj and
v_proj, SDPA attention, a sequence-classification head (the 2-way `score` layer on every token: negligible), micro-batch B of rows padded to the longest row of the batch.
The longest batch comes first in every epoch (LengthBucketSampler), so the peak of the whole run is reached at optimizer step 1: B rows of about 4,096 tokens.

Activation bytes per token and layer (what autograd saves for the backward pass without checkpointing), in units of h = 1536, kv = 256, I = 8960:
    input RMSNorm: fp32 input (4h) + fp32 output (4h) + its fp16 copy for the matmuls (2h)                   = 10 h
    LoRA dropout on the input of q_proj and v_proj: two fp16 copies (2h each) + two bool masks (h each)      =  6 h
    q, k, v outputs (fp16) and the rotated q, k (fp16): 2 (h + 2 kv) + 2 (h + kv)                            = 2 (h + 2 kv) + 2 (h + kv)  bytes
    attention output (fp16, the input of o_proj)                                                              =  2 h
    residual after attention (fp32)                                                                           =  4 h
    post-attention RMSNorm (as the first one)                                                                 = 10 h
    MLP: gate, up, silu(gate), silu(gate) * up, all fp16                                                      =  8 I
  = about 125 KiB per token and layer = 3.5 MB per token over 28 layers: 13.7 GiB for one 4,096-token row (5.1 GiB = 5.5 GB for a 1,536-token row, the figure of D-020).
With gradient checkpointing only the input of every layer is kept (the fp32 residual stream, 4 h) and one layer at a time is recomputed in the backward pass: its full
activations (125 KiB per token) plus the gradients that exist at the same time (about 72 KiB per token).
Attention: SDPA with the memory-efficient kernel (the T4 has no flash kernel; the notebook now asks for "sdpa" explicitly) keeps nothing quadratic. The pessimistic scenario
assumes the math kernel instead: 12 heads x T x T x (2 + 4 + 2) bytes saved (fp16 scores, fp32 softmax, fp16 copy) plus 4 bytes x 12 x T x T of gradient, per row, per layer
that is being recomputed.
"""

H, KV, I, L, V = 1536, 256, 8960, 28, 151936
GIB = 2 ** 30
LINEAR_PARAMS = 1.31e9
EMB_PARAMS = V * H
LORA_PARAMS = 28 * ((H * 16 + 16 * H) + (H * 16 + 16 * KV)) + H * 2   # q_proj: A (h x r) and B (r x h); v_proj: A (h x r) and B (r x kv); plus the 2-way head
CONTEXT_GIB = 0.6      # CUDA context, cuBLAS / cuDNN workspaces and the like: counted by the driver, not by torch.cuda.memory_allocated
DRIVER_TOTAL_GIB = 14.6   # what torch.cuda.mem_get_info() reports for a 16 GB T4 (15,109 MiB)
SLACK = 1.15           # caching-allocator rounding and fragmentation on the activation part


def fixed_gib():
    w4 = LINEAR_PARAMS * (4 + 0.13) / 8 / GIB        # NF4 plus the double-quantization constants
    emb = EMB_PARAMS * 4 / GIB                        # fp32 after prepare_model_for_kbit_training
    lora = LORA_PARAMS * 16 / GIB                     # fp32 weight, gradient, two Adam moments
    dequant = 3 * I * H * 2 / GIB                     # temporary fp16 copies of the largest weight matrices while a layer runs
    return w4, emb, lora, dequant


def per_token_layer_bytes():
    full = 10 * H + 6 * H + 2 * (H + 2 * KV) + 2 * (H + KV) + 2 * H + 4 * H + 10 * H + 8 * I
    grads = 3 * 2 * I + 3 * 4 * H                     # the gradients of the MLP tensors and of the fp32 hidden states that exist while one layer is recomputed
    return full, grads


def estimate(B, T, ckpt, math_attention=False):
    w4, emb, lora, dequant = fixed_gib()
    full, grads = per_token_layer_bytes()
    tokens = B * T
    if ckpt:
        saved = L * tokens * 4 * H                    # fp32 residual stream at the input of every layer
        working = tokens * (full + grads)
        act = saved + working
        attn = B * 12 * T * T * (8 + 4) if math_attention else 0
    else:
        act = L * tokens * full
        attn = L * B * 12 * T * T * 8 if math_attention else 0
    act_gib = (act + attn) / GIB * SLACK
    total = w4 + emb + lora + dequant + CONTEXT_GIB + act_gib
    return total, w4 + emb + lora + dequant, act_gib


def eval_gib(B, T, math_attention=False):
    w4, emb, lora, dequant = fixed_gib()
    work = B * T * (4 * 2 * I + 3 * 4 * H) / GIB      # forward only: gate, up, act, product (fp16) and a few fp32 hidden-state copies of ONE layer at a time
    attn = B * 12 * T * T * (2 + 4 + 2) / GIB if math_attention else 0
    return w4 + emb + lora + dequant + CONTEXT_GIB + (work + attn) * SLACK


def main():
    w4, emb, lora, dequant = fixed_gib()
    full, grads = per_token_layer_bytes()
    print("Qwen2.5-1.5B 4-bit LoRA on a 16 GB T4 (driver-visible total %.1f GiB): an arithmetic estimate, not a measurement" % DRIVER_TOTAL_GIB)
    print(f"fixed: NF4 weights {w4:.2f} GiB + fp32 embedding {emb:.2f} + LoRA and head with optimizer {lora:.3f} + dequantization temporaries {dequant:.2f} + CUDA context {CONTEXT_GIB:.1f} = {w4 + emb + lora + dequant + CONTEXT_GIB:.2f} GiB")
    print(f"activations per token and layer without checkpointing: {full / 1024:.0f} KiB (-> {full * L / 1e6:.2f} MB per token over {L} layers: {full * L * 4096 / GIB:.1f} GiB for a 4,096-token row, {full * L * 1536 / GIB:.1f} GiB for a 1,536-token row)")
    print(f"with checkpointing per token: {4 * H * L / 1024:.0f} KiB saved over all layers (fp32 residual stream) + {(full + grads) / 1024:.0f} KiB for the one layer being recomputed\n")
    print(f"{'micro-batch':>11s} {'checkpointing':>13s} {'tokens':>7s} | {'expected (SDPA)':>16s} {'pessimistic (math attention)':>29s} | verdict against {DRIVER_TOTAL_GIB} GiB")
    rows = [(1, True), (2, True), (4, True), (8, True), (1, False), (2, False)]
    for B, ck in rows:
        e, _, a = estimate(B, 4096, ck)
        p, _, _ = estimate(B, 4096, ck, math_attention=True)
        v = "fits" if p < DRIVER_TOTAL_GIB else ("fits if SDPA is used" if e < DRIVER_TOTAL_GIB else "does NOT fit")
        print(f"{B:11d} {'on' if ck else 'off':>13s} {B * 4096:7d} | {e:13.1f} GiB {p:26.1f} GiB | {v}")
    print("\nchecks against what is known:")
    for label, B, T, ck, observed in (
        ("Kaggle run: micro-batch 2, checkpointing off, 4,096 tokens", 2, 4096, False, "out of memory"),
        ("Kaggle run: micro-batch 1, checkpointing off, 4,096 tokens", 1, 4096, False, "out of memory"),
        ("Kaggle run: micro-batch 8, checkpointing on, 4,096 tokens", 8, 4096, True, "out of memory"),
        ("Kaggle run: micro-batch 4, checkpointing on, 4,096 tokens", 4, 4096, True, "stopped without a message (not reported as out of memory)"),
        ("Colab run (v1 input, random batches, checkpointing on): micro-batch 8, 1,536 tokens", 8, 1536, True, "completed (the calibration run; it may have fallen back to 4 or 2)"),
    ):
        e, _, _ = estimate(B, T, ck)
        p, _, _ = estimate(B, T, ck, math_attention=True)
        print(f"   {label}: expected {e:5.1f} GiB, pessimistic {p:5.1f} GiB; observed: {observed}")
    print("   -> the three out-of-memory errors of the Kaggle run are what the estimate says: micro-batch 1 and 2 without checkpointing are far over the limit (18 and 34 GiB), micro-batch 8 with")
    print("      checkpointing is just over it (15.3 GiB expected). Micro-batch 4 with checkpointing is predicted to FIT with SDPA attention (8.7 GiB); it would be an out-of-memory error only if")
    print("      the attention fell back to the math kernel (19.1 GiB), and that would have been reported as an out-of-memory error, not as a silent stop. So the silent stop at micro-batch 4")
    print("      is probably not an out-of-memory error of that attempt (host RAM, or a state left by the earlier failures, are the candidates).")
    print("\nevaluation (forward only, no checkpointing needed): validation batch 2 and final-evaluation batch 4 at 4,096 tokens")
    for B in (2, 4, 8):
        print(f"   batch {B}: expected {eval_gib(B, 4096):.1f} GiB, pessimistic (math attention) {eval_gib(B, 4096, True):.1f} GiB" + ("   <- the old default of the final evaluation" if B == 8 else ""))
    print("\nsensitivity of micro-batch 2 with checkpointing (the default): activation part x1.0 / x1.5 / x2.0 of the estimate -> " +
          " / ".join(f"{estimate(2, 4096, True)[1] + CONTEXT_GIB + estimate(2, 4096, True)[2] * f:.1f} GiB" for f in (1.0, 1.5, 2.0)) + f" (limit {DRIVER_TOTAL_GIB}): the estimate would have to be low by a factor of {(DRIVER_TOTAL_GIB - estimate(2, 4096, True)[1] - CONTEXT_GIB) / estimate(2, 4096, True)[2]:.1f} for micro-batch 2 to fail")
    print(f"and micro-batch 1: factor {(DRIVER_TOTAL_GIB - estimate(1, 4096, True)[1] - CONTEXT_GIB) / estimate(1, 4096, True)[2]:.1f}")


if __name__ == "__main__":
    main()
