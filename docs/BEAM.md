# Reflection Beam-1.0 on maya-mm3

[Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B) (Reflection AI, Apache-2.0) is a 501B
MoE model: 52 layers (3 dense, then 49 MoE of 256 experts, 8 a token, ~23B active), 1M context.

- **Attention.** 9 global layers attend over the whole context with no RoPE, their queries scaled by position. The other
  43 attend over the last 512 positions with RoPE. Every attention output is gated per head.
- **Norms and residual.** Each layer has four norms (sandwich) and a depth-scaled residual.
- **Router.** Group-limited: the 4 best of 8 groups.
- **Head.** The logits are soft-capped at 30.

It runs on maya-mm3, MiniMax-M3.1's engine. The placement, the expert tiers (VRAM, staging slots, pinned RAM, the
CPU lane, the SSD) and the GGUF expert kernels are shared. Beam's own parts are switched on by the pack's
`model_type beam`:

| Part | Where |
|---|---|
| geometry | `src/mm3/mm3_geo.hpp` `Geo::read_beam` |
| kernels (attention with a window and a ring, gate, NoPE scale, residual, SwiGLU, router, softcap) | `src/mm3/mm3_kernels.cu` (the Beam section at the end) |
| forward: one front for the token, window and prompt paths; the experts by each path's own machinery | `src/mm3/mm3_model.cu` `beam_front` / `beam_layer` / `moe_out` |
| the experts' plain SwiGLU (GPU, CPU lane) | `mm3_gguf.cu` `set_plain_swiglu`, `mm3_cpu_gg.cpp` |
| pack from a GGUF set (unsloth's, bartowski's ...) | `tools/mm3/pack_beam_gguf.py` |
| server config | `tools/mm3/beam_setup.py` |
| tokenizer (`qwen2` pre-tokenizer, NFC) | `tools/strata_tokenizer.py` |
| chat template (harmony_xml) and output parsing | `serve/frontend.py` `BeamRewriter`, `tools/mm3/beam_chat_template.jinja` |
| end tokens `<|return|>` / `<|call|>` | `serve/server.py`, `mm3_main.cpp` (`Model::eos_ids`) |

## KV cache

- **Global layers:** 36 KiB a token in BF16 (9 layers x 8 KV heads x 128 x 2 x 2 bytes), 18 KiB in INT8, the default.
  1M positions take 36 GiB in BF16 or 18 GiB in INT8, spread over the GPUs with their layers.
- **Sliding layers:** a ring of 512 + a prompt chunk's positions each, a few MB.

## Running it

```sh
cmake -S . -B build -G Ninja -DSTRATA_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target maya-mm3 mm3_beam_test
build/mm3_beam_test                                          # Beam's kernels against F64 references
hf download unsloth/Beam-1.0-501B-A23B-GGUF --include 'UD-IQ2_XXS/*' --local-dir ~/models/beam
python tools/mm3/beam_setup.py --model ~/models/beam/UD-IQ2_XXS --context 262144
.venv/bin/python serve/server.py --engine maya --config maya-beam.json
```

## Limits

- GGUF packs only.
- The layer split only: no `--tp`, `--parallel`, DSpark, vision, KV streaming or `--ablate`.
- An expert set whose gate and up are of two quant types is refused. unsloth's UD-Q2_K_XL and UD-IQ2_XXS are fine.
- The thinking budget does not apply: Beam's reasoning ends with its final channel's header, not one token.
- Reflection's DFlash2 drafter is not ported.

## What has been checked, and what has not

**Checked without a GPU:**

- the build;
- the pack of unsloth's real UD-Q2_K_XL and UD-IQ2_XXS headers: every one of the 38,366 tensors mapped, the types
  covered, the geometry as config.json's;
- the tokenizer against Beam's tokenizer.json;
- the chat template and the output parsing, including tool calls and streaming;
- the server end to end on a stand-in engine (OpenAI and Anthropic APIs);
- the existing server tests unchanged.

**Not run yet - first steps on a GPU:**

1. `mm3_beam_test`: the kernels.
2. Logits on a short prompt against llama.cpp's on the same GGUF. llama.cpp routes Beam over all 256 experts:
   `pack_beam_gguf.py --no-group-limit` for that comparison.
3. A long prompt past 512 positions (the rings) and decode speed (`maya-mm3 --bench`).

**Not done for speed:** the global layers' prompt attention reads each query's keys alone (no query tiling).
