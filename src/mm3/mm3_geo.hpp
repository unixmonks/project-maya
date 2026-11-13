// src/mm3/mm3_geo.hpp - maya-mm3: MiniMax-M3.1-Flash's geometry from mm3.txt (both engine paths' - mm3_model.cu,
// mm3_tp.cu), with the checks of what the kernels cover.  A pack whose model_type is beam (tools/mm3/pack_beam_gguf.py:
// Reflection Beam-1.0) reads Beam's geometry instead (beam true; the layer-split path only - mm3_model.cu).
#pragma once

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "mm3_exl3.hpp"
#include "mm3_env.hpp"
#include "mm3_pack.hpp"

namespace mm3 {

struct Geo {
    int dim, n_layers, n_head, n_kv, hd, n_rot, vocab, n_exp, k_exp, ff, ff_dense, ff_shared;
    float theta, eps, alpha, limit, scale;
    std::vector<int> moe, sparse;
    int ix_heads, ix_dim, blk, topk, local;
    int bos, eos;
    int topk_override = 0;   // (experiments: MM3_TOPK keeps that many blocks instead of the model's)
    // Reflection Beam (beam): global layers (swa[li] 0) attend every position without RoPE, their queries scaled by
    // (1 + (p + 1) / nope_len) ^ nope_pow; sliding ones (swa[li] 1) the last `window` with RoPE; each attention output
    // gated per head; four norms a layer (sandwich) and the depth-scaled residual; the router group-limited (n_group,
    // topk_group); the logits soft-capped (softcap, 0: none)
    bool beam = false;
    std::vector<int> swa;
    int window = 0, n_group = 1, topk_group = 1;
    float softcap = 0.0f, nope_pow = 0.0f, nope_len = 1.0f;
    int eos2 = -1;   // (Beam) a second end: <|call|> beside <|return|> (-1: none)

    bool read(const Pack& p, std::string& err) {
        if (p.str("model_type") == "beam") return read_beam(p, err);
        dim = (int) p.num("hidden_size");
        topk_override = ::mm3::env("MM3_TOPK") ? std::atoi(::mm3::env("MM3_TOPK")) : 0;
        n_layers = (int) p.num("num_hidden_layers");
        n_head = (int) p.num("num_attention_heads");
        n_kv = (int) p.num("num_key_value_heads");
        hd = (int) p.num("head_dim");
        n_rot = (int) p.num("rotary_dim", hd * p.num("partial_rotary_factor", 1.0));
        vocab = (int) p.num("vocab_size");
        n_exp = (int) p.num("num_local_experts");
        k_exp = (int) p.num("num_experts_per_tok");
        ff = (int) p.num("intermediate_size");
        ff_dense = (int) p.num("dense_intermediate_size");
        ff_shared = (int) p.num("shared_intermediate_size");
        theta = (float) p.num("rope_theta");
        eps = (float) p.num("rms_norm_eps");
        alpha = (float) p.num("swiglu_alpha");
        limit = (float) p.num("swiglu_limit");
        scale = (float) p.num("routed_scaling_factor");
        moe = p.ints("moe_layer_freq");
        sparse = p.ints("sparse_attention_config.sparse_attention_freq");
        ix_heads = (int) p.num("sparse_attention_config.sparse_num_index_heads");
        ix_dim = (int) p.num("sparse_attention_config.sparse_index_dim");
        blk = (int) p.num("sparse_attention_config.sparse_block_size");
        topk = (int) p.num("sparse_attention_config.sparse_topk_blocks");
        local = (int) p.num("sparse_attention_config.sparse_local_block");
        bos = (int) p.num("bos_token_id");
        eos = (int) p.num("eos_token_id");
        const auto no = [&](const std::string& what) {
            err = "geometry not covered: " + what;
            return false;
        };
        if (hd != 128 || n_rot % 2 || n_rot > hd) return no("head_dim " + std::to_string(hd));
        if (n_head != 16 * n_kv) return no("query heads per KV head " + std::to_string(n_head / std::max(1, n_kv)));
        if (ix_heads != n_kv || ix_dim != 128 || ix_heads != 4) return no("indexer heads / dim");
        if (topk_override > 0) topk = topk_override;
        if (blk != 128 || local != 1 || topk < 1 || topk > 64) return no("sparse blocks (block 128, local 1)");
        if (p.num("sparse_attention_config.sparse_init_block", 0) != 0) return no("init blocks");
        if ((int) moe.size() != n_layers || (int) sparse.size() != n_layers) return no("per-layer lists");
        for (int s : sparse)
            if (!s) return no("a full-attention layer (every layer of M3.1 is sparse)");
        if (std::fabs(alpha - strata::mm3::exl3::kSwigluAlpha) > 1e-6f) return no("swiglu_alpha " + std::to_string(alpha));
        if (k_exp > 8 || n_exp > 256) return no("experts");
        if (dim % 128 || ff % 128 || ff_dense % 128 || ff_shared % 128) return no("matrix sides");
        return true;
    }

    bool read_beam(const Pack& p, std::string& err) {
        beam = true;
        dim = (int) p.num("hidden_size");
        n_layers = (int) p.num("num_hidden_layers");
        n_head = (int) p.num("num_attention_heads");
        n_kv = (int) p.num("num_key_value_heads");
        hd = (int) p.num("head_dim");
        n_rot = hd;   // (the sliding layers' RoPE over the whole head; the global layers none)
        vocab = (int) p.num("vocab_size");
        n_exp = (int) p.num("num_local_experts");
        k_exp = (int) p.num("num_experts_per_tok");
        ff = (int) p.num("intermediate_size");
        ff_dense = (int) p.num("dense_intermediate_size");
        ff_shared = 0;
        theta = (float) p.num("rope_theta", 10000.0);
        eps = (float) p.num("rms_norm_eps", 1e-6);
        alpha = 1.0f;
        limit = 0.0f;   // (the experts' SwiGLU unclamped)
        scale = (float) p.num("routed_scaling_factor", 1.0);
        moe = p.ints("moe_layer_freq");
        swa = p.ints("sliding_layers");
        sparse.assign((size_t) std::max(0, n_layers), 0);
        window = (int) p.num("sliding_window");
        n_group = (int) p.num("n_group", 1);
        topk_group = (int) p.num("topk_group", n_group);
        softcap = (float) p.num("final_logit_softcapping", 0.0);
        nope_pow = (float) p.num("nope_scale_power", 0.1);
        nope_len = (float) p.num("nope_scale_length", 2048.0);
        // (no indexer: blk only keys the decode graphs - the global layers' splits move with the position's block)
        ix_heads = ix_dim = 0;
        blk = 128;
        topk = 1;
        local = 1;
        bos = (int) p.num("bos_token_id", -1);
        eos = (int) p.num("eos_token_id");
        eos2 = (int) p.num("eos2_token_id", -1);
        const auto no = [&](const std::string& what) {
            err = "Beam geometry not covered: " + what;
            return false;
        };
        if (hd != 128) return no("head_dim " + std::to_string(hd));
        if (n_kv <= 0 || n_head % n_kv) return no("query heads per KV head");
        const int G = n_head / n_kv;
        if (G != 4 && G != 6 && G != 8 && G != 16) return no("query heads per KV head " + std::to_string(G));
        if ((int) moe.size() != n_layers || (int) swa.size() != n_layers) return no("per-layer lists");
        if (window <= 0) return no("sliding_window");
        if (k_exp < 1 || k_exp > 8 || n_exp > 256) return no("experts");
        if (n_group < 1 || n_group > 32 || n_exp % n_group || topk_group < 1 || topk_group > n_group ||
            (n_group > 1 && n_exp / n_group < 2))
            return no("expert groups " + std::to_string(n_group) + " / " + std::to_string(topk_group));
        if (dim % 128 || ff % 128 || ff_dense % 128) return no("matrix sides");
        if (nope_len <= 0.0f) return no("nope_scale_length");
        return true;
    }
};


}  // namespace mm3
