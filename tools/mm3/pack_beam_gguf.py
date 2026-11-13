"""tools/mm3/pack_beam_gguf.py - maya-mm3's pack for a Reflection Beam-1.0-501B-A23B GGUF set (llama.cpp's `beam`
layout: unsloth's, bartowski's ...), beside the model's files (no weights copied).

    python tools/mm3/pack_beam_gguf.py --model ~/models/Beam-1.0-501B-A23B-GGUF/UD-Q2_K_XL

writes <model>/maya-mm3/ in pack_gguf.py's format, so maya-mm3 reads it:
  mm3.txt       model_type beam and Beam's geometry under config.json's names (from the GGUF's beam.* keys): the
                layers' sliding pattern (sliding_layers), the window, the NoPE query scale, the logit softcap, the expert
                groups; format gguf, quant_name, model_dir
  tensors.txt   every tensor under the names the engine reads (pack_gguf.py's, MiniMax's where the role is the same):
                the routed experts an entry each (layers.N.block_sparse_moe.experts.E.w1 / w3 / w2.weight); Beam's own
                as self_attn.gate_proj.weight / self_attn.gate_bias (the attention gate), self_attn.post_norm.weight and
                mlp_post_norm.weight (the sandwich norms after the attention and after the MLP);
                post_attention_layernorm.weight is the MLP's input norm (llama.cpp's ffn_norm), as MiniMax's
  tokenizer/    the server's tokenizer from the GGUF's vocabulary (pre "qwen2", NFC - Beam's tokenizer.json), its
                special ids (eos <|return|>, eos2 <|call|>), and Beam's chat template (the GGUF's own, else this
                directory's beam_chat_template.jinja - Reflection's, Apache-2.0)

The expert groups: Beam routes within the 4 best of 8 groups of 32 experts (config.json's n_group / topk_group, as
transformers' and SGLang's BeamTopkRouter).  llama.cpp's converter writes no group keys, so its `beam` graph routes over
all 256: --no-group-limit packs that (for comparing logits with llama.cpp on the same GGUF); the default is Beam's.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
TEMPLATE = HERE / "beam_chat_template.jinja"

_spec = importlib.util.spec_from_file_location("ds4_pack_gguf", HERE.parent / "ds4" / "pack_gguf.py")
_ds4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ds4)
Gguf, TYPES = _ds4.Gguf, _ds4.TYPES   # (maya-ds4's GGUF reader: the shards' metadata and tensor infos)

ARCH = "beam"
P = "language_model.model."
# GGUF name (blk.N. stripped) -> the engine's under layers.N.
LAYER = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "self_attn.post_norm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",          # (the MLP's input norm: the engine's post1)
    "post_ffw_norm.weight": "mlp_post_norm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_gate.weight": "self_attn.gate_proj.weight",
    "attn_gate.bias": "self_attn.gate_bias",
    "ffn_gate_inp.weight": "block_sparse_moe.gate.weight",
    "exp_probs_b.bias": "block_sparse_moe.e_score_correction_bias",
    "ffn_gate_exps.weight": "block_sparse_moe.experts.w1",          # [n_exp][ff][dim]: an entry an expert
    "ffn_up_exps.weight": "block_sparse_moe.experts.w3",
    "ffn_down_exps.weight": "block_sparse_moe.experts.w2",          # [n_exp][dim][ff]
    "ffn_gate.weight": "mlp.gate_proj.weight",                      # (the leading dense layers)
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
GLOBAL = {"token_embd.weight": P + "embed_tokens.weight", "output_norm.weight": P + "norm.weight",
          "output.weight": "language_model.lm_head.weight"}
EVERY = ("attn_norm.weight", "post_attention_norm.weight", "ffn_norm.weight", "post_ffw_norm.weight", "attn_q.weight",
         "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
         "attn_gate.weight", "attn_gate.bias")
DENSE = ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")
MOE = ("ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
# the matrices the engine's GGUF kernels take as they are stored (ds4_iq's mv types; the experts: its expert types)
MATRIX_TYPES = {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S",
                "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL"}
EXPERT_TYPES = MATRIX_TYPES - {"Q6_K", "Q8_0"}
BEAM_GROUPS = (8, 4)   # config.json's n_group, topk_group
NOPE = (0.1, 2048.0)   # transformers' compute_attention_scaling (llama.cpp's defaults where the GGUF has no key)


def geometry(kv: dict, vocab: int, group_limit: bool) -> dict:
    """config.json's names (what the engine's Geo::read_beam reads) from the GGUF's beam.* keys."""
    g = lambda k, d=None: kv.get("%s.%s" % (ARCH, k), d)  # noqa: E731
    nl, lead = int(g("block_count")), int(g("leading_dense_block_count", 0))
    pattern = g("attention.sliding_window_pattern")
    if not isinstance(pattern, list) or len(pattern) != nl:
        raise SystemExit("the GGUF has no beam.attention.sliding_window_pattern of %d layers" % nl)
    heads = g("attention.head_count")
    kvh = g("attention.head_count_kv")
    if isinstance(heads, list) or isinstance(kvh, list):
        if len(set(heads if isinstance(heads, list) else [heads])) != 1 or \
                len(set(kvh if isinstance(kvh, list) else [kvh])) != 1:
            raise SystemExit("per-layer head counts that differ: not Beam's geometry")
        heads = heads[0] if isinstance(heads, list) else heads
        kvh = kvh[0] if isinstance(kvh, list) else kvh
    ng, tg = (int(g("expert_group_count", BEAM_GROUPS[0])), int(g("expert_group_used_count", BEAM_GROUPS[1]))) \
        if group_limit else (1, 1)
    return {
        "model_type": "beam",
        "hidden_size": g("embedding_length"),
        "num_hidden_layers": nl,
        "num_attention_heads": heads,
        "num_key_value_heads": kvh,
        "head_dim": g("attention.key_length"),
        "vocab_size": vocab,
        "num_local_experts": g("expert_count"),
        "num_experts_per_tok": g("expert_used_count"),
        "intermediate_size": g("expert_feed_forward_length"),
        "dense_intermediate_size": g("feed_forward_length"),
        "rope_theta": g("rope.freq_base_swa", g("rope.freq_base", 10000.0)),
        "rms_norm_eps": g("attention.layer_norm_rms_epsilon"),
        "routed_scaling_factor": g("expert_weights_scale", 1.0),
        "max_position_embeddings": g("context_length"),
        "moe_layer_freq": [0] * lead + [1] * (nl - lead),
        "sliding_layers": [1 if p else 0 for p in pattern],
        "sliding_window": g("attention.sliding_window"),
        "n_group": ng,
        "topk_group": tg,
        "final_logit_softcapping": g("final_logit_softcapping", 0.0),
        "nope_scale_power": g("attention.temperature_scale", NOPE[0]),
        "nope_scale_length": g("attention.temperature_length", NOPE[1]),
    }


def write_tokenizer(out: pathlib.Path, kv: dict) -> dict:
    """pack_gguf.py's layout from the GGUF's vocabulary (pre qwen2), Beam's special ids, its chat template."""
    tokens = kv["tokenizer.ggml.tokens"]
    types = kv.get("tokenizer.ggml.token_type") or [1] * len(tokens)
    merges = kv.get("tokenizer.ggml.merges") or []
    if len(set(tokens)) != len(tokens):
        raise SystemExit("the GGUF's vocabulary has duplicate token strings")
    if kv.get("tokenizer.ggml.model") != "gpt2":
        raise SystemExit("the GGUF's tokenizer is not byte-level BPE (tokenizer.ggml.model %r)" %
                         kv.get("tokenizer.ggml.model"))
    pre = kv.get("tokenizer.ggml.pre")
    if pre not in (None, "qwen2"):
        print("note: the GGUF names its pre-tokenizer %r - the pack transcribes Beam's (qwen2)" % pre)
    ids = {t: i for i, t in enumerate(tokens)}
    for t in ("<|return|>", "<|call|>", "<|start|>", "<|end|>", "<|message|>", "<|channel|>"):
        if t not in ids:
            raise SystemExit("the GGUF's vocabulary has no %s: not Beam's" % t)
    special = {"eos": ids["<|return|>"], "eos2": ids["<|call|>"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps([int(t) for t in types]))
    (out / "tokenizer.json").write_text(json.dumps({"pre": "qwen2", "special_ids": special}, indent=1))
    own = kv.get("tokenizer.chat_template")
    tpl = own if isinstance(own, str) and '<rfl:parameter name="' in own else TEMPLATE.read_text(encoding="utf-8")
    (out / "chat_template.jinja").write_text(tpl, encoding="utf-8")
    return special


def find_shards(model: pathlib.Path, names: list[str] | None) -> list[pathlib.Path]:
    paths = [model / n for n in names] if names else sorted(model.glob("*.gguf"))
    missing = [p for p in paths if not p.is_file()]
    if not paths or missing:
        raise SystemExit("%s: no .gguf files" % model if not paths else "%s: not there" % missing[0])
    return paths


def pack(model: pathlib.Path, out: pathlib.Path, shards: list[str] | None = None, group_limit: bool = True,
         quiet: bool = False) -> dict:
    paths = find_shards(model, shards)
    gs = [Gguf(p) for p in paths]
    kv = next((g.kv for g in gs if "general.architecture" in g.kv), None)
    if kv is None or kv.get("general.architecture") != ARCH:
        raise SystemExit("%s: not a Beam GGUF (architecture %r)" % (model, kv and kv.get("general.architecture")))
    split = int(kv.get("split.count", 1))
    if split != len(gs):
        raise SystemExit("%s: %d of the %d shards are here" % (model, len(gs), split))
    # (unsloth's first shard holds the metadata alone, no tensors: it ends before its data's aligned start - whole)
    why = next((w for w in (_ds4.short(g) for g in gs if g.tensors) if w), None)
    if why:
        raise SystemExit("%s: not whole (%s) - a download still running?" % (model, why))
    rel = lambda p: os.path.relpath(p, model).replace(os.sep, "/")  # noqa: E731
    files = [rel(p) for p in paths]
    if any(" " in f for f in files):
        raise SystemExit("a file name with a space: %s" % [f for f in files if " " in f])
    n_exp = int(kv["%s.expert_count" % ARCH])
    nl = int(kv["%s.block_count" % ARCH])
    lead = int(kv.get("%s.leading_dense_block_count" % ARCH, 0))
    tensors, unknown, have = [], [], set()
    for fi, g in enumerate(gs):
        for name, typ, dims, off in g.tensors:
            if typ not in TYPES:
                raise SystemExit("%s: %s has ggml type %d, which the engine does not read" % (g.path.name, name, typ))
            tname, per, bpb = TYPES[typ]
            nel = 1
            for d in dims:
                nel *= d
            if dims[0] % per:
                raise SystemExit("%s: %s rows of %d are not whole %s blocks" % (g.path.name, name, dims[0], tname))
            nbytes = nel // per * bpb
            m = re.match(r"blk\.(\d+)\.(.+)$", name)
            if m and m.group(2) in LAYER:
                ck = "%slayers.%s.%s" % (P, m.group(1), LAYER[m.group(2)])
                have.add((int(m.group(1)), m.group(2)))
            elif name in GLOBAL:
                ck = GLOBAL[name]
                have.add((-1, name))
            else:
                unknown.append(name)
                continue
            shape = list(reversed(dims))     # ggml ne0 innermost -> torch order, outermost first
            base = g.data_start + off
            role = m.group(2) if m else name
            if role.endswith("_exps.weight"):
                if tname not in EXPERT_TYPES:
                    raise SystemExit("%s: %s is %s - the engine's expert kernels take %s" %
                                     (g.path.name, name, tname, ", ".join(sorted(EXPERT_TYPES))))
                if shape[0] != n_exp:
                    raise SystemExit("%s: %s holds %d experts, not %d" % (g.path.name, name, shape[0], n_exp))
                part = nbytes // n_exp
                pre, r = ck.rsplit(".", 1)
                for i in range(n_exp):
                    tensors.append(("%s.%d.%s.weight" % (pre, i, r), tname, fi, base + i * part, part, shape[1:]))
                continue
            if len(shape) == 2 and role not in ("token_embd.weight", "ffn_gate_inp.weight") and \
                    tname not in MATRIX_TYPES:
                raise SystemExit("%s: %s is %s - the engine's GGUF matrices are block types (%s)" %
                                 (g.path.name, name, tname, ", ".join(sorted(MATRIX_TYPES))))
            tensors.append((ck, tname, fi, base, nbytes, shape))
    if unknown:
        raise SystemExit("tensors the pack does not map (%d): %s" % (len(unknown), ", ".join(unknown[:12])))
    lacking = []
    for li in range(nl):
        for n in EVERY + (DENSE if li < lead else MOE):
            if (li, n) not in have:
                lacking.append("blk.%d.%s" % (li, n))
    for n in GLOBAL:
        if (-1, n) not in have and n != "output.weight":
            lacking.append(n)
    if (-1, "output.weight") not in have:
        lacking.append("output.weight (a tied head is not covered)")
    if lacking:
        raise SystemExit("the set lacks %d tensors: %s" % (len(lacking), ", ".join(lacking[:8])))
    gate_up = {}
    for t in tensors:   # (the engine's plain SwiGLU kernels: a layer's gate and up of one type)
        mm = re.search(r"layers\.(\d+)\.block_sparse_moe\.experts\.0\.(w1|w3)\.weight$", t[0])
        if mm:
            gate_up.setdefault(int(mm.group(1)), {})[mm.group(2)] = t[1]
    two = [li for li, d in sorted(gate_up.items()) if d.get("w1") != d.get("w3")]
    if two:
        raise SystemExit("layer%s %s: the experts' gate and up are of two types (%s / %s) - not covered" % (
            "s" if len(two) > 1 else "", ", ".join(map(str, two[:6])), gate_up[two[0]].get("w1"),
            gate_up[two[0]].get("w3")))
    embd = next(t for t in tensors if t[0] == GLOBAL["token_embd.weight"])
    geo = geometry(kv, embd[5][0], group_limit)

    out.mkdir(parents=True, exist_ok=True)
    special = write_tokenizer(out / "tokenizer", kv)
    dense = sorted({t[1] for t in tensors if ".experts." not in t[0] and t[1] != "F32"})
    exp_types = {}
    for t in tensors:
        if ".experts.0." in t[0]:
            exp_types[t[1]] = exp_types.get(t[1], 0) + 1
    with (out / "mm3.txt").open("w", newline="\n") as f:
        for k, v in geo.items():
            if v is None:
                continue
            if isinstance(v, bool):
                v = int(v)
            f.write("%s %s\n" % (k, ",".join(str(x) for x in v) if isinstance(v, list) else v))
        f.write("bos_token_id -1\n")
        f.write("eos_token_id %d\n" % special["eos"])
        f.write("eos2_token_id %d\n" % special["eos2"])
        for k, v in special.items():
            f.write("special.%s %d\n" % (k, v))
        f.write("format gguf\n")
        f.write("quant_name %s\n" % str(kv.get("general.name", model.name)).replace(" ", "-"))
        f.write("gguf.dense %s\n" % ",".join(dense))
        f.write("gguf.experts %s\n" % ",".join("%sx%d" % (k, v) for k, v in sorted(exp_types.items())))
        f.write("model_dir %s\n" % model)
    with (out / "tensors.txt").open("w", newline="\n") as f:
        for i, name in enumerate(files):
            f.write("file %d %s\n" % (i, name))
        for name, dt, fi, off, nb, shape in tensors:
            f.write("t %s %s %d %d %d %s\n" % (name, dt, fi, off, nb, " ".join(map(str, shape))))
    if not quiet:
        by = {}
        for t in tensors:
            by[t[1]] = by.get(t[1], 0) + t[4]
        nswa = sum(geo["sliding_layers"])
        print("%s: %d tensors in %d files (%s) -> %s" % (
            model.name, len(tensors), len(files), ", ".join("%s %.1f GB" % (k, v / 1e9) for k, v in sorted(by.items())),
            out))
        print("  layers: %d (%d dense, %d MoE; %d global, %d sliding at %s); experts %s, groups %d/%d" % (
            nl, lead, nl - lead, nl - nswa, nswa, geo["sliding_window"],
            ", ".join("%s %d" % (k, v) for k, v in sorted(exp_types.items())), geo["n_group"], geo["topk_group"]))
        print("  special ids: %s" % special)
    return geo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the directory of the GGUF's shards")
    ap.add_argument("--out", help="default: <model>/maya-mm3")
    ap.add_argument("--shards", nargs="+", help="the set's files in --model (default: every .gguf there)")
    ap.add_argument("--no-group-limit", action="store_true",
                    help="route over every expert (llama.cpp's beam graph) instead of Beam's 4 of 8 groups")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser() if a.out else model / "maya-mm3"
    pack(model, out, a.shards, not a.no_group_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
