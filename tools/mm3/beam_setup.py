"""tools/mm3/beam_setup.py - Reflection Beam-1.0 on maya-mm3 without the installer's model list: the pack beside a Beam
GGUF set (pack_beam_gguf.py, once) and the server's config, maya-beam.json in the Maya folder.

    python tools/mm3/beam_setup.py --model ~/models/Beam-1.0-501B-A23B-GGUF/UD-Q2_K_XL --context 262144
    .venv/bin/python serve/server.py --engine maya --config maya-beam.json

The engine is build/maya-mm3 (cmake --build build --target maya-mm3); Beam runs on its layer split (no --tp, no
--parallel, no DSpark draft, text only).  Its K / V: the 9 global layers' every position (36 KiB a token in BF16, half
in INT8 - the default), the 43 sliding layers' last 512 in a ring.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
_spec = importlib.util.spec_from_file_location("pack_beam_gguf", HERE / "pack_beam_gguf.py")
_pack = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pack)

MODEL_NAME = "Beam-1.0-501B-A23B"
SAMPLING = {"temperature": 0.7, "top_p": 0.99}   # (generation_config.json's)


def quant_name(pack: pathlib.Path) -> str:
    """The pack's quant_name (the GGUF's general.name), for the server's label."""
    for line in (pack / "mm3.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("quant_name "):
            return line.split(" ", 1)[1].strip()
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the folder of the Beam GGUF's shards")
    ap.add_argument("--context", type=int, default=131072, help="positions (Beam reads up to 1048576)")
    ap.add_argument("--kv-cache", default="int8", choices=["bf16", "int8", "fp8"], help="the K / V cache's format")
    ap.add_argument("--gpus", help="e.g. 0,1 (default: every GPU)")
    ap.add_argument("--chunk", type=int, help="prompt tokens a forward (the engine's default 1024)")
    ap.add_argument("--repack", action="store_true", help="build the pack again")
    ap.add_argument("--no-group-limit", action="store_true", help="llama.cpp's routing (every expert a candidate)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", help="where the server listens (default: the server's, 127.0.0.1)")
    ap.add_argument("--api-key", help="the key clients send (MAYA_API_KEY works as well)")
    ap.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high", "xhigh", "max"],
                    help="the default thinking level (none: the template's, max)")
    ap.add_argument("--config", default=str(ROOT / "maya-beam.json"))
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    pack = model / "maya-mm3"
    if a.repack or not (pack / "mm3.txt").is_file() or "model_type beam" not in (pack / "mm3.txt").read_text():
        _pack.pack(model, pack, None, not a.no_group_limit)
    else:
        print("pack: %s (kept; --repack builds it again)" % pack)
    exe = ROOT / "build" / ("maya-mm3.exe" if sys.platform == "win32" else "maya-mm3")
    if not exe.is_file():
        print("note: %s is not built yet - cmake --build build --target maya-mm3" % exe)
    args = ["--pack", str(pack), "--max-context", str(a.context), "--kv-cache", a.kv_cache, "--vision", "off"]
    if a.gpus:
        args += ["--gpus", a.gpus]
    if a.chunk:
        args += ["--chunk", str(a.chunk)]
    cfg_path = pathlib.Path(a.config).expanduser()
    cfg = {"exe": str(exe), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": MODEL_NAME, "sampling": dict(SAMPLING), "port": a.port,
           "log": str(cfg_path.with_suffix(".log")),
           "installer": {"family": "minimax", "quant": MODEL_NAME, "model_dir": str(model), "beam": True,
                         "quant_name": quant_name(pack),
                         "written": time.strftime("%Y-%m-%d %H:%M")}}
    if a.reasoning_effort:
        cfg["reasoning_effort"] = a.reasoning_effort
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print("config: %s" % cfg_path)
    print("start:  %s serve/server.py --engine maya --config %s" % (ROOT / ".venv" / "bin" / "python", cfg_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
