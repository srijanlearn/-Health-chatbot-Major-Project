#!/usr/bin/env python3
"""
HealthyPartner v2 — Export Fine-Tuned Model to Ollama

Merges QLoRA adapters into the base model, exports to GGUF (Q4_K_M),
and registers with Ollama so it can be used as a drop-in replacement.

Steps:
  1. Merge LoRA adapters → full model weights
  2. Convert to GGUF via llama.cpp convert script
  3. Quantise to Q4_K_M (best accuracy/size tradeoff)
  4. Write Modelfile with system prompt
  5. Register: ollama create healthypartner

After this, update HP_MAIN_MODEL=healthypartner in your .env file.

Usage:
  python training/export_to_ollama.py
  python training/export_to_ollama.py --config training/config.yaml --skip-quantise
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "training" / "config.yaml"

LLAMA_CPP_CONVERT = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"
LLAMA_CPP_QUANTISE = Path.home() / "llama.cpp" / "llama-quantize"


def _run(cmd: list, desc: str) -> None:
    log.info("%s", desc)
    log.info("  $ %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error("Command failed (exit %d)", result.returncode)
        sys.exit(1)


def export(config_path: Path, skip_quantise: bool = False) -> None:
    # ── Load config ──────────────────────────────────────────────────────────────
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    training_cfg = cfg["training"]
    export_cfg   = cfg["export"]
    model_cfg    = cfg["model"]
    system_prompt = cfg.get("system_prompt", "").strip()

    adapter_dir  = PROJECT_ROOT / training_cfg["output_dir"] / "final_adapter"
    merged_dir   = PROJECT_ROOT / export_cfg["merged_model_dir"]
    gguf_dir     = PROJECT_ROOT / export_cfg["gguf_dir"]
    quant        = export_cfg["gguf_quantization"]
    model_name   = export_cfg["ollama_model_name"]
    modelfile    = PROJECT_ROOT / export_cfg["ollama_modelfile"]

    if not adapter_dir.exists():
        log.error("Adapter not found: %s", adapter_dir)
        log.error("Run fine-tuning first: python training/finetune.py")
        sys.exit(1)

    gguf_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Merge LoRA adapters ───────────────────────────────────────────────
    log.info("Step 1/4 — Merging LoRA adapters into base model...")
    log.info("  This may take 5-10 minutes and requires ~10 GB disk space.")

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        log.error("Unsloth not installed. Run: pip install unsloth")
        sys.exit(1)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )

    # Merge weights and unload to 16-bit
    model = model.merge_and_unload()
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    log.info("  Merged model saved → %s", merged_dir)

    # ── Step 2: Convert to GGUF ───────────────────────────────────────────────────
    log.info("Step 2/4 — Converting to GGUF (fp16 intermediate)...")

    gguf_fp16 = gguf_dir / "healthypartner-f16.gguf"

    if not LLAMA_CPP_CONVERT.exists():
        log.error(
            "llama.cpp convert script not found at: %s\n"
            "  Install llama.cpp:\n"
            "    git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp\n"
            "    cd ~/llama.cpp && pip install -r requirements.txt",
            LLAMA_CPP_CONVERT,
        )
        sys.exit(1)

    _run(
        [sys.executable, str(LLAMA_CPP_CONVERT), str(merged_dir),
         "--outfile", str(gguf_fp16), "--outtype", "f16"],
        "Running convert_hf_to_gguf.py...",
    )
    log.info("  FP16 GGUF saved → %s (%.1f GB)", gguf_fp16, gguf_fp16.stat().st_size / 1e9)

    # ── Step 3: Quantise ──────────────────────────────────────────────────────────
    gguf_quantised = gguf_dir / f"healthypartner-{quant}.gguf"

    if skip_quantise:
        log.info("Step 3/4 — Skipping quantisation (--skip-quantise flag)")
        gguf_final = gguf_fp16
    else:
        log.info("Step 3/4 — Quantising to %s...", quant.upper())
        log.info("  %s accuracy vs FP16, ~25%% of file size", ">98%" if "q4" in quant else "~96%")

        if not LLAMA_CPP_QUANTISE.exists():
            log.error(
                "llama-quantize binary not found at: %s\n"
                "  Build llama.cpp first:\n"
                "    cd ~/llama.cpp && make llama-quantize",
                LLAMA_CPP_QUANTISE,
            )
            sys.exit(1)

        _run(
            [str(LLAMA_CPP_QUANTISE), str(gguf_fp16), str(gguf_quantised), quant.upper()],
            f"Quantising {gguf_fp16.name} → {gguf_quantised.name}...",
        )

        # Remove fp16 intermediate to save disk space
        gguf_fp16.unlink()
        log.info("  Removed intermediate FP16 file")
        log.info("  Quantised GGUF → %s (%.1f GB)", gguf_quantised, gguf_quantised.stat().st_size / 1e9)
        gguf_final = gguf_quantised

    # ── Step 4: Write Modelfile ────────────────────────────────────────────────────
    log.info("Step 4/4 — Writing Ollama Modelfile...")

    # Determine the base model tag that matches Qwen3-4B for Ollama FROM
    base_model_tag = "qwen3:4b"

    modelfile_content = f"""# HealthyPartner v2 — Ollama Modelfile
# Generated by training/export_to_ollama.py
# Fine-tuned on Indian healthcare Q&A data (IRDAI, PMJAY, drug info, lab reports)

FROM {gguf_final}

SYSTEM \"\"\"{system_prompt}\"\"\"

# Generation parameters (matches training config)
PARAMETER temperature 0.3
PARAMETER num_ctx {model_cfg["max_seq_length"]}
PARAMETER num_predict 512
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""

    modelfile.parent.mkdir(parents=True, exist_ok=True)
    modelfile.write_text(modelfile_content, encoding="utf-8")
    log.info("  Modelfile written → %s", modelfile)

    # ── Register with Ollama ──────────────────────────────────────────────────────
    log.info("Registering model with Ollama as '%s'...", model_name)
    _run(
        ["ollama", "create", model_name, "-f", str(modelfile)],
        f"ollama create {model_name}",
    )

    log.info(
        "\n✅ Export complete!\n\n"
        "  Model registered: %s\n"
        "  Test it:          ollama run %s\n\n"
        "  To use in HealthyPartner, update your .env:\n"
        "    HP_MAIN_MODEL=%s\n"
        "  Then restart the backend:\n"
        "    uvicorn app.main:app --reload",
        model_name, model_name, model_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fine-tuned HealthyPartner model to Ollama")
    parser.add_argument("--config",         type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-quantise",  action="store_true",
                        help="Skip quantisation step (keeps fp16 GGUF, larger file)")
    args = parser.parse_args()

    if not args.config.exists():
        log.error("Config not found: %s", args.config)
        sys.exit(1)

    export(args.config, skip_quantise=args.skip_quantise)
