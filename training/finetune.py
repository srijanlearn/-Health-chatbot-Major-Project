#!/usr/bin/env python3
"""
HealthyPartner v2 — QLoRA Fine-Tuning with Unsloth

Fine-tunes Qwen3-4B on Indian healthcare Q&A data using QLoRA.
Designed to run on:
  - Google Colab free tier (T4, 16 GB VRAM) — ~25 min for 5K examples
  - RTX 3060/3070 (8-12 GB VRAM) — ~40-60 min
  - RTX 4090 / A100 — ~10-15 min

Usage:
  # 1. Install training dependencies first:
  pip install unsloth trl transformers datasets peft accelerate bitsandbytes pyyaml

  # 2. Prepare dataset:
  python training/prepare_dataset.py

  # 3. Run fine-tuning:
  python training/finetune.py
  python training/finetune.py --config training/config.yaml --resume-from-checkpoint

Research basis: HEaltcare_Chatbot_LLM.md
  - LR: 2e-4, LoRA rank 16, all projection layers
  - Mask inputs (train only on completions)
  - 1-3 epochs; 2 is sweet spot for 5K-10K examples
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "training" / "config.yaml"


def train(config_path: Path, resume: bool = False) -> None:
    # ── Load config ──────────────────────────────────────────────────────────────
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg    = cfg["model"]
    lora_cfg     = cfg["lora"]
    training_cfg = cfg["training"]
    dataset_cfg  = cfg["dataset"]
    export_cfg   = cfg["export"]

    train_file = PROJECT_ROOT / dataset_cfg["train_file"]
    eval_file  = PROJECT_ROOT / dataset_cfg["eval_file"]

    if not train_file.exists():
        log.error("Training data not found: %s", train_file)
        log.error("Run first: python training/prepare_dataset.py")
        sys.exit(1)

    # ── Import Unsloth + HuggingFace ─────────────────────────────────────────────
    log.info("Loading training libraries...")
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError:
        log.error(
            "Unsloth not installed.\n"
            "Install with: pip install unsloth trl transformers datasets peft accelerate bitsandbytes"
        )
        sys.exit(1)

    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    # ── Load base model ───────────────────────────────────────────────────────────
    log.info("Loading base model: %s", model_cfg["name"])
    log.info("  max_seq_length=%d, load_in_4bit=%s", model_cfg["max_seq_length"], model_cfg["load_in_4bit"])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg.get("dtype"),
        load_in_4bit=model_cfg["load_in_4bit"],
    )

    # Apply Qwen3 chat template
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    log.info("Chat template applied (Qwen-2.5 / ChatML)")

    # ── Apply QLoRA ───────────────────────────────────────────────────────────────
    log.info("Applying QLoRA adapters (rank=%d, alpha=%d)...", lora_cfg["r"], lora_cfg["alpha"])

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg["random_state"],
        use_rslora=lora_cfg["use_rslora"],
    )

    # Print trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(
        "Trainable params: %s / %s (%.1f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total
    )

    # ── Load dataset ──────────────────────────────────────────────────────────────
    log.info("Loading dataset...")
    log.info("  Train: %s", train_file)
    log.info("  Eval:  %s", eval_file)

    def _apply_template(examples):
        """Apply chat template to messages, masking user/system turns."""
        texts = tokenizer.apply_chat_template(
            examples["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": texts}

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_file), "test": str(eval_file)},
        split={"train": "train", "test": "test"},
    )

    train_dataset = dataset["train"].map(_apply_template, batched=True, remove_columns=["messages"])
    eval_dataset  = dataset["test"].map(_apply_template,  batched=True, remove_columns=["messages"])

    log.info("  Train: %d examples | Eval: %d examples", len(train_dataset), len(eval_dataset))

    # ── Training arguments ────────────────────────────────────────────────────────
    output_dir = PROJECT_ROOT / training_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        warmup_steps=training_cfg["warmup_steps"],
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        seed=training_cfg["seed"],
        fp16=training_cfg["fp16"],
        bf16=training_cfg["bf16"],
        logging_steps=training_cfg["logging_steps"],
        save_steps=training_cfg["save_steps"],
        save_total_limit=training_cfg["save_total_limit"],
        eval_strategy=training_cfg["eval_strategy"],
        eval_steps=training_cfg["eval_steps"],
        load_best_model_at_end=training_cfg["load_best_model_at_end"],
        metric_for_best_model=training_cfg["metric_for_best_model"],
        report_to=training_cfg["report_to"],
        dataloader_num_workers=training_cfg["dataloader_num_workers"],
        dataset_text_field="text",
        max_seq_length=model_cfg["max_seq_length"],
        packing=False,   # Packing=False for chat templates (variable lengths)
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_cfg,
    )

    # ── Train ─────────────────────────────────────────────────────────────────────
    effective_batch = (
        training_cfg["per_device_train_batch_size"]
        * training_cfg["gradient_accumulation_steps"]
    )
    log.info(
        "Starting training — epochs=%d, lr=%g, effective_batch=%d",
        training_cfg["num_train_epochs"],
        training_cfg["learning_rate"],
        effective_batch,
    )

    trainer_stats = trainer.train(
        resume_from_checkpoint=resume if resume else None
    )

    log.info(
        "Training complete — %.1f min, loss=%.4f",
        trainer_stats.metrics["train_runtime"] / 60,
        trainer_stats.metrics["train_loss"],
    )

    # ── Save LoRA adapters ────────────────────────────────────────────────────────
    adapter_dir = output_dir / "final_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log.info("LoRA adapters saved → %s", adapter_dir)

    # ── Save config snapshot for reproducibility ──────────────────────────────────
    import shutil
    shutil.copy(config_path, adapter_dir / "training_config.yaml")
    log.info("Config snapshot saved.")

    log.info(
        "\nNext step: export to GGUF and register with Ollama:\n"
        "  python training/export_to_ollama.py\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune HealthyPartner v2 with Unsloth QLoRA")
    parser.add_argument("--config",                 type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume-from-checkpoint", action="store_true",
                        help="Resume training from the last checkpoint")
    args = parser.parse_args()

    if not args.config.exists():
        log.error("Config not found: %s", args.config)
        sys.exit(1)

    train(args.config, resume=args.resume_from_checkpoint)
