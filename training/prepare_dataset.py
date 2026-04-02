#!/usr/bin/env python3
"""
HealthyPartner v2 — Training Dataset Preparation

Converts raw data sources into a ChatML-formatted JSONL dataset for QLoRA fine-tuning.

Sources:
  1. training/data/sample_qa.jsonl          — hand-curated Q&A pairs
  2. app/knowledge/data/*.json              — auto-converts knowledge graph entries to Q&A

Output:
  training/data/train.jsonl                — 90% split for training
  training/data/eval.jsonl                 — 10% split for evaluation

Each output line is a JSON object:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage:
  python training/prepare_dataset.py
  python training/prepare_dataset.py --config training/config.yaml --preview 5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Default paths (relative to project root) ────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "training" / "config.yaml"

# ── System prompt (loaded from config) ──────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = (
    "You are HealthyPartner, a trusted AI healthcare assistant for India. "
    "You specialize in health insurance policies, IRDAI regulations, Ayushman Bharat (PMJAY), "
    "generic medicine alternatives, lab report interpretation, and general health guidance. "
    "Always give accurate, concise answers. For serious medical concerns, recommend consulting a doctor."
)


# ── Knowledge graph → Q&A converters ───────────────────────────────────────────

def _kg_irdai_to_qa(entry: Dict[str, Any]) -> List[Dict]:
    """Convert an IRDAI regulation entry to Q&A pairs."""
    pairs = []
    condition = entry.get("condition", "")
    waiting = entry.get("waiting_period_days", 0)
    note = entry.get("note", "")
    keywords = entry.get("keywords", [])

    if not condition:
        return pairs

    waiting_years = waiting // 365
    waiting_str = f"{waiting_years} year{'s' if waiting_years != 1 else ''}" if waiting_years else f"{waiting} days"

    pairs.append({
        "user": f"What is the waiting period for {condition} under health insurance in India?",
        "assistant": (
            f"Under IRDAI regulations, the standard waiting period for {condition} is **{waiting_str}** "
            f"from the policy start date.\n\n{note}\n\n"
            f"Always check your specific policy document as waiting periods can vary between insurers."
        ),
    })

    if keywords:
        pairs.append({
            "user": f"Does health insurance cover {keywords[0]} from day one?",
            "assistant": (
                f"No, {condition} typically has a **{waiting_str} waiting period** under most health insurance policies in India. "
                f"This means claims related to {condition} will only be accepted after {waiting_str} of continuous coverage.\n\n"
                f"{note}"
            ),
        })

    return pairs


def _kg_generic_medicine_to_qa(entry: Dict[str, Any]) -> List[Dict]:
    """Convert a generic medicine entry to Q&A pairs."""
    pairs = []
    brand = entry.get("brand_name", "")
    generic = entry.get("generic_name", "")
    dosage = entry.get("dosage", "")
    branded_price = entry.get("branded_price_inr", 0)
    generic_price = entry.get("generic_price_inr", 0)
    savings_pct = entry.get("savings_percent", 0)
    use = entry.get("use", "")

    if not brand or not generic:
        return pairs

    pairs.append({
        "user": f"What is the generic alternative to {brand}?",
        "assistant": (
            f"The active ingredient in **{brand}** is **{generic} {dosage}**, used for {use}.\n\n"
            f"**Cost comparison:**\n"
            f"• Branded ({brand}): ₹{branded_price}/tablet\n"
            f"• Generic ({generic} {dosage}): ₹{generic_price}/tablet\n"
            f"• **Savings: ~{savings_pct}%**\n\n"
            f"Ask your chemist for '{generic} {dosage}' or visit the nearest Jan Aushadhi Kendra. "
            f"Both contain the same active ingredient at the same dose — only the brand and packaging differ."
        ),
    })

    pairs.append({
        "user": f"Is there a cheaper substitute for {brand}?",
        "assistant": (
            f"Yes. **{brand}** contains **{generic} {dosage}**, which is available as a generic medicine "
            f"at Jan Aushadhi stores for ₹{generic_price}/tablet — about **{savings_pct}% cheaper** than the branded version.\n\n"
            f"Generic medicines meet the same quality standards and contain the same active ingredient. "
            f"Just ask your pharmacist for '{generic} {dosage}'."
        ),
    })

    return pairs


def _kg_drug_interaction_to_qa(entry: Dict[str, Any]) -> List[Dict]:
    """Convert a drug interaction entry to Q&A pairs."""
    pairs = []
    drug_a = entry.get("drug_a", "")
    drug_b = entry.get("drug_b", "")
    severity = entry.get("severity", "")
    effect = entry.get("effect", "")
    recommendation = entry.get("recommendation", "")

    if not drug_a or not drug_b:
        return pairs

    pairs.append({
        "user": f"Can I take {drug_a} with {drug_b}?",
        "assistant": (
            f"**Interaction detected — severity: {severity}**\n\n"
            f"Taking {drug_a} with {drug_b} can cause: {effect}\n\n"
            f"**Recommendation:** {recommendation}\n\n"
            f"⚠️ Always consult your doctor or pharmacist before combining medications."
        ),
    })

    return pairs


def _kg_symptom_to_qa(entry: Dict[str, Any]) -> List[Dict]:
    """Convert an ICD-10 symptom entry to Q&A pairs."""
    pairs = []
    symptom = entry.get("symptom", "")
    conditions = entry.get("possible_conditions", [])
    urgency = entry.get("urgency", "")
    action = entry.get("action", "")

    if not symptom or not conditions:
        return pairs

    conditions_str = ", ".join(conditions[:4])
    is_emergency = urgency.lower() in ("emergency", "urgent")

    pairs.append({
        "user": f"What could cause {symptom}?",
        "assistant": (
            f"{'🚨 **This symptom can indicate a medical emergency. Seek care immediately.**' + chr(10) + chr(10) if is_emergency else ''}"
            f"{symptom.capitalize()} can be caused by several conditions including: {conditions_str}.\n\n"
            f"**Urgency level:** {urgency}\n\n"
            f"**What to do:** {action}\n\n"
            f"⚠️ This is general guidance only. A doctor must evaluate your specific situation."
        ),
    })

    return pairs


def _kg_pmjay_to_qa(entry: Dict[str, Any]) -> List[Dict]:
    """Convert a PMJAY entry to Q&A pairs."""
    pairs = []
    topic = entry.get("topic", "")
    details = entry.get("details", "")
    keywords = entry.get("keywords", [])

    if not topic or not details:
        return pairs

    if keywords:
        pairs.append({
            "user": f"How does Ayushman Bharat cover {keywords[0]}?",
            "assistant": f"**{topic}**\n\n{details}",
        })

    pairs.append({
        "user": f"Tell me about {topic} under PM-JAY.",
        "assistant": details,
    })

    return pairs


# ── Knowledge graph loader ───────────────────────────────────────────────────────

KG_CONVERTERS = {
    "irdai_regulations.json":  _kg_irdai_to_qa,
    "generic_medicines.json":  _kg_generic_medicine_to_qa,
    "drug_interactions.json":  _kg_drug_interaction_to_qa,
    "icd10_symptoms.json":     _kg_symptom_to_qa,
    "ayushman_bharat.json":    _kg_pmjay_to_qa,
}


def load_knowledge_graph_qa(data_dir: Path, system_prompt: str) -> List[Dict]:
    """Convert all knowledge graph JSON files into ChatML Q&A examples."""
    examples = []
    for filename, converter in KG_CONVERTERS.items():
        path = data_dir / filename
        if not path.exists():
            log.warning("Knowledge file not found: %s", path)
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # KG files are either a list directly or {"data": [...]}
        entries = data if isinstance(data, list) else data.get("data", [])

        count = 0
        for entry in entries:
            for qa in converter(entry):
                if qa.get("user") and qa.get("assistant"):
                    examples.append({
                        "messages": [
                            {"role": "system",    "content": system_prompt},
                            {"role": "user",      "content": qa["user"].strip()},
                            {"role": "assistant", "content": qa["assistant"].strip()},
                        ]
                    })
                    count += 1

        log.info("  %s → %d Q&A pairs", filename, count)

    return examples


# ── Raw JSONL loader ─────────────────────────────────────────────────────────────

def load_raw_jsonl(path: Path, system_prompt: str) -> List[Dict]:
    """
    Load a JSONL file of Q&A pairs.

    Supports two formats:
      1. {"messages": [...]}                    — already in ChatML format
      2. {"user": "...", "assistant": "..."}    — simple Q&A, wraps with system prompt
    """
    examples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("Line %d in %s: JSON parse error — %s", line_no, path.name, e)
                continue

            if "messages" in obj:
                # Already formatted — ensure system prompt is present
                msgs = obj["messages"]
                if not any(m.get("role") == "system" for m in msgs):
                    msgs = [{"role": "system", "content": system_prompt}] + msgs
                examples.append({"messages": msgs})

            elif "user" in obj and "assistant" in obj:
                examples.append({
                    "messages": [
                        {"role": "system",    "content": system_prompt},
                        {"role": "user",      "content": obj["user"].strip()},
                        {"role": "assistant", "content": obj["assistant"].strip()},
                    ]
                })
            else:
                log.warning("Line %d in %s: unknown format, skipping", line_no, path.name)

    return examples


# ── Main ─────────────────────────────────────────────────────────────────────────

def prepare(config_path: Path, preview: int = 0) -> None:
    # Load config
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    system_prompt = cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT).strip()
    dataset_cfg = cfg.get("dataset", {})
    raw_sources = [PROJECT_ROOT / p for p in dataset_cfg.get("raw_sources", [])]
    train_file  = PROJECT_ROOT / dataset_cfg.get("train_file", "training/data/train.jsonl")
    eval_file   = PROJECT_ROOT / dataset_cfg.get("eval_file",  "training/data/eval.jsonl")
    eval_split  = dataset_cfg.get("eval_split", 0.1)
    max_examples = dataset_cfg.get("max_examples", 10000)
    kg_data_dir = PROJECT_ROOT / "app" / "knowledge" / "data"

    log.info("Loading data sources...")

    all_examples: List[Dict] = []

    # Load raw JSONL sources
    for src in raw_sources:
        if not src.exists():
            log.warning("Source not found: %s", src)
            continue
        if src.suffix == ".jsonl":
            loaded = load_raw_jsonl(src, system_prompt)
            log.info("  %s → %d examples", src.name, len(loaded))
            all_examples.extend(loaded)
        elif src.suffix == ".json" and src.parent == kg_data_dir:
            pass  # handled below via KG loader

    # Load knowledge graph data
    log.info("Converting knowledge graph data...")
    kg_examples = load_knowledge_graph_qa(kg_data_dir, system_prompt)
    log.info("  Knowledge graph total → %d examples", len(kg_examples))
    all_examples.extend(kg_examples)

    # Deduplicate by user message
    seen: set[str] = set()
    deduped = []
    for ex in all_examples:
        user_msg = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
        if user_msg not in seen:
            seen.add(user_msg)
            deduped.append(ex)
    log.info("After deduplication: %d examples (removed %d duplicates)", len(deduped), len(all_examples) - len(deduped))

    # Shuffle and cap
    random.seed(42)
    random.shuffle(deduped)
    if len(deduped) > max_examples:
        log.info("Capping at %d examples (had %d)", max_examples, len(deduped))
        deduped = deduped[:max_examples]

    # Split
    eval_count  = max(1, int(len(deduped) * eval_split))
    train_count = len(deduped) - eval_count
    train_data  = deduped[:train_count]
    eval_data   = deduped[train_count:]

    log.info("Split: %d train / %d eval", len(train_data), len(eval_data))

    # Preview
    if preview > 0:
        log.info("\n── Preview (%d examples) ─────────────────────", preview)
        for ex in train_data[:preview]:
            for msg in ex["messages"]:
                role = msg["role"].upper()
                content = msg["content"][:120].replace("\n", " ")
                print(f"  [{role}] {content}")
            print()

    # Write
    train_file.parent.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, data: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        log.info("Wrote %d examples → %s", len(data), path)

    write_jsonl(train_file, train_data)
    write_jsonl(eval_file,  eval_data)
    log.info("Dataset preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare fine-tuning dataset for HealthyPartner v2")
    parser.add_argument("--config",  type=Path, default=DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument("--preview", type=int,  default=0,              help="Print N example messages")
    args = parser.parse_args()

    if not args.config.exists():
        log.error("Config not found: %s", args.config)
        sys.exit(1)

    prepare(args.config, args.preview)
