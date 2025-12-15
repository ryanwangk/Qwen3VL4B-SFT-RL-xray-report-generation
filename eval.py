#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a Unsloth Qwen3 VL checkpoint (base, SFT, or RL) on a FLARE style
JSONL validation set.

- Loads model with FastVisionModel.from_pretrained
- Uses UnslothVisionDataCollator to build inputs from `messages`
- Opens images from IMAGE_ROOT / relative_path
- Generates reports and computes GREEN, BLEU, clinical efficacy
- Saves metrics JSON and preds_refs JSONL as:
    eval_<ckpt_name>_metrics.json
    eval_<ckpt_name>_preds_refs.jsonl
in the same directory as the JSONL file.
"""

import os
import json
import argparse
from copy import deepcopy
from pathlib import Path
import re

import torch
from PIL import Image
import metrics  # your GREEN / BLEU / clinical metrics module

import unsloth  # ensure unsloth patches first
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Validation JSONL path, for example /home/ryanwk/scratch/FLARE_Task5/val_sft_full.jsonl",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Checkpoint directory or HF id to evaluate, for example "
             "/home/ryanwk/scratch/FLARE_Task5/RL_full_40gb_bs8_len640",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root directory where image paths in JSONL are located, e.g. /home/ryanwk/scratch/FLARE_Task5",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Max sequence length for tokenizer / model.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens to generate per example.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="If >0, only evaluate on the first N samples for debugging.",
    )

    return parser.parse_args()


def load_jsonl_dataset(jsonl_path: str):
    """Load a JSONL dataset where each line is a JSON object with a `messages` field."""
    data = []
    n_total = 0
    n_bad = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            n_total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except Exception as e:
                print(f"[warn] Failed to parse line {n_total} of {jsonl_path}, skip: {e}")
                n_bad += 1
    print(f"[data] loaded {len(data)} samples from {jsonl_path}, bad lines = {n_bad}")
    return data


def extract_gt_report_from_messages(messages):
    """
    From FLARE style messages, get the last assistant text as ground truth report.
    """
    if not messages:
        return ""

    # assume last assistant is the label
    last_assistant = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_assistant = m
            break

    if last_assistant is None:
        return ""

    content = last_assistant.get("content", [])
    parts = []
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif isinstance(c, str):
                parts.append(c)
    elif isinstance(content, str):
        parts.append(content)

    return " ".join(p for p in parts if p).strip()


def attach_pil_images(sample, image_root: str):
    """
    Replace `image` string paths in user messages with PIL Images opened from image_root / rel_path.
    This matches how your original baseline eval script worked, but will now feed PIL images
    into UnslothVisionDataCollator.
    """
    root = Path(image_root)

    messages = sample.get("messages", [])
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "image":
                img_path = c.get("image", "")
                if not img_path:
                    continue
                img_full = root / img_path
                try:
                    image = Image.open(img_full).convert("RGB")
                    c["image"] = image
                except Exception as e:
                    print(f"[warn] Failed to open image {img_full}: {e}")
    return sample


def clean_model_output(text: str) -> str:
    """
    Clean the raw decoded output. In particular, strip a leading 'assistant'
    token or line introduced by chat templates.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()

    # remove leading 'assistant' with optional colon and newline
    text = re.sub(r"^\s*assistant\s*[:\n]*\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def load_model_and_tokenizer(ckpt_dir: str, max_seq_length: int = 2048):
    """
    Load a Unsloth FastVisionModel checkpoint (base, SFT, or RL).
    Returns model, tokenizer, image_processor.
    """
    print(f"[model] loading from: {ckpt_dir}")
    model_and_processors = FastVisionModel.from_pretrained(
        ckpt_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        use_gradient_checkpointing="unsloth",
    )

    if isinstance(model_and_processors, tuple):
        if len(model_and_processors) == 3:
            model, tokenizer, image_processor = model_and_processors
        elif len(model_and_processors) == 2:
            model, tokenizer = model_and_processors
            image_processor = None
        else:
            model = model_and_processors[0]
            tokenizer = model_and_processors[1] if len(model_and_processors) > 1 else None
            image_processor = None
    else:
        model = model_and_processors
        tokenizer = None
        image_processor = None

    model.eval()
    return model, tokenizer, image_processor


@torch.no_grad()
def generate_for_sample(sample, model, tokenizer, device, max_new_tokens: int):
    """
    Generate a report for a single sample using UnslothVisionDataCollator.
    Uses all messages except the final assistant as prompt.
    """
    messages = sample.get("messages", [])
    if not messages:
        return ""

    # prompt is all messages up to (but not including) the last assistant
    # if there is no assistant, just use all messages
    prompt_messages = []
    for m in messages:
        if m.get("role") == "assistant":
            break
        prompt_messages.append(m)

    if not prompt_messages:
        prompt_messages = messages

    infer_sample = deepcopy(sample)
    infer_sample["messages"] = prompt_messages

    data_collator = UnslothVisionDataCollator(model, tokenizer)
    batch = data_collator([infer_sample])
    batch.pop("labels", None)
    batch = {k: v.to(device) for k, v in batch.items()}

    input_ids = batch["input_ids"]

    generated_ids = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )

    # keep only new tokens beyond the prompt
    if generated_ids.shape[1] > input_ids.shape[1]:
        new_tokens = generated_ids[:, input_ids.shape[1]:]
    else:
        new_tokens = generated_ids

    if tokenizer is not None:
        out = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )[0]
    else:
        out = "<no tokenizer available>"

    return clean_model_output(out)


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] using: {device}")

    data = load_jsonl_dataset(args.jsonl)
    if args.max_samples > 0:
        data = data[:args.max_samples]
    print(f"[info] num_samples to process = {len(data)}")

    model, tokenizer, _ = load_model_and_tokenizer(
        args.ckpt_dir,
        max_seq_length=args.max_seq_length,
    )
    model.to(device)

    preds = []
    refs = []

    for idx, sample in enumerate(data, start=1):
        # attach PIL images so collator can process them
        sample = attach_pil_images(sample, args.image_root)

        gt_report = extract_gt_report_from_messages(sample.get("messages", []))
        pred = generate_for_sample(
            sample,
            model,
            tokenizer,
            device,
            max_new_tokens=args.max_new_tokens,
        )

        preds.append(pred)
        refs.append(gt_report)

        if idx % 50 == 0:
            print(f"[info] processed {idx} / {len(data)}")

    # free model memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    print("[info] Computing metrics on validation set...")

    green_results = metrics.calculate_green_score(preds, refs)
    bleu = metrics.calculate_bleu_score(preds, refs)
    ce = metrics.calculate_clinical_efficacy_score(preds, refs)

    print("===== GREEN metrics on validation =====")
    for k, v in green_results.items():
        print(f"{k}: {v}")

    print(f"\nBLEU score: {bleu}")
    print(f"Clinical efficacy score: {ce}")

    # save metrics and preds/refs
    val_path = Path(args.jsonl)
    out_dir = val_path.parent

    ckpt_name = Path(args.ckpt_dir).name.rstrip("/")

    metrics_path = out_dir / f"eval_{ckpt_name}_metrics.json"
    preds_refs_path = out_dir / f"eval_{ckpt_name}_preds_refs.jsonl"

    payload = {
        "checkpoint": args.ckpt_dir,
        "green": green_results,
        "bleu": bleu,
        "clinical_efficacy": ce,
        "num_samples": len(preds),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(preds_refs_path, "w", encoding="utf-8") as f:
        for pred, ref in zip(preds, refs):
            rec = {"prediction": pred, "reference": ref}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[info] Saved metrics JSON to: {metrics_path}")
    print(f"[info] Saved predictions and references JSONL to: {preds_refs_path}")


if __name__ == "__main__":
    main()
