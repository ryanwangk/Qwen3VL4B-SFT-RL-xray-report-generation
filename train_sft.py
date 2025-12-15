#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===========================================================
 Supervised Fine Tuning (SFT) for Qwen3-VL-4B
 FLARE Task 5 - X-ray Report Generation

 - Reads train_sft.jsonl and val_sft.jsonl in Qwen3-VL chat format
 - Uses Unsloth FastVisionModel to load base Qwen3-VL-4B-Instruct
 - Applies LoRA to both language and vision backbone
 - Trains with SFTTrainer + UnslothVisionDataCollator
 - Supports wandb logging and optional early stopping
===========================================================
"""

import os
import json
import argparse

import torch
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

import wandb


# ============================================================
# Helpers - load JSONL
# ============================================================
def load_jsonl_dataset(jsonl_path: str):
    """
    Read a JSONL file where each line is a dict like:
      {
        "messages": [
          {
            "role": "user",
            "content": [
              {"type": "image", "image": "/path/to/image1.png"},
              {"type": "image", "image": "/path/to/image2.png"},
              {"type": "text",  "text": "question text here"}
            ]
          },
          {
            "role": "assistant",
            "content": [
              {"type": "text", "text": "x ray report here"}
            ]
          }
        ]
      }

    Returns a Python list of dicts. Unsloth's vision data collator
    knows how to consume this directly.
    """
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
                print(f"[warn] Failed to parse line {n_total} of {jsonl_path}: {e}")
                n_bad += 1

    print(f"[data] loaded {len(data)} samples from {jsonl_path}, bad={n_bad}")
    return data


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_jsonl",
        type=str,
        required=True,
        help="Path to training jsonl file in Qwen3-VL chat format.",
    )
    parser.add_argument(
        "--val_jsonl",
        type=str,
        required=True,
        help="Path to validation jsonl file in Qwen3-VL chat format.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save SFT LoRA checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Qwen3-VL-4B-Instruct",
        help="Base vision language model to fine tune.",
    )

    # Training hyperparameters
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=10)

    # We will control training using max_steps
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Total optimization steps. Use small value on Colab for testing.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1.0,
        help="Kept as a positive constant to avoid None inside Unsloth. "
             "Training will actually be limited by max_steps.",
    )

    # Early stopping and wandb
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Enable EarlyStopping on eval_loss if set to a positive integer.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="flare_task5_sft",
        help="wandb project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="qwen3vl_sft_run",
        help="wandb run name.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["none", "wandb"],
        help="Whether to log to wandb.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # wandb setup
    if args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.wandb_project
        # Optional: set WANDB_ENTITY in the environment for team logging
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # Load datasets
    print("=== Loading datasets ===")
    train_dataset = load_jsonl_dataset(args.train_jsonl)
    eval_dataset = load_jsonl_dataset(args.val_jsonl)
    print(f"[data] train size = {len(train_dataset)}, val size = {len(eval_dataset)}")

    # Basic CUDA debug
    print("=== Torch and CUDA info ===")
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU 0:", torch.cuda.get_device_name(0))

    # Load base Qwen3 VL model
    print("=== Loading base Qwen3-VL via Unsloth ===")
    model_and_processors = FastVisionModel.from_pretrained(
        args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        use_gradient_checkpointing="unsloth",
    )

    # Handle (model, tokenizer, image_processor) or (model, tokenizer)
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

    # Attach LoRA adapters - finetune both language and vision
    print("=== Attaching LoRA adapters ===")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    FastVisionModel.for_training(model)

    # Build SFTConfig
    print("=== Building SFTConfig ===")

    use_early_stopping = (
        args.early_stopping_patience is not None
        and args.early_stopping_patience > 0
    )

    sft_config_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        max_seq_length=args.max_seq_length,
        report_to=args.report_to,
        run_name=args.wandb_run_name,
        optim="adamw_torch",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
    )

    # Mixed precision
    if is_bf16_supported():
        sft_config_kwargs["bf16"] = True
        sft_config_kwargs["fp16"] = False
    else:
        sft_config_kwargs["bf16"] = False
        sft_config_kwargs["fp16"] = True

    # Important: always set both max_steps and num_train_epochs to real numbers
    # to avoid None comparisons inside Unsloth
    sft_config_kwargs["max_steps"] = int(args.max_steps)
    sft_config_kwargs["num_train_epochs"] = float(args.num_train_epochs)

    training_args = SFTConfig(**sft_config_kwargs)

    # Callbacks - optional early stopping on eval_loss
    print("=== Building callbacks (EarlyStopping) ===")
    callbacks = []
    if use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.0,
            )
        )
        print(f"[early stopping] enabled with patience = {args.early_stopping_patience}")
    else:
        print("[early stopping] disabled")

    # Build trainer
    print("=== Building SFTTrainer ===")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        args=training_args,
        callbacks=callbacks,
    )

    # Train
    print("=== Start training ===")
    trainer.train()

    # Save final checkpoint and processors
    print("=== Saving final adapter and processors ===")
    trainer.save_model(args.output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(args.output_dir)
    if image_processor is not None:
        try:
            image_processor.save_pretrained(args.output_dir)
        except Exception as e:
            print(f"[warn] Failed to save image_processor: {e}")

    print(f"[done] SFT finished. Checkpoint saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
