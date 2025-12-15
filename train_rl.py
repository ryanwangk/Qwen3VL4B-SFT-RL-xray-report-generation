#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===========================================================
 Reinforcement Learning (GRPO) for Qwen3-VL-4B
 FLARE Task 5 - X-ray Report Generation
===========================================================
Works with Unsloth + SFT LoRA checkpoints and requires
RadGraph for clinical rewards.
"""

import os
import json
import argparse
import re
import sys
import types

# Unsloth must be imported early so it can patch transformers / bitsandbytes
import unsloth
from unsloth import FastVisionModel, is_bf16_supported

import torch
from trl import GRPOTrainer, GRPOConfig
import wandb

from rouge_score import rouge_scorer

import transformers
from transformers import tokenization_utils_base as tub

# ============================================================
# Patches for RadGraph / old AllenNLP stack
# ============================================================

# 1) RadGraph expects transformers.AdamW and transformers.optimization.AdamW.
#    Modern Transformers removed these, so we alias to torch.optim.AdamW.
if not hasattr(transformers, "AdamW"):
    transformers.AdamW = torch.optim.AdamW

try:
    import transformers.optimization as _tf_optim
except Exception:
    _tf_optim = types.ModuleType("transformers.optimization")
    sys.modules["transformers.optimization"] = _tf_optim

if not hasattr(_tf_optim, "AdamW"):
    _tf_optim.AdamW = torch.optim.AdamW

# 2) RadGraph passes `add_special_tokens` as a kwarg into
#    PreTrainedTokenizerBase.__init__, which conflicts with modern API.
_old_init = tub.PreTrainedTokenizerBase.__init__


def _patched_init(self, *args, **kwargs):
    kwargs.pop("add_special_tokens", None)
    return _old_init(self, *args, **kwargs)


tub.PreTrainedTokenizerBase.__init__ = _patched_init

# ============================================================
# RadGraph: mandatory clinical scorer
# ============================================================
from radgraph import F1RadGraph  # noqa: E402

print("[info] Initializing RadGraph F1 scorer (mandatory)...")
try:
    f1radgraph = F1RadGraph(
        reward_level="all",
        model_type="radgraph-xl",  # main model used in paper
    )
    print("[info] RadGraph initialized successfully.")
except Exception as e:
    raise RuntimeError(
        "RadGraph is required for clinical reward, but failed to initialize.\n"
        f"Underlying error: {e}\n"
        "Make sure:\n"
        "  - `pip install radgraph` is done in this venv, and\n"
        "  - `radgraph.tar.gz` is present under the RadGraph cache directory,\n"
        "    e.g. ~/.cache/radgraph/0.1.0/radgraph.tar.gz.\n"
    )


# ============================================================
# Dataset loading
# ============================================================

def load_jsonl_dataset(jsonl_path: str):
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
            except Exception as e:
                print(f"[warn] failed to parse line {n_total} of {jsonl_path}, skip: {e}")
                n_bad += 1
                continue
            data.append(obj)
    print(f"[data] loaded {len(data)} samples from {jsonl_path}, bad lines = {n_bad}")
    return data


def load_rl_dataset(path):
    """
    Load RL dataset from SFT-style JSONL.

    Expected JSONL format:
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "..."},
                    ...,
                    {"type": "text", "text": "question or instruction"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "reference report"}
                ]
            }
        ]
    }

    We convert this into a list of dicts with keys:
      - "prompt": single user message (chat format) with image placeholders
      - "images": list of image paths
      - "references": list[str], ground truth report text
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            messages = ex.get("messages", [])
            if not messages:
                continue

            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
            if user_msg is None or assistant_msg is None:
                continue

            user_content = user_msg.get("content", [])
            assistant_content = assistant_msg.get("content", [])

            # Extract image paths from user content
            images = [
                c.get("image")
                for c in user_content
                if isinstance(c, dict) and c.get("type") == "image" and c.get("image")
            ]

            # Extract first user text as the question
            user_texts = [
                c.get("text", "")
                for c in user_content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            prompt_text = user_texts[0] if user_texts else ""

            # Extract assistant reference text
            ref_text_parts = []
            if isinstance(assistant_content, list):
                for c in assistant_content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        ref_text_parts.append(c.get("text", ""))
                    elif isinstance(c, str):
                        ref_text_parts.append(c)
            elif isinstance(assistant_content, str):
                ref_text_parts.append(assistant_content)
            reference_text = "\n".join(p for p in ref_text_parts if p).strip()

            if not reference_text:
                continue

            # Build chat-style prompt
            content = []
            for _ in images:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt_text})

            prompt = [
                {
                    "role": "user",
                    "content": content,
                }
            ]

            items.append(
                {
                    "prompt": prompt,
                    "images": images,
                    "references": [reference_text],
                }
            )

    print(f"[data] RL items loaded: {len(items)}")
    return items


# ============================================================
# Reward helpers
# ============================================================

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _flatten_reference_texts(references, num_completions):
    """
    Normalize `references` into a list of plain strings aligned with `completions`.
    """
    if not isinstance(references, (list, tuple)):
        return [str(references)] * num_completions

    if len(references) == num_completions:
        base_list = list(references)
    elif len(references) == 1:
        base_list = [references[0]] * num_completions
    else:
        base_list = list(references)
        if len(base_list) < num_completions:
            base_list = base_list + [base_list[-1]] * (num_completions - len(base_list))
        elif len(base_list) > num_completions:
            base_list = base_list[:num_completions]

    texts = []
    for r in base_list:
        if isinstance(r, str):
            texts.append(r)
        elif isinstance(r, (list, tuple)) and r:
            inner = r[0]
            texts.append(inner if isinstance(inner, str) else str(inner))
        else:
            texts.append(str(r))
    return texts


def _extract_sections(text):
    """
    Return a dict with 'findings' / 'impression' sections if present.
    Looks for headings like "Findings:" or "Impression:".
    """
    if not isinstance(text, str):
        text = str(text)

    findings_re = re.compile(r'^\s*findings?\s*[:\-]', re.IGNORECASE | re.MULTILINE)
    impression_re = re.compile(
        r'^\s*impressions?\s*[:\-]|^\s*impression\s*[:\-]',
        re.IGNORECASE | re.MULTILINE,
    )

    sections = {}
    length = len(text)

    f_match = findings_re.search(text)
    i_match = impression_re.search(text)

    def slice_section(start_match, other_match):
        if not start_match:
            return ""
        start = start_match.end()
        end_candidates = []
        if other_match and other_match.start() > start:
            end_candidates.append(other_match.start())
        end = min(end_candidates) if end_candidates else length
        return text[start:end].strip()

    if f_match:
        sections["findings"] = slice_section(f_match, i_match)
    if i_match:
        sections["impression"] = slice_section(i_match, f_match)

    return sections


def _split_sentences(s):
    s = re.sub(r'\s+', ' ', s.strip())
    if not s:
        return []
    parts = re.split(r'(?<=[\.!?])\s+', s)
    return [p.strip() for p in parts if p.strip()]


def format_reward_func(prompts, completions, references, **kwargs):
    """
    Reward basic formatting and section usage.
    Encourages Findings/Impression sections and penalizes debug artifacts.
    """
    scores = []
    for c in completions:
        text = c if isinstance(c, str) else str(c)

        score = 0.0

        # Length sanity
        n_tokens = len(text.strip().split())
        if 20 <= n_tokens <= 400:
            score += 0.3
        elif n_tokens < 10:
            score -= 0.3

        sections = _extract_sections(text)
        f_text = sections.get("findings", "")
        i_text = sections.get("impression", "")

        has_f = bool(f_text)
        has_i = bool(i_text)

        if has_f and has_i:
            score += 0.7
        elif has_f or has_i:
            score += 0.3

        # Penalize debug artefacts
        if "<REASONING>" in text:
            score -= 1.0
        if "addCriterion" in text:
            score -= 1.0

        # Encourage non identical sections
        if has_f and has_i:
            f_sents = set(_split_sentences(f_text.lower()))
            i_sents = set(_split_sentences(i_text.lower()))
            if f_sents and i_sents:
                jaccard = len(f_sents & i_sents) / max(1, len(f_sents | i_sents))
                score += 0.6 * (1.0 - jaccard)

        score = max(min(score, 2.0), -2.0)
        scores.append(score)

    return scores


def rouge_reward_func(prompts, completions, references, **kwargs):
    """
    ROUGE-L between generation and reference report.
    """
    scores = []
    ref_texts = _flatten_reference_texts(references, len(completions))
    for gen, ref in zip(completions, ref_texts):
        g = gen if isinstance(gen, str) else str(gen)
        r = ref if isinstance(ref, str) else str(ref)
        try:
            val = rouge.score(r, g)["rougeL"].fmeasure
        except Exception:
            val = 0.0
        scores.append(float(val))
    return scores


MC_LABELS = set("ABCDEF")


def _is_mc_reference(ref_text: str) -> bool:
    """
    Check if reference is a single choice like A/B/C/D/E/F.
    """
    if not isinstance(ref_text, str):
        ref_text = str(ref_text)
    ref_text = ref_text.strip().upper()
    return len(ref_text) == 1 and ref_text in MC_LABELS


def _is_numeric_reference(ref_text: str) -> bool:
    """
    Check if reference is numeric (for ABR style questions).
    Allows "72", "73.5", "80%".
    """
    if not isinstance(ref_text, str):
        ref_text = str(ref_text)
    ref_text = ref_text.strip()

    if ref_text.endswith("%"):
        ref_text = ref_text[:-1].strip()

    try:
        float(ref_text)
        return True
    except ValueError:
        return False


def _extract_first_float(text: str):
    """
    Extract first float from a text string.
    """
    if not isinstance(text, str):
        text = str(text)

    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def clinical_reward_func(prompts, completions, references, **kwargs):
    """
    Factual correctness reward via F1-RadGraph.

    For MC or numeric questions we skip RadGraph.
    Otherwise we call the global `f1radgraph` scorer.
    """
    num_completions = len(completions)
    if num_completions == 0:
        return []

    ref_texts = _flatten_reference_texts(references, num_completions)

    hyps_all = [c if isinstance(c, str) else str(c) for c in completions]
    refs_all = [r if isinstance(r, str) else str(r) for r in ref_texts]

    scores = [0.0] * num_completions

    hyps_rg = []
    refs_rg = []
    positions = []

    for i, (hyp, ref) in enumerate(zip(hyps_all, refs_all)):
        ref_str = str(ref).strip()

        if _is_mc_reference(ref_str) or _is_numeric_reference(ref_str):
            continue

        hyps_rg.append(hyp)
        refs_rg.append(ref)
        positions.append(i)

    if not hyps_rg:
        return scores

    n = len(hyps_rg)

    try:
        mean_reward, reward_list, _, _ = f1radgraph(hyps=hyps_rg, refs=refs_rg)
        rg_scores = []
        for tpl in reward_list:
            if isinstance(tpl, (list, tuple)) and len(tpl) >= 2:
                rg_scores.append(float(tpl[1]))
            else:
                rg_scores.append(float(tpl))
    except TypeError:
        # fallback for older APIs
        try:
            result = f1radgraph(refs_rg, hyps_rg)
            rg_scores = []
            for tpl in result:
                if isinstance(tpl, (list, tuple)) and len(tpl) >= 2:
                    rg_scores.append(float(tpl[1]))
                else:
                    rg_scores.append(float(tpl))
        except Exception:
            rg_scores = [0.0] * n
    except Exception:
        rg_scores = [0.0] * n

    if len(rg_scores) < n:
        rg_scores = rg_scores + [0.0] * (n - len(rg_scores))
    elif len(rg_scores) > n:
        rg_scores = rg_scores[:n]

    for pos, val in zip(positions, rg_scores):
        scores[pos] = float(val)

    return scores


def combined_reward(prompts, completions, references, **kwargs):
    """
    Main reward: distinguishes MCQ, numeric questions, and free text reports.
    Uses:
      - clinical reward (RadGraph)
      - format reward
      - ROUGE reward
    """
    f_scores = format_reward_func(prompts, completions, references, **kwargs)
    r_scores = rouge_reward_func(prompts, completions, references, **kwargs)
    c_scores = clinical_reward_func(prompts, completions, references, **kwargs)

    ref_texts = _flatten_reference_texts(references, len(completions))
    final_scores = []

    for comp, f1, r1, c1, ref_raw in zip(completions, f_scores, r_scores, c_scores, ref_texts):
        ref_str = ref_raw if isinstance(ref_raw, str) else str(ref_raw)
        ref_str = ref_str.strip()
        comp_text = comp if isinstance(comp, str) else str(comp)

        # 1) Multiple choice question
        if _is_mc_reference(ref_str):
            comp_upper = comp_text.upper()
            m = re.search(r"\b([A-F])\b", comp_upper)
            pred_label = m.group(1) if m else None

            if pred_label == ref_str.upper():
                score = 1.0
            else:
                score = -0.2

        # 2) Numeric question
        elif _is_numeric_reference(ref_str):
            ref_clean = ref_str
            if ref_clean.endswith("%"):
                ref_clean = ref_clean[:-1].strip()
            try:
                gold = float(ref_clean)
            except ValueError:
                gold = None

            pred = _extract_first_float(comp_text)

            if gold is None or pred is None:
                value_term = -0.5
            else:
                abs_err = abs(pred - gold)

                if abs_err <= 2:
                    value_term = 1.0
                elif abs_err <= 5:
                    value_term = 0.5
                elif abs_err <= 10:
                    value_term = 0.0
                else:
                    value_term = -0.5

                if pred is not None and (pred < 0 or pred > 100):
                    value_term -= 0.3

            n_tokens = len(comp_text.strip().split())
            if n_tokens >= 15:
                len_bonus = 0.2
            elif n_tokens <= 3:
                len_bonus = -0.2
            else:
                len_bonus = 0.0

            fmt_term = f1 + len_bonus
            score = 0.7 * value_term + 0.3 * fmt_term

        # 3) Regular chest report
        else:
            # Full weighting: 0.7 clinical, 0.2 format, 0.1 ROUGE
            score = 0.7 * c1 + 0.2 * f1 + 0.1 * r1

        score = max(min(score, 2.0), -2.0)
        final_scores.append(float(score))

    return final_scores


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--sft_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--wandb_project", type=str, default="flare_task5_rl")
    parser.add_argument("--wandb_run_name", type=str, default="qwen3vl_rl")
    parser.add_argument("--report_to", type=str, default="wandb", choices=["none", "wandb"])

    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=512)
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.wandb_project
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    print("=== Loading SFT checkpoint ===")
    loaded = FastVisionModel.from_pretrained(
        args.sft_checkpoint,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        use_gradient_checkpointing="unsloth",
    )

    if isinstance(loaded, tuple):
        if len(loaded) == 2:
            model, tokenizer = loaded
            image_processor = None
        elif len(loaded) == 3:
            model, tokenizer, image_processor = loaded
        else:
            raise RuntimeError("Unknown return format from FastVisionModel.")
    else:
        model = loaded
        tokenizer = None
        image_processor = None

    if image_processor is None:
        print("SFT checkpoint missing image processor - loading from base model...")
        base = FastVisionModel.from_pretrained("unsloth/Qwen3-VL-4B-Instruct")
        if isinstance(base, tuple) and len(base) == 3:
            _, tokenizer_base, image_processor = base
        elif isinstance(base, tuple) and len(base) == 2:
            _, tokenizer_base = base
            if hasattr(tokenizer_base, "image_processor"):
                image_processor = tokenizer_base.image_processor
            else:
                raise RuntimeError("Base model tokenizer has no image_processor.")
        else:
            raise RuntimeError("Base model returned unexpected values.")

    print("=== Loading RL dataset ===")
    train_data = load_rl_dataset(args.train_jsonl)

    # Apply chat template to prompts
    print("=== Applying chat templates ===")
    for ex in train_data:
        ex["prompt"] = tokenizer.apply_chat_template(
            ex["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )

    print("=== Building GRPOConfig ===")
    training_args = GRPOConfig(
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.1,
        num_generations=2,
        max_prompt_length=args.max_seq_length,
        max_completion_length=args.max_completion_length,
        importance_sampling_level="sequence",
        loss_type="dr_grpo",
        bf16=is_bf16_supported(),
        max_steps=args.max_steps,
        report_to=args.report_to,
        output_dir=args.output_dir,
        save_steps=100,
        optim="adamw_torch",
    )

    print("=== Initializing GRPOTrainer ===")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[combined_reward],
        train_dataset=train_data,
    )

    print("=== Starting RL training ===")
    trainer.train()

    print("=== Saving RL model ===")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    image_processor.save_pretrained(args.output_dir)

    print(f"[DONE] RL saved to {args.output_dir}")


if __name__ == "__main__":
    main()
