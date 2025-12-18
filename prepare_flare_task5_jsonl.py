#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import zipfile
from typing import List, Dict, Any, Optional
import glob


# tool functions

def unzip_all_zips(data_root: str) -> None:
    """recursion unpack all zip under root_path.
    """
    print(f"[unzip] scan and unpack zip file under {data_root}...")
    for root, dirs, files in os.walk(data_root):
        for fname in files:
            if not fname.lower().endswith(".zip"):
                continue
            zip_path = os.path.join(root, fname)
            # remove .zip
            base = fname[:-4]
            dest_dir = os.path.join(root, base)
            if os.path.isdir(dest_dir) and os.listdir(dest_dir):
                print(f"[unzip] if non-empty dir exist，skip: {dest_dir}")
                continue
            print(f"[unzip] unzip {zip_path} -> {dest_dir}")
            os.makedirs(dest_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
    print("[unzip] unpack completed")


# tool function, get sample from json

def load_samples_from_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        for key in ["data", "samples", "annotations", "items", "questions"]:
            v = raw.get(key, None)
            if isinstance(v, list):
                return v

    raise ValueError(f"cant from {json_path} deduct sample structure，please manually check json format")


# tool function, guess phrase name

def find_key(d: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    if not isinstance(d, dict):
        return None

    keys = list(d.keys())
    lower_map = {k.lower(): k for k in keys}

    # 1. try exact match
    for c in candidates:
        if c in d:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    # 2. try inclusion match
    for k in keys:
        kl = k.lower()
        for c in candidates:
            if c.lower() in kl:
                return k

    return None

def extract_image_names(ex: Dict[str, Any]) -> List[str]:

    img_key_candidates = [
        "image", "image_path", "image_name", "img_name",
        "img", "imageid", "image_id", "file_name", "filename", "path", "imagename"
    ]
    key = find_key(ex, img_key_candidates)
    value = ex.get(key) if key is not None else None

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        names: List[str] = []
        for v in value:
            if isinstance(v, str):
                names.append(v)
            elif isinstance(v, dict):

                k2 = find_key(v, img_key_candidates)
                if k2 is not None and isinstance(v[k2], str):
                    names.append(v[k2])
        if names:
            return names

    if isinstance(value, dict):
        k2 = find_key(value, img_key_candidates)
        v2 = value.get(k2) if k2 is not None else None
        if isinstance(v2, str):
            return [v2]
        if isinstance(v2, list):
            names: List[str] = []
            for v in v2:
                if isinstance(v, str):
                    names.append(v)
                elif isinstance(v, dict):
                    k3 = find_key(v, img_key_candidates)
                    if k3 is not None and isinstance(v[k3], str):
                        names.append(v[k3])
            if names:
                return names

    raise ValueError(f"cant deduct string: {ex}")



def extract_text_field(ex: Dict[str, Any], kind: str) -> str:
    """extract question / answer."""
    if kind == "question":
        cand = ["question", "Question", "instruction", "query", "q", "prompt"]
    elif kind == "answer":
        cand = ["answer", "Answer", "gt_answer", "response", "label",
                "A", "target", "output", "text"]
    else:
        raise ValueError(f"kind must be question or answer, now is {kind}")

    key = find_key(ex, cand)
    value = ex.get(key) if key is not None else None

    if isinstance(value, str):
        return value

    if kind == "answer" and isinstance(value, (int, float, bool)):

        return str(value)
       
    if isinstance(value, list):
        parts = []
        for v in value:
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, dict):
                tkey = find_key(v, ["text", "Text", "answer", "Answer"])
                if tkey is not None and isinstance(v[tkey], str):
                    parts.append(v[tkey])
        if parts:
            return " ".join(parts)

    raise ValueError(f"cant extract {kind}: {ex} from samples")



def resolve_image_path(img_dir: str, image_name: str) -> Optional[str]:
    """based on image_name and img_dir deduct images' absolute path."""
    if os.path.isabs(image_name) and os.path.exists(image_name):
        return image_name

    # merge
    candidate = os.path.join(img_dir, image_name)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    candidate2 = os.path.join(img_dir, os.path.basename(image_name))
    if os.path.exists(candidate2):
        return os.path.abspath(candidate2)

    # safety mechanism，use glob to recursion search under img_dir
    matches = glob.glob(os.path.join(img_dir, "**", os.path.basename(image_name)), recursive=True)
    if matches:
        return os.path.abspath(matches[0])

    return None


# Main

def convert_dataset(
    json_path: str,
    img_dir: str,
    out_f,
    split: str,
    dataset_name: str,
) -> None:
    print(f"[convert] deal {split} set: {dataset_name}")
    print(f"          json: {json_path}")
    print(f"          img_dir: {img_dir}")

    if not os.path.isfile(json_path):
        print(f"[warn] cant find json file，skip: {json_path}")
        return
    if not os.path.isdir(img_dir):
        print(f"[warn] cant find image dir，skip: {img_dir}")
        return

    samples = load_samples_from_json(json_path)
    print(f"[convert] 读入样本数: {len(samples)}")

    n_ok = 0
    n_skip = 0
    for ex in samples:
        try:
            image_names = extract_image_names(ex)
            question = extract_text_field(ex, "question")
            answer = extract_text_field(ex, "answer")

            img_paths: List[str] = []
            for name in image_names:
                p = resolve_image_path(img_dir, name)
                if p is not None:
                    img_paths.append(p)

            if not img_paths:
                print(f"[warn] cant find any images: {image_names} (json={json_path})，sample skipped")
                n_skip += 1
                continue

            # construct Qwen3-VL content

            user_content = []
            for p in img_paths:
                user_content.append({"type": "image", "image": p})

            # plus text question
            user_content.append({"type": "text", "text": question})

            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_content,
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer},
                        ],
                    },
                ]
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_ok += 1


        except Exception as e:
            print(f"[warn] error on single sample，skip: {e}")
            n_skip += 1


    print(f"[convert] {dataset_name} transformation completed: wrote {n_ok} sampels, skip {n_skip} samples")


# ------------------------ main ------------------------ #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="FLARE_Task5 root，such as /scratch/FLARE_Task5")
    parser.add_argument("--train_output", type=str, required=True,
                        help="train set output jsonl path")
    parser.add_argument("--val_output", type=str, required=True,
                        help="val set output jsonl path")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = os.path.abspath(args.data_root)

    # 1. unzip all .zip
    unzip_all_zips(data_root)

    # 2. define dataset (relative data_root path）
    DATASETS = [
        # ---- train ----
        {
            "split": "train",
            "name": "IU_XRay_train",
            "json_rel": "train/training/Xray/IU_XRay/IU_XRay_all_train.json",
            "img_dir_rel": "train/training/Xray/IU_XRay/imagesTr",
        },
        {
            "split": "train",
            "name": "boneresorption",
            "json_rel": "train/training/Xray/boneresorption/boneresorption_questions_train.json",
            "img_dir_rel": "train/training/Xray/boneresorption/imagesTr",
        },
        {
            "split": "train",
            "name": "chestdr",
            "json_rel": "train/training/Xray/chestdr/chestdr_questions_train.json",
            "img_dir_rel": "train/training/Xray/chestdr/imagesTr",
        },
        {
            "split": "train",
            "name": "dental",
            "json_rel": "train/training/Xray/dental/dental_questions_train.json",
            "img_dir_rel": "train/training/Xray/dental/imagesTr",
        },
        {
            "split": "train",
            "name": "periapical",
            "json_rel": "train/training/Xray/periapical/periapical_questions_train.json",
            "img_dir_rel": "train/training/Xray/periapical/imagesTr",
        },
        # ---- val ----
        {
            "split": "val",
            "name": "IU_XRay_val",
            "json_rel": "val/validation-public/Xray/IU_XRay/IU_XRay_all_val.json",
            "img_dir_rel": "val/validation-public/Xray/IU_XRay/imagesVal",
        },
    ]

    # 3. open output file
    os.makedirs(os.path.dirname(args.train_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.val_output), exist_ok=True)

    train_f = open(args.train_output, "w", encoding="utf-8")
    val_f = open(args.val_output, "w", encoding="utf-8")

    try:
        for ds in DATASETS:
            json_path = os.path.join(data_root, ds["json_rel"])
            img_dir = os.path.join(data_root, ds["img_dir_rel"])

            if ds["split"] == "train":
                out_f = train_f
            else:
                out_f = val_f

            convert_dataset(
                json_path=json_path,
                img_dir=img_dir,
                out_f=out_f,
                split=ds["split"],
                dataset_name=ds["name"],
            )
    finally:
        train_f.close()
        val_f.close()

    print("[done] all transformation done")
    print(f"train set wrote: {args.train_output}")
    print(f"val set wrote: {args.val_output}")


if __name__ == "__main__":
    main()
