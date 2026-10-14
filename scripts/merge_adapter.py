#!/usr/bin/env python3
"""Merge a LoRA adapter into base model weights.

Loads the base model on CPU, applies the adapter, merges the weights, and
saves the resulting full model + tokenizer to the output path.

Usage:
    python scripts/merge_adapter.py \
        --base-model meta-llama/Llama-3.2-3B-Instruct \
        --adapter output/dapt/final_adapter \
        --output output/dapt/merged_model
"""
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="HuggingFace model name or path to the base model.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to the LoRA adapter directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory where the merged model will be saved.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    # ---- Skip if output already exists ------------------------------------
    if output_path.exists() and any(output_path.iterdir()):
        logger.info(
            "Merged model already exists at %s. Skipping.", output_path
        )
        return

    # ---- Load base model on CPU -------------------------------------------
    logger.info("Loading base model: %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    # ---- Load and merge adapter -------------------------------------------
    logger.info("Loading adapter from: %s", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)

    logger.info("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    # ---- Save merged model + tokenizer ------------------------------------
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving merged model to: %s", output_path)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    logger.info("Merge complete.")


if __name__ == "__main__":
    main()
