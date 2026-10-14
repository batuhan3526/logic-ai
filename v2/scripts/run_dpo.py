#!/usr/bin/env python3
"""Stage 3: Direct Preference Optimization (DPO) training with LoRA.

Loads a merged SFT model, applies a fresh LoRA adapter, and trains with
DPOTrainer from TRL using preference pairs (chosen / rejected).

Usage:
    accelerate launch --config_file configs/accelerate_2gpu.yaml \
        scripts/run_dpo.py --config baseline_dpo_config.yaml

    accelerate launch --config_file configs/accelerate_2gpu.yaml \
        scripts/run_dpo.py --config logic_dpo_config.yaml
"""
import argparse
import json
import logging

import torch
from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from trl import DPOConfig, DPOTrainer

from v2.scripts.utils import (
    BASE_DIR,
    find_latest_checkpoint,
    load_config,
    stage_complete,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dpo_dataset(path: str) -> HFDataset:
    """Load a JSONL file with DPO preference pairs into a HuggingFace Dataset.

    Expected line format::

        {"prompt": str, "chosen": str, "rejected": str}

    Rows with missing or empty required fields are skipped with a warning.
    """
    prompts, chosens, rejecteds = [], [], []
    skipped = 0
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = row.get("prompt", "")
            chosen = row.get("chosen", "")
            rejected = row.get("rejected", "")
            if not prompt or not chosen or not rejected:
                skipped += 1
                continue
            prompts.append(prompt)
            chosens.append(chosen)
            rejecteds.append(rejected)

    if skipped:
        logger.warning("Skipped %d rows with missing fields from %s", skipped, path)
    logger.info("Loaded %d preference pairs from %s", len(prompts), path)
    return HFDataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DPO training for Llama-3.1.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Name of the DPO config YAML file (e.g. baseline_dpo_config.yaml).",
    )
    args = parser.parse_args()

    # ---- Load configs ------------------------------------------------------
    config = load_config(args.config)
    base_config = load_config("base_config.yaml")

    output_dir = str(BASE_DIR / config["output_dir"])
    final_adapter_path = str(BASE_DIR / config["output_dir"] / "final_adapter")

    # ---- Check if already complete ----------------------------------------
    if stage_complete(final_adapter_path):
        logger.info(
            "DPO stage '%s' already complete (found %s). Skipping.",
            config["stage"],
            final_adapter_path,
        )
        return

    # ---- Resolve paths -----------------------------------------------------
    data_path = str(BASE_DIR / config["data_path"])
    eval_path = str(BASE_DIR / config["eval_path"])
    base_model_path = str(BASE_DIR / config["base_model_path"])

    lora_cfg = config["lora"]
    train_cfg = config["training"]
    max_seq_length = train_cfg["max_seq_length"]

    # ---- Load tokenizer ----------------------------------------------------
    logger.info("Loading tokenizer from: %s", base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load policy model -------------------------------------------------
    logger.info("Loading policy model from: %s", base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=base_config["model"]["attn_implementation"],
        trust_remote_code=True,
    )

    # ---- Apply LoRA to policy model ----------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # NOTE: No explicit reference model is loaded.  When the policy model
    # is a PeftModel and ref_model=None, DPOTrainer automatically computes
    # reference logits from the frozen base weights of the PEFT model.
    # This saves ~8 GB VRAM -- critical on 2x RTX 3090 (24 GB each).

    # ---- Datasets ----------------------------------------------------------
    logger.info("Loading training data from %s", data_path)
    train_dataset = load_dpo_dataset(data_path)

    eval_dataset = None
    if eval_path:
        logger.info("Loading eval data from %s", eval_path)
        eval_dataset = load_dpo_dataset(eval_path)

    # ---- DPO training config -----------------------------------------------
    dpo_training_args = DPOConfig(
        output_dir=output_dir,
        max_steps=train_cfg.get("max_steps", -1),
        num_train_epochs=train_cfg.get("epochs", 1),
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        beta=train_cfg["beta"],
        max_length=max_seq_length,
        max_prompt_length=train_cfg.get("max_prompt_length", max_seq_length // 2),
        bf16=base_config["hardware"]["bf16"],
        gradient_checkpointing=base_config["hardware"]["gradient_checkpointing"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=train_cfg.get("eval_steps") if eval_dataset else None,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # ---- Resume from checkpoint --------------------------------------------
    resume_ckpt = find_latest_checkpoint(output_dir)
    if resume_ckpt:
        logger.info("Resuming from checkpoint: %s", resume_ckpt)
    else:
        logger.info("No checkpoint found; starting from scratch.")

    # ---- Train with DPOTrainer ---------------------------------------------
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ---- Save final adapter ------------------------------------------------
    logger.info("Saving final adapter to %s", final_adapter_path)
    model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    logger.info("DPO stage '%s' complete.", config["stage"])


if __name__ == "__main__":
    main()
