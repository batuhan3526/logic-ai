#!/usr/bin/env python3
"""Stage 1: Domain-Adaptive Pretraining (DAPT) with LoRA on Llama-3.2-3B-Instruct.

Tokenizes financial texts, packs them into fixed-length windows, and trains
a LoRA adapter using causal language modelling loss.

Usage:
    accelerate launch --config_file configs/accelerate_2gpu.yaml scripts/run_dapt.py
"""
import json
import logging

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

from scripts.utils import (
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
# Dataset
# ---------------------------------------------------------------------------

class PackedTextDataset(Dataset):
    """Tokenise texts from a JSONL file and pack into fixed-length windows.

    Each JSONL line must contain a ``"text"`` field.  All texts are
    concatenated (separated by double newlines) and then sliced into
    non-overlapping windows of ``context_length`` tokens.
    """

    def __init__(
        self,
        text_file: str,
        tokenizer,
        context_length: int = 2048,
        max_tokens: int | None = None,
    ):
        all_ids: list[int] = []
        sep_ids = tokenizer.encode("\n\n", add_special_tokens=False)

        with open(text_file, "r") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text", "")
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                if ids:
                    all_ids.extend(ids)
                    all_ids.extend(sep_ids)
                if max_tokens and len(all_ids) >= max_tokens:
                    all_ids = all_ids[:max_tokens]
                    break

        total_tokens = len(all_ids)
        self.samples: list[list[int]] = []
        for start in range(0, total_tokens - context_length + 1, context_length):
            chunk = all_ids[start : start + context_length]
            self.samples.append(chunk)

        logger.info(
            "PackedTextDataset: %d tokens -> %d samples of %d tokens",
            total_tokens,
            len(self.samples),
            context_length,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Load config -------------------------------------------------------
    config = load_config("dapt_config.yaml")
    base_config = load_config("base_config.yaml")

    output_dir = str(BASE_DIR / config["output_dir"])
    final_adapter_path = str(BASE_DIR / config["output_dir"] / "final_adapter")

    # ---- Check if already complete ----------------------------------------
    if stage_complete(final_adapter_path):
        logger.info("DAPT already complete (found %s). Skipping.", final_adapter_path)
        return

    # ---- Resolve paths -----------------------------------------------------
    data_path = str(BASE_DIR / config["data_path"])
    eval_path = str(BASE_DIR / config["eval_path"])

    lora_cfg = config["lora"]
    train_cfg = config["training"]
    model_name = base_config["model"]["name"]
    context_length = train_cfg["max_seq_length"]

    # ---- Load tokenizer & model -------------------------------------------
    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=base_config["model"]["attn_implementation"],
        trust_remote_code=True,
    )

    # ---- Apply LoRA -------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Datasets ----------------------------------------------------------
    logger.info("Loading training data from %s", data_path)
    train_dataset = PackedTextDataset(data_path, tokenizer, context_length=context_length)

    eval_dataset = None
    if eval_path:
        logger.info("Loading eval data from %s", eval_path)
        eval_dataset = PackedTextDataset(eval_path, tokenizer, context_length=context_length)

    # ---- Training arguments ------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        bf16=base_config["hardware"]["bf16"],
        gradient_checkpointing=base_config["hardware"]["gradient_checkpointing"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=train_cfg.get("eval_steps") if eval_dataset else None,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )

    # ---- Resume from checkpoint --------------------------------------------
    resume_ckpt = find_latest_checkpoint(output_dir)
    if resume_ckpt:
        logger.info("Resuming from checkpoint: %s", resume_ckpt)
    else:
        logger.info("No checkpoint found; starting from scratch.")

    # ---- Train -------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ---- Save final adapter ------------------------------------------------
    logger.info("Saving final adapter to %s", final_adapter_path)
    model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    logger.info("DAPT training complete.")


if __name__ == "__main__":
    main()
