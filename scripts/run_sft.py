#!/usr/bin/env python3
"""Stage 2: Supervised Fine-Tuning (SFT) with LoRA on a merged DAPT model.

Applies Llama-3.1 chat template, masks non-assistant tokens so the loss
is computed only on assistant responses.

Usage:
    accelerate launch --config_file configs/accelerate_2gpu.yaml \
        scripts/run_sft.py --config baseline_sft_config.yaml

    accelerate launch --config_file configs/accelerate_2gpu.yaml \
        scripts/run_sft.py --config logic_sft_config.yaml
"""
import argparse
import json
import logging
from dataclasses import dataclass

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

class ChatSFTDataset(Dataset):
    """Apply the Llama-3.1 chat template and mask non-assistant tokens.

    Each JSONL line must contain a ``"messages"`` field with the standard
    chat format: ``[{"role": "...", "content": "..."}, ...]``.

    Labels are set to ``-100`` for all tokens that do NOT belong to
    assistant turns so the model only learns from its own responses.

    Token boundaries are determined by applying the chat template to
    progressively longer message prefixes and diffing the token counts,
    which avoids the fragile decode-and-re-encode round-trip.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples: list[dict] = []

        with open(data_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = row.get("messages")
                if not messages:
                    continue
                self.samples.append(messages)

        logger.info("ChatSFTDataset: loaded %d conversations from %s", len(self.samples), data_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self.samples[idx]

        # Tokenize the full conversation to get the ground-truth input_ids.
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )

        # Truncate to max_seq_length
        full_ids = full_ids[: self.max_seq_length]
        seq_len = len(full_ids)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones(seq_len, dtype=torch.long)

        # Build labels: start with -100 everywhere, then unmask assistant
        # segments by tokenizing progressively longer message prefixes and
        # diffing token counts to find each message's boundaries.
        labels = torch.full((seq_len,), -100, dtype=torch.long)

        prev_len = 0
        for turn_idx in range(len(messages)):
            # Tokenize messages[0..turn_idx] inclusive
            prefix_ids = self.tokenizer.apply_chat_template(
                messages[: turn_idx + 1],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            )
            cur_len = min(len(prefix_ids), seq_len)

            if messages[turn_idx]["role"] == "assistant":
                # Unmask this turn's tokens (assistant content + eot_id)
                labels[prev_len:cur_len] = input_ids[prev_len:cur_len]

            prev_len = cur_len
            if cur_len >= seq_len:
                break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

@dataclass
class PaddingCollator:
    """Pad input_ids, attention_mask, and labels to the longest sequence in a batch.

    Labels are padded with ``-100`` so padded positions are ignored by the loss.
    """

    tokenizer: object
    max_length: int | None = None

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].size(0) for f in features)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        pad_token_id = self.tokenizer.pad_token_id

        for f in features:
            seq_len = f["input_ids"].size(0)
            pad_len = max_len - seq_len

            if pad_len > 0:
                input_ids = torch.cat([f["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
                attention_mask = torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
                labels = torch.cat([f["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            else:
                input_ids = f["input_ids"][:max_len]
                attention_mask = f["attention_mask"][:max_len]
                labels = f["labels"][:max_len]

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for Llama-3.1.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Name of the SFT config YAML file (e.g. baseline_sft_config.yaml).",
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
            "SFT stage '%s' already complete (found %s). Skipping.",
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

    # ---- Load tokenizer & model -------------------------------------------
    logger.info("Loading base model from: %s", base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
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
    train_dataset = ChatSFTDataset(data_path, tokenizer, max_seq_length=max_seq_length)

    eval_dataset = None
    if eval_path:
        logger.info("Loading eval data from %s", eval_path)
        eval_dataset = ChatSFTDataset(eval_path, tokenizer, max_seq_length=max_seq_length)

    collator = PaddingCollator(tokenizer=tokenizer, max_length=max_seq_length)

    # ---- Training arguments ------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=train_cfg.get("weight_decay", 0.0),
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
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ---- Save final adapter ------------------------------------------------
    logger.info("Saving final adapter to %s", final_adapter_path)
    model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    logger.info("SFT stage '%s' complete.", config["stage"])


if __name__ == "__main__":
    main()
