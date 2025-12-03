#!/usr/bin/env python3
"""
Fine-tune CodeLlama (or other causal LM) on preprocessed Spider JSONL
Supports LoRA (PEFT) and QLoRA (bitsandbytes 4-bit + LoRA).

Example:
    python finetune_codellama_spider.py \
        --model_name codellama/CodeLlama-7b-Instruct-hf \
        --train_file data/train.jsonl \
        --eval_file data/dev.jsonl \
        --output_dir ./codellama-spider-lora \
        --use_qlora
"""

import argparse
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers.utils import logging
from transformers import DataCollatorForSeq2Seq
    # ---------------------------

# ---------------------------
# Dataset wrapper
# ---------------------------
class Seq2SQLDataset(torch.utils.data.Dataset):
    def __init__(self, records, tokenizer, max_length=2048, target_max_length=512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_max_length = target_max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        inp, tgt = rec["input"], rec["target"]

        tok_inp = self.tokenizer(
            inp,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        tok_tgt = self.tokenizer(
            tgt,
            truncation=True,
            max_length=self.target_max_length,
            return_attention_mask=False,
            add_special_tokens=False,
        )

        input_ids = tok_inp["input_ids"] + tok_tgt["input_ids"] + [self.tokenizer.eos_token_id]
        labels = [-100] * len(tok_inp["input_ids"]) + tok_tgt["input_ids"] + [self.tokenizer.eos_token_id]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------
# Helper: read JSONL
# ---------------------------
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_dir", default="./codellama-spider-lora")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--target_max_length", type=int, default=512)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--logging_steps", type=int, default=50)
    args = parser.parse_args()

    # ---------------------------
    # Tokenizer
    # ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---------------------------
    # Data
    # ---------------------------
    train_recs = read_jsonl(args.train_file)
    eval_recs = read_jsonl(args.eval_file)

    # ---------------------------
    # Model loading (LoRA / QLoRA)
    # ---------------------------
    if args.use_qlora:
        import bitsandbytes as bnb  # noqa
        compute_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[args.bnb_4bit_compute_dtype]

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )

    # ---------------------------
    # LoRA config
    # ---------------------------
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)

    # ---------------------------
    # Datasets
    # ---------------------------
    train_dataset = Seq2SQLDataset(train_recs, tokenizer, args.max_length, args.target_max_length)
    eval_dataset = Seq2SQLDataset(eval_recs, tokenizer, args.max_length, args.target_max_length)

    # ---------------------------
    # Training args
    # ---------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=not args.use_qlora and torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch" if args.save_steps == 0 else "steps",
        save_steps=args.save_steps if args.save_steps > 0 else None,
        save_total_limit=args.save_total_limit,
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        disable_tqdm=False,  # ✅ always show tqdm progress bar
    )

    # Ensure progress output is visible
    logging.set_verbosity_info()

    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,          # dynamically pad to longest in batch
        max_length=None,       # let it handle length automatically
        return_tensors="pt",
    )
    # Trainer
    # ---------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Run one evaluation pass before training (baseline metrics)
    metrics = trainer.evaluate()
    print(f"Initial evaluation before training: {metrics}")


    # ---------------------------
    # Train
    # ---------------------------
    trainer.train()

    # ---------------------------
    # Save model + tokenizer
    # ---------------------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n✅ Training complete. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

# python finetune_codellama.py  --model_name meta-llama/CodeLlama-7b-hf \
#   --train_file .././processed_spider/train.jsonl \
#   --eval_file .././processed_spider/dev.jsonl \
#   --output_dir .././outputs/codellama-spider-lora
