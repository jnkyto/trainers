#!/usr/bin/env python3
# MIT ⓒ2024 Joona Kytöniemi
# Multi-node ORPO Trainer for instructional datasets

import os
import sys
import random

from datetime import datetime
from argparse import ArgumentParser

from accelerate.utils import set_seed
from datasets import load_dataset, Dataset
from trl import ORPOConfig, ORPOTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding
)

default_model = 'LumiOpen/Poro-34B'
curr_date = str(datetime.now().isoformat("T", "minutes")).replace(':', '')
default_model_save_dir = "/scratch/project_462000615/kytoniem/models/orpo"  # without trailing forward-slash


def argparser():
    ap = ArgumentParser()
    ap.add_argument("--input_data", type=str, required=True)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_length", type=int, default=8192)
    ap.add_argument("--gradient_steps", type=int, default=1)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=0)   # these don't work so just don't use 'em :)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--model", type=str, default=default_model)
    ap.add_argument("--tokenizer", type=str, default=default_model)
    ap.add_argument("--dry_run", action="store_true")
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])
    set_seed(args.seed)

    ds = load_dataset("json", data_files=args.input_data)["train"]

    ds.shuffle(random.seed(args.seed))
    ds = ds.train_test_split(test_size=0.15)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(ds)

    if not args.dry_run:
        train_args = ORPOConfig(
            beta=0.1,  # The lambda/alpha hyperparameter in the paper/code
            max_length=256,
            max_prompt_length=96,
            output_dir="./out/train_out",
            warmup_steps=args.warmup_steps,
            logging_steps=args.logging_steps,

            eval_steps=args.eval_steps,
            evaluation_strategy="steps",

            save_strategy="steps" if args.save_steps != 0 else "no",
            save_steps=args.save_steps,
            save_total_limit=3,

            gradient_checkpointing=True if args.gradient_steps >= 1 else False,
            gradient_accumulation_steps=args.gradient_steps,
            gradient_checkpointing_kwargs={'use_reentrant': False} if args.gradient_steps >= 1 else {},

            num_train_epochs=args.epochs,
            per_device_eval_batch_size=args.batch_size,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,

            lr_scheduler_type="constant_with_warmup",
            remove_unused_columns=False,
            bf16=True,
            bf16_full_eval=True,
            log_on_each_node=False,
            log_level="info",
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto"
        )

        if args.gradient_steps != 1:
            model.gradient_checkpointing_enable()

        collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            return_tensors='pt',
            padding="max_length"
        )

        trainer = ORPOTrainer(
            model=model,
            args=train_args,
            train_dataset=ds["train"],
            tokenizer=tokenizer,
            data_collator=collator,
            eval_dataset=ds["test"],
        )

        trainer.accelerator.print(f"DeepSpeed info:\n{trainer.deepspeed}")

        # Torch barrier before training start
        trainer.accelerator.wait_for_everyone()
        trainer.train()

        # Torch barrier before unwrap and save
        trainer.accelerator.wait_for_everyone()
        # * save function to be added when it's actually necessary *


if __name__ == "__main__":
    sys.exit(main(sys.argv))
