#!/usr/bin/env python3
# MIT ⓒ2024 Joona Kytöniemi
# Translate ORPO-formatted data from English to Finnish using Poro

import os
import sys
import json
import torch
from argparse import ArgumentParser, FileType
from transformers import AutoTokenizer, AutoModelForCausalLM

default_tokenizer = "LumiOpen/Poro-34B"
default_model = "LumiOpen/Poro-34B"


def argparser():
    ap = ArgumentParser()
    ap.add_argument("--input", type=FileType('r'), metavar='F', required=True)
    # ap.add_argument("--output", type=FileType('w+'), metavar='F', required=True)
    ap.add_argument("--tokenizer", type=str, default=default_tokenizer)
    ap.add_argument("--model", type=str, default=default_model)
    ap.add_argument("--dry_run", action="store_true")
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    en_samples = []
    for line in args.input:
        en_samples.append(json.loads(line))

    if not args.dry_run:
        supports_fa: bool = torch.cuda.get_device_capability()[0] >= 8
        attn_implementation = "flash_attention_2" if supports_fa and args.flash_attn else "eager"

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto",
                                                     attn_implementation=attn_implementation)

        template = "<|user|>Käännä suomeksi: {} <|assistant|>"




if __name__ == "__main__":
    sys.exit(main(sys.argv))
