#!/usr/bin/env python3
# MIT ⓒ2024 Joona Kytöniemi
# Translate ORPO-formatted data from English to Finnish using Poro

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
    ap.add_argument("--output", type=FileType('w+'), metavar='F', required=True)
    ap.add_argument("--tokenizer", type=str, default=default_tokenizer)
    ap.add_argument("--model", type=str, default=default_model)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--test_break", type=int, default=0)
    ap.add_argument("--flash_attn", action="store_true")
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
        print(f"Tokenizer: {args.tokenizer}, Model {args.model} -> loaded successfully.")

        template = "<|user|>Käännä suomeksi: {} <|assistant|>"
        keys = ["prompt", "chosen", "rejected"]

        fi_samples = []
        for i, en_sample in enumerate(en_samples):
            print(f"Starting translation of sample {i} out of {len(en_samples)}.")
            fi_sample = {}
            for j, entry in enumerate(en_sample):
                prompt = template.format(entry)
                encoded = tokenizer(prompt, return_tensors='pt').to(model.device)
                output = model.generate(**encoded, max_length=256)
                decoded = tokenizer.decode(output[0])

                assert decoded.startswith(prompt)
                pred = decoded[len(prompt):]
                pred = pred.rstrip('\n')

                if pred.endswith(tokenizer.eos_token):
                    pred = pred[:-len(tokenizer.eos_token)]

                pred = pred.rstrip('\n')
                fi_sample[keys[j]] = pred
            print(fi_sample)
            fi_samples.append(fi_sample)

            if args.test_break != 0:
                if i == args.test_break:
                    break

        for line in fi_samples:
            args.output.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
