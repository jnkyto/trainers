#!/usr/bin/env python3
# MIT ⓒ2024 Joona Kytöniemi

import sys
import torch.cuda
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM

default_model = "LumiOpen/Poro-34B"
default_tokenizer = "LumiOpen/Poro-34B"


def argparser():
    ap = ArgumentParser()
    ap.add_argument("--model", type=str, default=default_model)
    ap.add_argument("--tokenizer", type=str, default=default_tokenizer)
    ap.add_argument("--flash_attn", action="store_true")
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    supports_fa: bool = torch.cuda.get_device_capability()[0] >= 8
    attn_implementation = "flash_attention_2" if supports_fa and args.flash_attn else "eager"

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto",
                                                 attn_implementation=attn_implementation)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    chat_template = "<|prompt|>Toimit henkilökohtaisena assistenttina. Tehtäväsi on vastata seuraavaan viestiin parhaan tietämyksesi mukaan. Viesti: {}<|response|>"

    while True:
        usr_in = input("Message: ")
        prompt = chat_template.format(usr_in)
        decoded = tokenizer.decode(
            model.generate(**tokenizer(prompt, return_tensors="pt").to("cuda"), max_new_tokens=192)[0], do_sample=True,
            top_k=30, repetition_penalty=2, length_penalty=1.25)
        print(str(decoded).split("<|response|>", 1)[1] + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
