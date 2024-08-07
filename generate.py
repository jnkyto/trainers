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
    ap.add_argument("--penalty", type=float, default=1.1)
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    supports_fa: bool = torch.cuda.get_device_capability()[0] >= 8
    attn_implementation = "flash_attention_2" if supports_fa and args.flash_attn else "eager"

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto",
                                                 attn_implementation=attn_implementation)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    chat_template = "<|im_start|>system\nToimit henkilökohtaisena assistenttina. Tehtäväsi on vastata seuraavaan viestiin parhaan tietämyksesi mukaan. Vastaa viestiin kattavasti, mutta älä kuitenkaan toista itseäsi tai kirjoita mitään ylimääräistä. Kerro vastauksesi seuraavan indikaattorin jälkeen.\n<|im_start|>user\n{}\n<|im_start|>assistant\n"

    while True:
        usr_in = input("Message: ")
        prompt = chat_template.format(usr_in)
        decoded = tokenizer.decode(
            model.generate(**tokenizer(prompt, return_tensors="pt").to("cuda"), max_new_tokens=128, do_sample=True,
                           top_k=30, repetition_penalty=args.penalty)[0])
        print(str(decoded).split("\n<|im_start|>assistant\n", 1)[1] + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
