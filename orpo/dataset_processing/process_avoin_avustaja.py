#!/usr/bin/env python3
# MIT ⓒ2024 Joona Kytöniemi
# Process raw Avoin Avustaja data to ORPO format

import sys
import jsonlines
from argparse import ArgumentParser, FileType

default_prefix = "<|user|>Toimit henkilökohtaisena assistenttina. Tehtäväsi on vastata seuraavaan viestiin parhaan tietämyksesi mukaan. Viesti: "
default_suffix = "<|assistant|>"


def argparse():
    ap = ArgumentParser()
    ap.add_argument("--input", type=FileType('r'), metavar='F', required=True)
    ap.add_argument("--output", type=FileType('w+'), metavar='F', required=True)
    ap.add_argument("--label_override", action="store_true")
    # TODO: Need to implement args in a different way where they can be freely accessed within every method
    # ap.add_argument("--prefix", type=str, default=default_prefix)
    # ap.add_argument("--suffix", type=str, default=default_suffix)
    return ap


def score_calc(reply):
    weights = {
        "spam": -1.0, "fails_task": -1.0, "lang_mismatch": -1.0, "pii": -1.0, "not_appropriate": -1.0,
        "hate_speech": -1.0, "sexual_content": -1.0, "quality": 2.0, "toxicity": -1.0, "humor": 0.5, "helpfulness": 2.0,
        "creativity": 1.0, "violence": -1.0
    }

    score = 0.0
    for category, rating in reply["labels"].items():
        score += weights.get(category, 0) * rating["value"]

    return score


def reply_sorter(assistant_replies, override=False):
    flag = False
    sorted_replies = []
    for reply in assistant_replies:
        if "rank" in reply and reply["rank"] is not None and not override:
            flag = True
            sorted_replies.append((reply, reply["rank"]))
        else:
            if not flag:
                reply_with_score = (reply, score_calc(reply))
                sorted_replies.append(reply_with_score)
            else:
                reply_sorter(assistant_replies, True)

    sorted_replies.sort(key=lambda x: x[1], reverse=False if flag else True)
    return sorted_replies


def extract_features(msg, label_override):
    prompt_replies = []
    if "replies" in msg and len(msg["replies"]) >= 2:
        assistant_replies = [reply for reply in msg["replies"] if reply["role"] == "assistant"]
        if len(assistant_replies) >= 2:
            assistant_replies = reply_sorter(assistant_replies, label_override)
            chosen_reply = assistant_replies[0]
            rejected_reply = assistant_replies[-1]
            prompt_replies.append({
                "prompt": f'{default_prefix}{msg["text"]}{default_suffix}',
                "chosen": chosen_reply[0]["text"],
                "rejected": rejected_reply[0]["text"]}
            )

    if "replies" in msg:
        for reply in msg["replies"]:
            prompt_replies.extend(extract_features(reply, reply["role"]))

    return prompt_replies


def level_loop(reader, label_override):
    samples = []
    for i, obj in enumerate(reader):
        if obj["prompt"]["role"] == "prompter":
            samples.extend(extract_features(obj["prompt"], label_override))
    return samples


def main(argv):
    args = argparse().parse_args(argv[1:])
    reader = jsonlines.Reader(args.input)
    samples = level_loop(reader, args.label_override)
    writer = jsonlines.Writer(args.output)
    writer.write_all(samples)
    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv))
