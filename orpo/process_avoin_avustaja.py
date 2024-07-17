# ⓒ 2024 Joona Kytöniemi
# Process raw oasst to ORPO format

import sys
import jsonlines
from argparse import ArgumentParser, FileType


def argparse():
    ap = ArgumentParser()
    ap.add_argument("--input", type=FileType('r'), metavar='F')
    return ap


def extract_features(msg, parent_role=None):
    prompt_replies = []
    if "replies" in msg and len(msg["replies"]) >= 2:
        assistant_replies = [reply for reply in msg["replies"] if reply["role"] == "assistant"]
        if len(assistant_replies) >= 2:
            # TODO: Use label scores to rank replies if explicit rank is not available
            assistant_replies.sort(key=lambda x: x.get("rank", 0))
            chosen_reply = assistant_replies[0]
            rejected_reply = assistant_replies[-1]
            prompt_replies.append({
                "prompt": msg["text"],
                "chosen": chosen_reply["text"],
                "rejected": rejected_reply["text"]}
            )

    if "replies" in msg:
        for reply in msg["replies"]:
            prompt_replies.extend(extract_features(reply, reply["role"]))

    return prompt_replies


def level_loop(reader):
    samples = []
    for i, obj in enumerate(reader):
        if obj["prompt"]["role"] == "prompter":
            samples.extend(extract_features(obj["prompt"], "prompter"))
    return samples


def main(argv):
    args = argparse().parse_args(argv[1:])
    input_file = args.input
    reader = jsonlines.Reader(input_file)
    samples = level_loop(reader)
    for sample in samples:
        print(sample)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
