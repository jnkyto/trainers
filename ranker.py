import sys
import json
import textwrap
from argparse import ArgumentParser


def argparser():
    ap = ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    samples = []
    with open(args.input, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    for i, sample in enumerate(samples):
        print("\n" * 5)
        print(f"--> Viewing mode")
        print(f"Question ID {sample['question_id']}, Sample {i + 1} out of {len(samples)}:\n")
        keyset = {"turns": 0, "reference": 0}

        for key in keyset:
            if key in sample:
                keyset[key] += 1
            if keyset[key] > 0:
                print("Key: " + "\033[1m" + key + "\033[0m" + ";")
                for j, entry in enumerate(sample[key]):
                    prefix = f"{[j]}: "
                    wrapper = textwrap.TextWrapper(initial_indent=prefix, width=128,
                                                   subsequent_indent=' ' * len(prefix))
                    print(wrapper.fill(entry))

        usr_in = input("\033[1m" + "\nChange? " + "\033[0m" + "(input example: 'turns 1 2, reference 0') or press "
                                                              "Enter to skip: ").strip()

        if usr_in:
            changes = usr_in.split(',')
            for change in changes:
                parts = change.strip().split()
                key = parts[0].strip()  # First part is the key
                indices = [int(idx.strip()) for idx in parts[1:]]  # Rest are indices

                for idx in indices:
                    if key in sample and 0 <= idx < len(sample[key]):
                        print("\n" * 5)
                        print(f"--> Editing mode\n")
                        print(f"{key}{[idx]}: {sample[key][idx]}")
                        new_value = input(f"Enter new value for {key} {idx}: ").strip()
                        sample[key][idx] = new_value

    """
    # Save the modified samples back to the output file
    with open(args.output, 'w') as f_out:
        for sample in samples:
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
     """


if __name__ == "__main__":
    sys.exit(main(sys.argv))
