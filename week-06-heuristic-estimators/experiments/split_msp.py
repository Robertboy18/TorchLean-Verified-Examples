"""Split msp.py's full record into a published summary and a compressed per-context file.

`python3 msp.py --output full.json` writes one record with every per-context value. This script
moves the `per_context` arrays into `results-msp-contexts.json.gz` (keyed by setting key) and writes
everything else to `results-msp.json`. `--join` reverses the split, giving back the full record.
"""

import argparse
import gzip
import json


def split(full, summary_path, contexts_path):
    record = json.load(open(full))
    contexts = {}
    for group in (record["settings"], record["added_later"]["settings"]):
        for setting in group:
            contexts[setting["key"]] = setting.pop("per_context")
    record["per_context_file"] = contexts_path.rsplit("/", 1)[-1]
    with open(summary_path, "w") as out:
        json.dump(record, out, indent=1)
        out.write("\n")
    with gzip.open(contexts_path, "wt", compresslevel=9) as out:
        json.dump(contexts, out, separators=(",", ":"))


def join(summary_path, contexts_path, full):
    record = json.load(open(summary_path))
    contexts = json.load(gzip.open(contexts_path, "rt"))
    for group in (record["settings"], record["added_later"]["settings"]):
        for setting in group:
            setting["per_context"] = contexts[setting["key"]]
    record.pop("per_context_file", None)
    with open(full, "w") as out:
        json.dump(record, out, indent=1)
        out.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--join", action="store_true", help="rebuild the full record")
    parser.add_argument("--full", default="results-msp-full.json")
    parser.add_argument("--summary", default="results-msp.json")
    parser.add_argument("--contexts", default="results-msp-contexts.json.gz")
    args = parser.parse_args()
    if args.join:
        join(args.summary, args.contexts, args.full)
    else:
        split(args.full, args.summary, args.contexts)
