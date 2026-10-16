#!/usr/bin/env python3
"""Count the streamed documents that prepare_dataset.py reads for a --max-tokens budget.

prepare_dataset.py encodes every document with GPT-2 BPE, appends one end-of-text token, and stops
after the document that brings the written total to --max-tokens. This script repeats that count
on locally downloaded Parquet files of a Hugging Face dataset, read in the same order as the
streaming loader, without writing token shards. The result is the --skip-documents value for a
disjoint next shard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import tiktoken

GPT2_END_OF_TEXT_TOKENS = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("parquet", nargs="+", type=Path, help="files in streaming order")
    args = parser.parse_args()

    encoder = tiktoken.get_encoding("gpt2")
    written = 0
    documents = 0
    for path in args.parquet:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=4096, columns=[args.text_column]
        ):
            texts = batch.column(0).to_pylist()
            for ids in encoder.encode_ordinary_batch(texts, num_threads=args.threads):
                documents += 1
                written += min(len(ids) + GPT2_END_OF_TEXT_TOKENS, args.max_tokens - written)
                if written >= args.max_tokens:
                    print(json.dumps({
                        "max_tokens": args.max_tokens,
                        "documents_read": documents,
                        "tokens_written": written,
                        "files": [p.name for p in args.parquet],
                        "tiktoken": tiktoken.__version__,
                    }, indent=2))
                    return
    raise SystemExit(f"the files hold only {written} tokens in {documents} documents")


if __name__ == "__main__":
    main()
