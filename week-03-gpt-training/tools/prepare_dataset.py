#!/usr/bin/env python3
"""Prepare GPT-2 token shards for the Week 3 TorchLean runner.

The Lean executable reads a small interchange format: consecutive
little-endian unsigned 16-bit token ids. GPT-2's 50,257-token vocabulary fits
in that representation, and there is no pickle or Python object to trust when
Lean loads a shard.

With no source flag, this script downloads Tiny Shakespeare for a quick real
training run. A local text collection or one or more splits from a streaming
Hugging Face dataset can use the same output format for a longer experiment.
Dialogue datasets may also write a byte-per-token target mask. In that form,
only assistant tokens contribute to the training objective.
"""

from __future__ import annotations

import argparse
from array import array
from collections import deque
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from typing import Any, Iterable, Iterator
from urllib.request import urlopen

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
GPT2_VOCAB_URL = (
    "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json"
)
GPT2_MERGES_URL = (
    "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe"
)
GPT2_VOCAB_SIZE = 50_257
GPT2_END_OF_TEXT = 50_256
CHAT_ROLE_PREFIX = {
    "system": "System: ",
    "user": "User: ",
    "assistant": "Assistant: ",
}


@dataclass(frozen=True)
class Preparation:
    train_tokens: int
    validation_tokens: int
    source: str
    train_targets: int | None = None
    validation_targets: int | None = None
    train_records: int | None = None
    validation_records: int | None = None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create little-endian GPT-2 token shards for TorchLean."
    )
    source = p.add_mutually_exclusive_group()
    source.add_argument(
        "--input",
        action="append",
        type=Path,
        help="UTF-8 text file; repeat to concatenate several files.",
    )
    source.add_argument(
        "--hf-dataset",
        help="Hugging Face dataset name, read in streaming mode.",
    )
    p.add_argument("--hf-config", help="Optional Hugging Face dataset configuration.")
    p.add_argument(
        "--hf-revision",
        help="Dataset commit or tag recorded in the manifest and passed to load_dataset.",
    )
    p.add_argument(
        "--hf-split",
        action="append",
        help=(
            "Dataset split; repeat to interleave several splits. "
            "The default is train."
        ),
    )
    p.add_argument("--text-column", default="text", help="Text field in streamed rows.")
    p.add_argument(
        "--messages-column",
        help=(
            "Dialogue field containing a list of {role, content} objects. "
            "When supplied, write aligned token, target-mask, and dialogue-record "
            "files so training cannot cross conversation boundaries."
        ),
    )
    p.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help="Stop after this many streamed documents; 0 means no document limit.",
    )
    p.add_argument(
        "--skip-documents",
        type=int,
        default=0,
        help=(
            "Discard this many streamed documents before tokenization. This is "
            "much faster than --skip-tokens when a source-row boundary is known."
        ),
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Write at most this many tokens after skipping the source prefix; 0 means no limit.",
    )
    p.add_argument(
        "--skip-tokens",
        type=int,
        default=0,
        help=(
            "Discard this many source tokens before writing either shard. "
            "This supports disjoint continued-pretraining shards."
        ),
    )
    p.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of tokens or streamed documents reserved for validation.",
    )
    p.add_argument(
        "--record-context",
        type=int,
        default=1024,
        help=(
            "Maximum input length for each dialogue record (default: 1024). "
            "Long dialogues are split with one token of overlap."
        ),
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("week-03-gpt-training/data/tinyshakespeare"),
    )
    p.add_argument(
        "--no-tokenizer-assets",
        action="store_true",
        help="Do not download vocab.json and merges.txt for Lean-side generation.",
    )
    return p


def download(url: str, destination: Path) -> None:
    """Download atomically so an interrupted run cannot leave a valid-looking file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
        with urlopen(url) as response:
            shutil.copyfileobj(response, tmp)
    temporary.replace(destination)


def load_encoder():
    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit(
            "tiktoken is required; run `python -m pip install -r requirements.txt`"
        ) from exc
    return tiktoken.get_encoding("gpt2")


def checked_tokens(ids: Iterable[int]) -> array:
    """Build a portable uint16 array and reject accidental tokenizer drift."""
    result = array("H")
    for token in ids:
        if not 0 <= token < GPT2_VOCAB_SIZE:
            raise ValueError(f"GPT-2 token id out of range: {token}")
        result.append(token)
    return result


def write_little_endian(path: Path, tokens: array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = array("H", tokens)
    if sys.byteorder != "little":
        payload.byteswap()
    with path.open("wb") as handle:
        payload.tofile(handle)


def append_little_endian(handle, tokens: Iterable[int]) -> int:
    payload = checked_tokens(tokens)
    if sys.byteorder != "little":
        payload.byteswap()
    payload.tofile(handle)
    return len(payload)


def append_mask(handle, mask: Iterable[bool]) -> tuple[int, int]:
    payload = bytes(1 if enabled else 0 for enabled in mask)
    handle.write(payload)
    return len(payload), sum(payload)


def append_record(
    handle, offset: int, length: int, target_offset: int, target_length: int
) -> None:
    """Write one bounded dialogue window and its exact assistant-target interval."""
    if offset < 0 or length < 2 or target_length <= 0:
        raise ValueError("invalid dialogue record dimensions")
    if target_offset <= offset or target_offset + target_length > offset + length:
        raise ValueError("assistant targets must follow an input inside their dialogue window")
    handle.write(struct.pack("<QQQQ", offset, length, target_offset, target_length))


def dialogue_record_slices(
    mask: list[bool], context: int
) -> Iterator[tuple[int, int, int, int]]:
    """Make one prompt-preserving bounded record for each assistant-target run."""
    index = 1
    while index < len(mask):
        while index < len(mask) and not mask[index]:
            index += 1
        run_start = index
        while index < len(mask) and mask[index]:
            index += 1
        run_end = index
        target_start = run_start
        while target_start < run_end:
            target_end = min(target_start + context, run_end)
            window_start = max(0, target_end - context - 1)
            if window_start >= target_start:
                window_start = target_start - 1
            yield (
                window_start,
                target_end - window_start,
                target_start,
                target_end - target_start,
            )
            target_start = target_end


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_text_files(args: argparse.Namespace, encoder) -> Preparation:
    """Tokenize one logical corpus and make a contiguous 90/10-style split."""
    if args.input:
        pieces = [path.read_text(encoding="utf-8") for path in args.input]
        source_name = ",".join(str(path) for path in args.input)
    else:
        corpus_path = args.output / "source.txt"
        if not corpus_path.exists():
            print(f"downloading Tiny Shakespeare to {corpus_path}")
            download(TINY_SHAKESPEARE_URL, corpus_path)
        pieces = [corpus_path.read_text(encoding="utf-8")]
        source_name = "tiny-shakespeare"

    token_ids: list[int] = []
    source_limit = args.skip_tokens + args.max_tokens if args.max_tokens else 0
    for text in pieces:
        token_ids.extend(encoder.encode_ordinary(text))
        token_ids.append(GPT2_END_OF_TEXT)
        if source_limit and len(token_ids) >= source_limit:
            token_ids = token_ids[:source_limit]
            break
    token_ids = token_ids[args.skip_tokens :]
    if args.max_tokens:
        token_ids = token_ids[: args.max_tokens]
    if len(token_ids) < 2:
        raise SystemExit("the selected corpus produced fewer than two GPT-2 tokens")
    split = int(len(token_ids) * (1.0 - args.validation_fraction))
    split = min(max(split, 1), len(token_ids) - 1)
    train = checked_tokens(token_ids[:split])
    validation = checked_tokens(token_ids[split:])
    write_little_endian(args.output / "train.bin", train)
    write_little_endian(args.output / "val.bin", validation)
    return Preparation(len(train), len(validation), source_name)


def stable_validation_choice(seed: int, index: int, fraction: float) -> bool:
    """Choose a streamed document without depending on Python's salted hash."""
    digest = hashlib.sha256(f"{seed}:{index}".encode("ascii")).digest()
    sample = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return sample < fraction


def selected_hf_splits(args: argparse.Namespace) -> list[str]:
    """Return the requested splits while preserving command-line order."""
    return args.hf_split or ["train"]


def hf_source_name(args: argparse.Namespace) -> str:
    """Describe the selected dataset configuration and splits in the manifest."""
    split_names = ",".join(selected_hf_splits(args))
    config = f"/{args.hf_config}" if args.hf_config else ""
    return f"{args.hf_dataset}{config}[{split_names}]"


def streamed_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    """Interleave dataset splits and discard any requested source-row prefix."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets is required for --hf-dataset; install requirements.txt"
        ) from exc
    iterators = [
        iter(
            load_dataset(
                args.hf_dataset,
                args.hf_config,
                split=split,
                streaming=True,
                revision=args.hf_revision,
            )
        )
        for split in selected_hf_splits(args)
    ]
    streams = deque(iterators)
    source_index = 0
    try:
        while streams:
            stream = streams.popleft()
            try:
                row = next(stream)
            except StopIteration:
                continue
            if not isinstance(row, dict):
                raise SystemExit("the streamed dataset yielded a non-object row")
            streams.append(stream)
            if source_index < args.skip_documents:
                source_index += 1
                continue
            source_index += 1
            yield row
    finally:
        for stream in iterators:
            close = getattr(stream, "close", None)
            if close is not None:
                close()


def streamed_documents(args: argparse.Namespace) -> Iterator[str]:
    for index, row in enumerate(streamed_rows(args)):
        if args.max_documents and index >= args.max_documents:
            break
        value = row.get(args.text_column)
        if not isinstance(value, str):
            raise SystemExit(
                f"row {index} has no string column named {args.text_column!r}"
            )
        yield value


def prepare_stream(args: argparse.Namespace, encoder) -> Preparation:
    """Write large datasets incrementally instead of retaining every token in RAM."""
    train_path = args.output / "train.bin"
    validation_path = args.output / "val.bin"
    args.output.mkdir(parents=True, exist_ok=True)
    train_count = 0
    validation_count = 0
    source_tokens = 0
    written_tokens = 0
    with train_path.open("wb") as train_handle, validation_path.open("wb") as val_handle:
        for index, document in enumerate(streamed_documents(args)):
            ids = encoder.encode_ordinary(document)
            ids.append(GPT2_END_OF_TEXT)
            original_count = len(ids)
            if source_tokens < args.skip_tokens:
                discard = min(args.skip_tokens - source_tokens, original_count)
                ids = ids[discard:]
            source_tokens += original_count
            if not ids:
                continue
            if args.max_tokens:
                remaining = args.max_tokens - written_tokens
                if remaining <= 0:
                    break
                ids = ids[:remaining]
            if stable_validation_choice(
                args.seed, index, args.validation_fraction
            ):
                validation_count += append_little_endian(val_handle, ids)
            else:
                train_count += append_little_endian(train_handle, ids)
            written_tokens += len(ids)
            if index and index % 1000 == 0:
                print(
                    f"documents={index:,} source_tokens={source_tokens:,} "
                    f"train_tokens={train_count:,} "
                    f"validation_tokens={validation_count:,}"
                )
            if args.max_tokens and written_tokens >= args.max_tokens:
                break
    if train_count == 0 or validation_count == 0:
        raise SystemExit(
            "streaming split left one shard empty; increase the document/token limit"
        )
    return Preparation(train_count, validation_count, hf_source_name(args))


def local_dialogues(args: argparse.Namespace) -> Iterator[Any]:
    """Read JSON Lines dialogue records from one or more local input files."""
    if not args.input or not args.messages_column:
        raise RuntimeError("local dialogue input was not validated")
    index = 0
    for path in args.input:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if args.max_documents and index >= args.max_documents:
                    return
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(row, dict):
                    raise SystemExit(f"{path}:{line_number}: expected a JSON object")
                if args.messages_column not in row:
                    raise SystemExit(
                        f"{path}:{line_number}: no field named "
                        f"{args.messages_column!r}"
                    )
                yield row[args.messages_column]
                index += 1


def streamed_dialogues(args: argparse.Namespace) -> Iterator[Any]:
    """Read a dialogue column without retaining a streamed dataset in memory."""
    if not args.messages_column:
        raise RuntimeError("dialogue input was not validated")
    if args.input:
        yield from local_dialogues(args)
        return
    for index, row in enumerate(streamed_rows(args)):
        if args.max_documents and index >= args.max_documents:
            break
        if args.messages_column not in row:
            raise SystemExit(
                f"row {index} has no column named {args.messages_column!r}"
            )
        yield row[args.messages_column]


def encode_dialogue(encoder, raw_messages: Any, index: int) -> tuple[list[int], list[bool]]:
    """Encode one dialogue while keeping assistant-target alignment explicit."""
    if not isinstance(raw_messages, list):
        raise SystemExit(f"dialogue {index} is not a list of messages")

    token_ids: list[int] = []
    target_mask: list[bool] = []
    assistant_tokens = 0

    def append_segment(text: str, target: bool) -> None:
        nonlocal assistant_tokens
        ids = encoder.encode_ordinary(text)
        token_ids.extend(ids)
        target_mask.extend([target] * len(ids))
        if target:
            assistant_tokens += len(ids)

    for message_index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise SystemExit(
                f"dialogue {index}, message {message_index} is not an object"
            )
        role = message.get("role")
        content = message.get("content")
        if role not in CHAT_ROLE_PREFIX:
            raise SystemExit(
                f"dialogue {index}, message {message_index} has unsupported role "
                f"{role!r}; expected system, user, or assistant"
            )
        if not isinstance(content, str):
            raise SystemExit(
                f"dialogue {index}, message {message_index} has non-string content"
            )
        append_segment(CHAT_ROLE_PREFIX[role], False)
        append_segment(content.strip(" \t\n\r\v\f"), role == "assistant")
        append_segment("\n", role == "assistant")

    if assistant_tokens == 0:
        raise SystemExit(f"dialogue {index} contains no assistant target tokens")
    token_ids.append(GPT2_END_OF_TEXT)
    target_mask.append(True)
    return token_ids, target_mask


def prepare_dialogues(args: argparse.Namespace, encoder) -> Preparation:
    """Write aligned token, target-mask, and boundary-record shards."""
    train_path = args.output / "train.bin"
    validation_path = args.output / "val.bin"
    train_mask_path = args.output / "train.mask"
    validation_mask_path = args.output / "val.mask"
    train_records_path = args.output / "train.records"
    validation_records_path = args.output / "val.records"
    args.output.mkdir(parents=True, exist_ok=True)

    train_count = 0
    validation_count = 0
    train_targets = 0
    validation_targets = 0
    train_records = 0
    validation_records = 0
    source_tokens = 0
    written_tokens = 0
    handles = (
        train_path.open("wb"),
        validation_path.open("wb"),
        train_mask_path.open("wb"),
        validation_mask_path.open("wb"),
        train_records_path.open("wb"),
        validation_records_path.open("wb"),
    )
    try:
        (
            train_handle,
            validation_handle,
            train_mask_handle,
            validation_mask_handle,
            train_records_handle,
            validation_records_handle,
        ) = handles
        for index, raw_messages in enumerate(streamed_dialogues(args)):
            ids, mask = encode_dialogue(encoder, raw_messages, index)
            original_count = len(ids)
            source_tokens += original_count
            if args.max_tokens:
                remaining = args.max_tokens - written_tokens
                if remaining < len(ids):
                    break
            if len(ids) != len(mask):
                raise AssertionError("internal token/mask alignment failure")
            if len(ids) < 2 or not any(mask[1:]):
                continue

            if stable_validation_choice(args.seed, index, args.validation_fraction):
                record_offset = validation_count
                validation_count += append_little_endian(validation_handle, ids)
                mask_count, active_count = append_mask(validation_mask_handle, mask)
                if mask_count != len(ids):
                    raise AssertionError("validation token/mask write length mismatch")
                for start, length, target_start, target_length in dialogue_record_slices(
                    mask, args.record_context
                ):
                    append_record(
                        validation_records_handle,
                        record_offset + start,
                        length,
                        record_offset + target_start,
                        target_length,
                    )
                    validation_records += 1
                validation_targets += active_count
            else:
                record_offset = train_count
                train_count += append_little_endian(train_handle, ids)
                mask_count, active_count = append_mask(train_mask_handle, mask)
                if mask_count != len(ids):
                    raise AssertionError("training token/mask write length mismatch")
                for start, length, target_start, target_length in dialogue_record_slices(
                    mask, args.record_context
                ):
                    append_record(
                        train_records_handle,
                        record_offset + start,
                        length,
                        record_offset + target_start,
                        target_length,
                    )
                    train_records += 1
                train_targets += active_count

            written_tokens += len(ids)
            if index and index % 1000 == 0:
                print(
                    f"dialogues={index:,} source_tokens={source_tokens:,} "
                    f"train_tokens={train_count:,} "
                    f"train_targets={train_targets:,} "
                    f"validation_tokens={validation_count:,}"
                )
            if args.max_tokens and written_tokens >= args.max_tokens:
                break
    finally:
        for handle in handles:
            handle.close()

    if train_count == 0 or validation_count == 0:
        raise SystemExit(
            "dialogue split left one shard empty; increase the document/token limit"
        )
    if train_targets == 0 or validation_targets == 0:
        raise SystemExit(
            "dialogue split left one target mask empty; inspect the assistant messages"
        )
    if args.hf_dataset:
        source_name = hf_source_name(args)
    else:
        source_name = ",".join(str(path) for path in args.input)
    return Preparation(
        train_count,
        validation_count,
        source_name,
        train_targets,
        validation_targets,
        train_records,
        validation_records,
    )


def main() -> None:
    args = parser().parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must lie strictly between 0 and 1")
    if (
        args.max_documents < 0
        or args.skip_documents < 0
        or args.max_tokens < 0
        or args.skip_tokens < 0
    ):
        raise SystemExit(
            "document and token limits must be nonnegative"
        )
    if args.record_context <= 0:
        raise SystemExit("--record-context must be positive")
    if args.messages_column and not (args.hf_dataset or args.input):
        raise SystemExit(
            "--messages-column requires --hf-dataset or local JSON Lines --input"
        )
    if args.messages_column and args.skip_tokens:
        raise SystemExit(
            "--skip-tokens is not supported for dialogues because it can cut a conversation; "
            "use --skip-documents"
        )

    encoder = load_encoder()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.messages_column:
        result = prepare_dialogues(args, encoder)
    elif args.hf_dataset:
        result = prepare_stream(args, encoder)
    else:
        result = prepare_text_files(args, encoder)

    if not args.no_tokenizer_assets:
        vocab_path = args.output / "vocab.json"
        merges_path = args.output / "merges.txt"
        if not vocab_path.exists():
            print(f"downloading GPT-2 vocabulary to {vocab_path}")
            download(GPT2_VOCAB_URL, vocab_path)
        if not merges_path.exists():
            print(f"downloading GPT-2 merges to {merges_path}")
            download(GPT2_MERGES_URL, merges_path)

    train_path = args.output / "train.bin"
    validation_path = args.output / "val.bin"
    manifest = {
        "schema": "torchlean.gpt-token-shards.v3",
        "tokenizer": "gpt2",
        "tokenizer_package": {
            "name": "tiktoken",
            "version": metadata.version("tiktoken"),
        },
        "vocabulary_size": GPT2_VOCAB_SIZE,
        "encoding": "little-endian uint16",
        "source": result.source,
        "source_documents_skipped": args.skip_documents,
        "source_tokens_skipped": args.skip_tokens,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "dialogue_record_context": (
            args.record_context if args.messages_column else None
        ),
        "objective": (
            "assistant-token-weighted-next-token"
            if args.messages_column
            else "next-token"
        ),
        "train": {
            "path": train_path.name,
            "tokens": result.train_tokens,
            "sha256": sha256(train_path),
        },
        "validation": {
            "path": validation_path.name,
            "tokens": result.validation_tokens,
            "sha256": sha256(validation_path),
        },
    }
    if args.hf_dataset:
        manifest["hugging_face"] = {
            "dataset": args.hf_dataset,
            "config": args.hf_config,
            "revision": args.hf_revision,
            "splits": selected_hf_splits(args),
        }
    if args.messages_column:
        train_mask_path = args.output / "train.mask"
        validation_mask_path = args.output / "val.mask"
        manifest["messages_column"] = args.messages_column
        manifest["train"]["target_mask"] = {
            "path": train_mask_path.name,
            "active_tokens": result.train_targets,
            "sha256": sha256(train_mask_path),
        }
        manifest["validation"]["target_mask"] = {
            "path": validation_mask_path.name,
            "active_tokens": result.validation_targets,
            "sha256": sha256(validation_mask_path),
        }
        train_records_path = args.output / "train.records"
        validation_records_path = args.output / "val.records"
        manifest["train"]["dialogue_records"] = {
            "path": train_records_path.name,
            "records": result.train_records,
            "encoding": (
                "little-endian uint64 offset,length,target_offset,target_length records"
            ),
            "sha256": sha256(train_records_path),
        }
        manifest["validation"]["dialogue_records"] = {
            "path": validation_records_path.name,
            "records": result.validation_records,
            "encoding": (
                "little-endian uint64 offset,length,target_offset,target_length records"
            ),
            "sha256": sha256(validation_records_path),
        }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {result.train_tokens:,} training tokens and "
        f"{result.validation_tokens:,} validation tokens to {args.output}"
    )
    if args.messages_column:
        print(
            f"active assistant targets: train={result.train_targets:,} "
            f"validation={result.validation_targets:,}"
        )
        print(
            f"dialogue records: train={result.train_records:,} "
            f"validation={result.validation_records:,}"
        )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
