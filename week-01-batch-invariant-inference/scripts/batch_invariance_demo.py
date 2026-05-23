#!/usr/bin/env python3
"""Real-model probes for the batch-invariance article.

This script is for seeing the serving symptom before reading the proof. It does
two concrete things:

1. It mirrors the Thinking Machines experiment at a smaller scale by repeatedly
   requesting temperature-zero completions from a real hosted model through
   Tinker and counting unique outputs.
2. It can compare one local Hugging Face prompt alone versus the same prompt
   inside a padded batch, which exposes the kind of logit-level drift the Lean
   theorem is about.

The script never stores API keys. Pass Tinker credentials through TINKER_API_KEY
or with --tinker-key-file for local experiments.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path


def load_tinker_key(key_file: str | None) -> None:
    """Load a local Tinker API key into the environment, if one was provided."""
    if key_file is not None:
        os.environ["TINKER_API_KEY"] = Path(key_file).read_text().strip()


def tinker_repeated_completions(
    base_model: str,
    prompt_text: str,
    key_file: str | None,
    trials: int,
    max_tokens: int,
) -> None:
    """Run repeated temperature-zero completions through a hosted Tinker model."""
    load_tinker_key(key_file)

    # Tinker is optional: the Lean proof should build without any hosted-model
    # dependency. If the SDK is not installed, the probe exits cleanly.
    try:
        import tinker
        from tinker.lib.public_interfaces.sampling_client import _load_tokenizer_from_model_info
    except Exception as e:  # pragma: no cover - optional hosted demo
        print(f"Tinker probe skipped: could not import tinker ({e})")
        return

    if not os.environ.get("TINKER_API_KEY"):
        print("Tinker probe skipped: set TINKER_API_KEY or pass --tinker-key-file")
        return

    print("Tinker temperature-zero completion probe")
    print("----------------------------------------")
    print(f"base_model={base_model}")
    print(f"trials={trials}, max_tokens={max_tokens}, temperature=0, seed=0")
    print(f"prompt={prompt_text!r}")

    try:
        client = tinker.ServiceClient()
        sampler = client.create_sampling_client(base_model=base_model)
        tok = _load_tokenizer_from_model_info(base_model)
        prompt = tinker.ModelInput.from_ints(tok.encode(prompt_text))
        params = tinker.SamplingParams(max_tokens=max_tokens, temperature=0, seed=0)
    except Exception as e:  # pragma: no cover - optional hosted demo
        print(f"Tinker setup failed: {e}")
        return

    # Store token tuples rather than decoded text. Text decoding can hide token
    # differences, while the serving theorem is about the actual token stream.
    outputs: list[tuple[int, ...]] = []
    for n in range(trials):
        try:
            resp = sampler.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
        except Exception as e:  # pragma: no cover - optional hosted demo
            print(f"Tinker sample {n + 1} failed: {e}")
            return
        outputs.append(tuple(resp.sequences[0].tokens))

    counts = Counter(outputs)
    most_common_tokens, most_common_count = counts.most_common(1)[0]

    print(f"unique_completions={len(counts)}")
    print(f"most_common_count={most_common_count}/{trials}")
    print(f"most_common_tokens={list(most_common_tokens)}")
    print(f"most_common_text={tok.decode(list(most_common_tokens))!r}")

    if len(counts) > 1:
        print("other_outputs:")
        for tokens, count in counts.most_common()[1:5]:
            print(f"  count={count}, tokens={list(tokens)}, text={tok.decode(list(tokens))!r}")
        print("This run observed user-visible nondeterminism at temperature zero.")
    else:
        print("This run did not observe user-visible nondeterminism.")
        print("That is still useful: the Lean theorem is about the condition that explains this behavior.")


def tinker_list_models(key_file: str | None, limit: int) -> None:
    """Print a small model list so users can choose a hosted probe target."""
    load_tinker_key(key_file)

    try:
        import tinker
    except Exception as e:  # pragma: no cover - optional hosted demo
        print(f"Tinker model listing skipped: could not import tinker ({e})")
        return

    if not os.environ.get("TINKER_API_KEY"):
        print("Tinker model listing skipped: set TINKER_API_KEY or pass --tinker-key-file")
        return

    try:
        models = tinker.ServiceClient().get_server_capabilities().supported_models
    except Exception as e:  # pragma: no cover - optional hosted demo
        print(f"Tinker capability query failed: {e}")
        return

    print("Tinker supported models")
    print("-----------------------")
    for model in models[:limit]:
        print(f"{model.model_name}\tcontext={model.max_context_length}")


def local_hf_batch_probe(
    model_name: str,
    prompt: str,
    other_prompt: str,
    local_files_only: bool,
) -> None:
    """Compare one local HF forward pass alone versus inside a padded batch."""
    # This probe is intentionally lightweight. It is a convenient way to inspect
    # logit drift locally, but it is not used by any Lean theorem.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover - optional demo
        print(f"Local HF probe skipped: missing torch/transformers ({e})")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Local Hugging Face batch-context probe")
    print("--------------------------------------")
    print(f"model={model_name}")
    print(f"device={device}, dtype={dtype}, local_files_only={local_files_only}")

    try:
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Right padding keeps absolute positions for the prompt unchanged in
        # GPT-style models. Left padding can create a different batching bug by
        # shifting position ids, which is not the reduction-schedule issue here.
        tok.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(device)
    except Exception as e:  # pragma: no cover - optional demo
        print(f"could not load model: {e}")
        return

    model.eval()

    def final_prompt_logits(texts: list[str]) -> "torch.Tensor":
        """Return logits at the final real prompt token for each batch row."""
        enc = tok(texts, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        if tok.padding_side == "left":
            lengths = torch.full(
                (len(texts),),
                enc["attention_mask"].shape[1] - 1,
                dtype=torch.long,
                device=device,
            )
        else:
            lengths = enc["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(len(texts), device=device)
        return out.logits[batch_idx, lengths, :].float().cpu()

    single = final_prompt_logits([prompt])[0]
    batched = final_prompt_logits([prompt, other_prompt])[0]
    max_abs = (single - batched).abs().max().item()
    arg_single = int(single.argmax().item())
    arg_batched = int(batched.argmax().item())

    print(f"prompt={prompt!r}")
    print(f"other_prompt={other_prompt!r}")
    print(f"bitwise_equal_logits={torch.equal(single, batched)}")
    print(f"max_abs_logit_delta={max_abs:.8g}")
    print(f"argmax_single={arg_single} {tok.decode([arg_single])!r}")
    print(f"argmax_batched={arg_batched} {tok.decode([arg_batched])!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    # Hosted-model probe, closest in spirit to the Thinking Machines experiment.
    parser.add_argument("--tinker-repeat", action="store_true", help="repeat temp-zero Tinker completions")
    parser.add_argument("--tinker-list-models", action="store_true", help="list hosted Tinker models")
    # Local diagnostic probe for users without Tinker access.
    parser.add_argument("--local-hf-batch", action="store_true", help="compare local HF single vs padded batch")
    parser.add_argument("--tinker-base-model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--tinker-key-file", default=None)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--prompt", default="Tell me about Richard Feynman")
    parser.add_argument("--other-prompt", default="Batching an unrelated request should not change mine.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model-limit", type=int, default=20)
    args = parser.parse_args()

    ran = False
    if args.tinker_list_models:
        ran = True
        tinker_list_models(args.tinker_key_file, args.model_limit)
    if args.tinker_repeat:
        ran = True
        tinker_repeated_completions(
            args.tinker_base_model,
            args.prompt,
            args.tinker_key_file,
            args.trials,
            args.max_tokens,
        )
    if args.local_hf_batch:
        ran = True
        local_hf_batch_probe(
            args.model,
            args.prompt,
            args.other_prompt,
            local_files_only=not args.download,
        )
    if not ran:
        parser.print_help()


if __name__ == "__main__":
    main()
