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
import json
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
            dtype=dtype,
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


DEFAULT_MARGIN_PROMPTS = [
    "Tell me about Richard Feynman",
    "Explain why floating point addition is not associative",
    "Write one sentence about New York City",
    "What is a theorem prover?",
    "Give a short definition of batch inference",
    "Name one reason GPU reductions can be nondeterministic",
    "Summarize the idea of speculative decoding",
    "What does a neural network logit represent?",
    "Explain RMSNorm in simple terms",
    "Why do servers batch LLM requests?",
    "Give a concise description of CUDA",
    "What is the role of a verifier?",
]


def write_margin_svg(rows: list[dict[str, object]], path: Path) -> None:
    """Write a small dependency-free SVG for the margin theorem diagnostic."""
    width = 1100
    height = 620
    pad_l, pad_r, pad_t, pad_b = 92, 44, 112, 128
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    ymax = max(
        [float(r["margin_single"]) for r in rows]
        + [float(r["two_epsilon"]) for r in rows]
        + [1e-6]
    )
    ymax *= 1.15
    n = len(rows)
    group = plot_w / max(n, 1)
    bar_w = min(28, group * 0.26)

    def y(v: float) -> float:
        return pad_t + plot_h - (v / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs>',
        '<linearGradient id="paper" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#fffaf2"/><stop offset="100%" stop-color="#f6e6c8"/></linearGradient>',
        '<filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="14" stdDeviation="16" flood-color="#5a3611" flood-opacity=".12"/></filter>',
        '</defs>',
        '<rect width="100%" height="100%" rx="28" fill="url(#paper)"/>',
        '<rect x="28" y="28" width="1044" height="564" rx="24" fill="#fff8ee" stroke="#e1bd80" filter="url(#softShadow)"/>',
        '<style>text{font-family:Source Sans 3,Source Sans Pro,Arial,sans-serif;fill:#241b14}.title{font-size:28px;font-weight:700;letter-spacing:-.02em}.subtitle{font-size:15px;fill:#746656}.small{font-size:13px;fill:#746656}.axis{stroke:#b9873e;stroke-width:1.2}.margin{fill:#2f6bb8}.eps{fill:#d46f2a}.grid{stroke:#ead7b8;stroke-width:1}.rule{stroke:#d9b16d;stroke-width:1}</style>',
        '<text x="76" y="68" class="title">Margin certificate diagnostic</text>',
        '<text x="76" y="94" class="subtitle">For each prompt: compare the top-two logit margin with the theorem threshold 2ε.</text>',
    ]
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = pad_t + plot_h * (1 - frac)
        val = ymax * frac
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width-pad_r}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l-14}" y="{yy+4:.1f}" text-anchor="end" class="small">{val:.2g}</text>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{width-pad_r}" y2="{pad_t+plot_h}" class="axis"/>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" class="axis"/>')
    for idx, row in enumerate(rows):
        x0 = pad_l + idx * group + group / 2
        m = float(row["margin_single"])
        e2 = float(row["two_epsilon"])
        ym, ye = y(m), y(e2)
        parts.append(f'<rect x="{x0-bar_w-3:.1f}" y="{ym:.1f}" width="{bar_w:.1f}" height="{pad_t+plot_h-ym:.1f}" rx="6" class="margin"/>')
        parts.append(f'<rect x="{x0+3:.1f}" y="{ye:.1f}" width="{bar_w:.1f}" height="{pad_t+plot_h-ye:.1f}" rx="6" class="eps"/>')
        parts.append(f'<text x="{x0:.1f}" y="{pad_t+plot_h+28}" text-anchor="middle" class="small">{idx+1}</text>')
    parts.extend([
        f'<text x="{pad_l}" y="{pad_t-16}" class="small">logit units</text>',
        f'<rect x="{width-360}" y="58" width="16" height="16" rx="4" class="margin"/><text x="{width-336}" y="71" class="small">top-two margin</text>',
        f'<rect x="{width-360}" y="83" width="16" height="16" rx="4" class="eps"/><text x="{width-336}" y="96" class="small">2ε drift bound</text>',
        f'<line x1="{pad_l}" y1="{height-74}" x2="{width-pad_r}" y2="{height-74}" class="rule"/>',
        f'<text x="{pad_l}" y="{height-46}" class="small">If the blue bar is above the orange bar, the Lean margin theorem says the greedy token is stable for that perturbation bound.</text>',
        f'<text x="{pad_l}" y="{height-25}" class="small">Data: local HF single-vs-batched forwards for sshleifer/tiny-gpt2. Exact prompt labels and values are in the JSON output.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts))


def local_hf_margin_plot(
    model_name: str,
    prompts_file: str | None,
    other_prompt: str,
    local_files_only: bool,
    json_out: str,
    svg_out: str,
) -> None:
    """Log real top-two margins for local single-vs-batched HF forwards."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover - optional demo
        print(f"Local HF margin probe skipped: missing torch/transformers ({e})")
        return

    if prompts_file:
        prompts = [line.strip() for line in Path(prompts_file).read_text().splitlines() if line.strip()]
    else:
        prompts = DEFAULT_MARGIN_PROMPTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print("Local Hugging Face margin diagnostic")
    print("------------------------------------")
    print(f"model={model_name}")
    print(f"device={device}, dtype={dtype}, prompts={len(prompts)}, local_files_only={local_files_only}")

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    def final_prompt_logits(texts: list[str]) -> "torch.Tensor":
        enc = tok(texts, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        lengths = enc["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(len(texts), device=device)
        return out.logits[batch_idx, lengths, :].float().cpu()

    rows: list[dict[str, object]] = []
    for idx, prompt in enumerate(prompts, start=1):
        single = final_prompt_logits([prompt])[0]
        batched = final_prompt_logits([prompt, other_prompt])[0]
        top_single = torch.topk(single, k=2)
        top_batched = torch.topk(batched, k=2)
        max_abs = float((single - batched).abs().max().item())
        top1 = int(top_single.indices[0].item())
        top2 = int(top_single.indices[1].item())
        btop1 = int(top_batched.indices[0].item())
        row = {
            "index": idx,
            "prompt": prompt,
            "top1_token_id": top1,
            "top1_text": tok.decode([top1]),
            "top2_token_id": top2,
            "top2_text": tok.decode([top2]),
            "batched_top1_token_id": btop1,
            "batched_top1_text": tok.decode([btop1]),
            "margin_single": float((top_single.values[0] - top_single.values[1]).item()),
            "margin_batched": float((top_batched.values[0] - top_batched.values[1]).item()),
            "epsilon_max_abs_delta": max_abs,
            "two_epsilon": 2.0 * max_abs,
            "argmax_stable": top1 == btop1,
        }
        rows.append(row)
        print(
            f"{idx:02d}: margin={row['margin_single']:.6g}, "
            f"2eps={row['two_epsilon']:.6g}, stable={row['argmax_stable']}, "
            f"top={row['top1_text']!r}"
        )

    payload = {
        "model": model_name,
        "device": device,
        "dtype": str(dtype),
        "other_prompt": other_prompt,
        "rows": rows,
        "note": "Diagnostic only: local HF single-vs-batched forwards, not a proof and not a hosted Tinker logit trace.",
    }
    json_path = Path(json_out)
    svg_path = Path(svg_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))
    write_margin_svg(rows, svg_path)
    print(f"wrote_json={json_path}")
    print(f"wrote_svg={svg_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    # Hosted-model probe, closest in spirit to the Thinking Machines experiment.
    parser.add_argument("--tinker-repeat", action="store_true", help="repeat temp-zero Tinker completions")
    parser.add_argument("--tinker-list-models", action="store_true", help="list hosted Tinker models")
    # Local diagnostic probe for users without Tinker access.
    parser.add_argument("--local-hf-batch", action="store_true", help="compare local HF single vs padded batch")
    parser.add_argument("--local-hf-margin-plot", action="store_true", help="write JSON/SVG top-two margin diagnostic")
    parser.add_argument("--tinker-base-model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--tinker-key-file", default=None)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--prompt", default="Tell me about Richard Feynman")
    parser.add_argument("--other-prompt", default="Batching an unrelated request should not change mine.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model-limit", type=int, default=20)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--json-out", default="week-01-batch-invariant-inference/results/local_hf_margin_probe.json")
    parser.add_argument("--svg-out", default="week-01-batch-invariant-inference/results/local_hf_margin_probe.svg")
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
    if args.local_hf_margin_plot:
        ran = True
        local_hf_margin_plot(
            args.model,
            args.prompts_file,
            args.other_prompt,
            local_files_only=not args.download,
            json_out=args.json_out,
            svg_out=args.svg_out,
        )
    if not ran:
        parser.print_help()


if __name__ == "__main__":
    main()
