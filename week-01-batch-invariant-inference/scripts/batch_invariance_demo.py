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
import base64
import json
import math
import os
from collections import Counter
from html import escape
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
        if local_files_only:
            print("The model is not in the local Hugging Face cache; rerun with --download.")
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


def panel_units(values: list[float]) -> tuple[int, str]:
    """Pick a power of 1000 so the largest value in a panel prints between 1 and 1000."""
    peak = max(values + [0.0])
    for exponent, units in ((0, "logit units"), (3, "× 10⁻³ logit units"), (6, "× 10⁻⁶ logit units")):
        if peak * 10 ** exponent >= 1:
            return 10 ** exponent, units
    return 10 ** 9, "× 10⁻⁹ logit units"


def write_margin_svg(rows: list[dict[str, object]], path: Path, width: int = 720) -> None:
    """Plot the recorded margin and drift separately so small changes remain visible.

    Both axes start at zero. Their units are labelled explicitly; no measured value
    is enlarged to make a mark visible. Hollow dots distinguish zero from missing data.
    """
    compact = width < 600
    height = 700 if compact else 650
    left, right = (48 if compact else 68), width - 26
    fonts = Path(__file__).resolve().parents[2] / "site" / "assets" / "fonts"
    # Embedded fonts also work when the figure is opened directly or used as an image.
    font = base64.b64encode((fonts / "ibm-plex-sans-latin-400-normal.woff2").read_bytes()).decode()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        '<title id="title">Top-two score gaps and observed batching drift</title>',
        '<desc id="description">Two panels show each prompt’s top-two score gap and twice '
        'its maximum observed logit change. Both axes start at zero, but use different scales. '
        'Hollow dots denote zero change. This is a measured comparison, not a bound for future runs.</desc>',
        f'<metadata id="measurements">{escape(json.dumps(rows))}</metadata>',
        '<style>'
        f'@font-face{{font-family:Plex;src:url(data:font/woff2;base64,{font}) format("woff2");}}'
        'text{font-family:Plex,Arial,sans-serif;fill:#252b32;font-size:15px;}'
        '.muted{fill:#69737d}.grid{stroke:#e0e5e8;stroke-width:1}'
        f'.heading{{font-size:{18 if compact else 21}px;font-weight:600}}.small{{font-size:13px}}'
        '</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="33" class="heading">Batching and the next token</text>',
        ('<text x="24" y="59" class="muted">A prompt alone vs. the same prompt</text>'
         '<text x="24" y="81" class="muted">in a padded batch</text>') if compact else
        '<text x="24" y="59" class="muted">A prompt alone vs. the same prompt in a padded batch</text>',
    ]
    for key, title, top, bottom, color in [
        ("margin_single", "Gap between the top two scores", 130, 280, "#286493"),
        ("two_epsilon", "Twice the observed logit change", 402, 552, "#b16943"),
    ]:
        if compact:
            top += 25
            bottom += 30
        multiplier, units = panel_units([float(row[key]) for row in rows])
        values = [float(row[key]) * multiplier for row in rows]
        peak = max(values + [1e-6])
        # Round the displayed ceiling up to a useful tick; leave the observations exact.
        raw_tick = peak * 1.12 / 4
        magnitude = 10 ** math.floor(math.log10(raw_tick))
        tick = next(x for x in (1, 2, 2.5, 5, 6, 8, 10) if x * magnitude >= raw_tick) * magnitude
        ceiling = tick * 4
        parts.extend([
            f'<circle cx="30" cy="{top-32}" r="4" fill="{color}"/>',
            f'<text x="42" y="{top-27}">{title}</text>',
            (f'<text x="42" y="{top-8}" class="small muted">{units}</text>' if compact else
             f'<text x="{right}" y="{top-27}" text-anchor="end" class="small muted">{units}</text>'),
        ])
        for i in range(5):
            value = tick * i
            y = bottom - (bottom - top) * i / 4
            parts.extend([
                f'<line class="grid" x1="{left}" y1="{y}" x2="{right}" y2="{y}"/>',
                f'<text x="{left-12}" y="{y+5}" text-anchor="end" class="muted">{value:g}</text>',
            ])
        for idx, (row, value) in enumerate(zip(rows, values)):
            x = left + (idx + 0.5) * (right - left) / max(len(rows), 1)
            y = bottom - (bottom - top) * value / ceiling
            label = escape(f'{row["prompt"]}: {key} = {row[key]}')
            parts.extend([
                f'<g data-prompt="{idx+1}" data-series="{key}" data-value="{row[key]}"><title>{label}</title>',
                f'<line x1="{x:.3f}" y1="{bottom}" x2="{x:.3f}" y2="{y:.3f}" '
                f'stroke="{color}" stroke-width="2" stroke-opacity=".45"/>',
                f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4.5" fill="{color if value else "white"}" '
                f'stroke="{color}" stroke-width="1.5"/>',
                f'<text x="{x:.3f}" y="{bottom+25}" text-anchor="middle" class="muted">{idx+1}</text></g>',
            ])
        parts.append(f'<text x="{(left+right)/2}" y="{bottom+49}" text-anchor="middle" class="small muted">Prompt number</text>')
    stable = sum(bool(row["argmax_stable"]) for row in rows)
    zero = sum(float(row["two_epsilon"]) == 0 for row in rows)
    if compact:
        parts.extend([
            f'<line x1="24" y1="654" x2="{right}" y2="654" class="grid"/>',
            f'<text x="24" y="675" class="small muted">Same winner in {stable}/{len(rows)} pairs</text>',
            f'<text x="24" y="695" class="small muted">Zero observed drift in {zero}/{len(rows)} pairs</text>',
        ])
    else:
        parts.extend([
            f'<line x1="24" y1="618" x2="{right}" y2="618" class="grid"/>',
            f'<text x="24" y="641" class="small muted">Same winner in {stable}/{len(rows)} pairs · '
            f'Zero observed drift in {zero}/{len(rows)} pairs</text>',
        ])
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    if not compact:
        write_margin_svg(rows, path.with_stem(path.stem + "-mobile"), width=360)


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

    try:
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            local_files_only=local_files_only,
        ).to(device)
    except OSError as e:
        print(f"could not load model: {e}")
        if local_files_only:
            print("The model is not in the local Hugging Face cache; rerun with --download.")
        raise SystemExit(1) from None
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
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
            "top1_logit_single": float(top_single.values[0].item()),
            "top2_logit_single": float(top_single.values[1].item()),
            "top1_logit_batched": float(top_batched.values[0].item()),
            "top2_logit_batched": float(top_batched.values[1].item()),
            "max_abs_delta_token_id": int((single - batched).abs().argmax().item()),
            "bitwise_equal_logits": bool(torch.equal(single, batched)),
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

    import transformers

    payload = {
        "model": model_name,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": str(dtype),
        "other_prompt": other_prompt,
        "rows": rows,
        "note": "Diagnostic only: local HF single-vs-batched forwards, not a proof and not a hosted Tinker logit trace.",
    }
    json_path = Path(json_out)
    svg_path = Path(svg_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
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
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompt", default="Tell me about Richard Feynman")
    parser.add_argument("--other-prompt", default="Batching an unrelated request should not change mine.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model-limit", type=int, default=20)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--json-out", default="week-01-batch-invariant-inference/results/local_hf_margin_probe_tinyllama.json")
    parser.add_argument("--svg-out", default="week-01-batch-invariant-inference/results/local_hf_margin_probe_tinyllama.svg")
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
