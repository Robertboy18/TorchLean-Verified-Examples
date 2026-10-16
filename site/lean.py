"""Record Lean-server hovers and command output for the article's runnable examples.

Run `python3 site/lean.py` after changing Examples.lean or the Lean sources it imports.
One server checks one document. Publishing the static site never starts a compiler.
"""

from hashlib import sha256
from html import escape
import argparse
import json
from html import unescape
from pathlib import Path
from queue import Empty, Queue
import re
import subprocess
from threading import Thread

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "site" / "Examples.lean"
RECORD = ROOT / "site" / "assets" / "lean.json"
IMPORTS = {
    "rounding": "BatchInvariantInference.Core",
    "prompts": "VerifiableTransformers.Certificate.FiniteEval",
    "resume": "TorchLeanGPT.TrainingCorrectness",
    "cap": "KimiK3.FeedForward.Activation",
    "kda": "KimiK3.Sequence.KDA\nimport NN.Tensor",
    "ssm": "Nemotron.SSM\nimport NN.Tensor",
    "expert": "KimiK3.GraphSpec.Expert",
    "expertBound": "KimiK3.GraphSpec.Expert\nimport NN.Proofs.Tensor.Algebra",
    "tensorFunction": "NN.Kernel",
    "tensorGradient": "NN.Kernel",
    "tensorRecurrence": "NN.Kernel",
    "tensorRows": "NN.Kernel",
    "tensorPrecision": "NN.Kernel",
    "tensorProof": "NN.Kernel\nimport NN.Tactic.Autograd",
    "tensorBlend": "NN.Kernel",
    "tensorRelaxation": "NN.Kernel\nimport Mathlib.Tactic.Ring",
    "tensorLoss": "NN.Kernel\nimport NN.Tactic.Autograd",
    "tensorContraction": "NN.Tensor",
    "tensorPatterns": "NN.Tensor",
    "tensorAttention": "NN.Tensor",
    "tensorModel": "NN.API",
    "tensorChainRule": "NN.Tactic.Autograd",
}
# A GPU call whose frontend compilation failed still typechecks: the elaborator stores the
# failure as an `Except.error` literal that the call only raises when it runs. This command runs
# after each example, outside the displayed code, and rejects every definition that stores one.
# It is conservative: a definition that only runs on CPU is rejected too, so keep such calls in
# `#eval`. The type names are unchecked so that examples without the GPU frontend still elaborate.
GPU_CHECK = """
open Lean in
elab "#site_reject_failed_gpu_compilation" : command => Lean.Elab.Command.liftTermElabM do
  let frontend := [`NN.Kernel.Elementwise, `NN.Kernel.Zip, `TorchLean.autograd.Internal.Recording]
  let failures ← IO.mkRef (#[] : Array Lean.MessageData)
  for (name, info) in (← Lean.getEnv).constants.map₂.toList do
    let some value := info.value? | continue
    value.forEach fun e => do
      if e.isAppOfArity `Except.error 3 && frontend.any (e.getArg! 1).isAppOf then
        if let .lit (.strVal message) := e.appArg! then
          failures.modify (·.push m!"{name}: {message}")
  let found ← failures.get
  unless found.isEmpty do
    let found := Lean.MessageData.joinSep found.toList "\n"
    Lean.logError m!"GPU compilation failed during elaboration; the call would fail when run:\n{found}"

#site_reject_failed_gpu_compilation
"""


def fingerprint(compilation=False):
    """Reject recordings made against different source or dependency revisions."""
    digest = sha256()
    paths = [ROOT / "lean-toolchain", ROOT / "lake-manifest.json", ROOT / "lakefile.lean"]
    if not compilation:
        paths += [SOURCE, Path(__file__), ROOT / "site" / "blocks.json"]
    for folder in sorted(ROOT.glob("week-*")):
        paths.extend(sorted(folder.rglob("*.lean")))
    for path in paths:
        if any(part in {".lake", "build"} for part in path.relative_to(ROOT).parts):
            continue
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class Server:
    """The standard editor protocol; no custom evaluator or inferred-type guesses."""

    def __init__(self):
        self.process = subprocess.Popen(
            ["lake", "env", "lean", "--server"], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        self.messages = Queue()
        self.diagnostics = {}
        self.sequence = 0
        Thread(target=self.read, daemon=True).start()

    def read(self):
        try:
            while True:
                headers = {}
                while (line := self.process.stdout.readline().strip()):
                    key, value = line.decode().split(":", 1)
                    headers[key.lower()] = value.strip()
                if not headers:
                    raise EOFError("Lean server closed its output")
                size = int(headers["content-length"])
                self.messages.put(json.loads(self.process.stdout.read(size)))
        except Exception as error:
            self.messages.put(error)

    def write(self, message):
        data = json.dumps(message).encode()
        self.process.stdin.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
        self.process.stdin.flush()

    def notify(self, method, params):
        self.write({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method, params):
        self.sequence += 1
        ident = self.sequence
        self.write({"jsonrpc": "2.0", "id": ident, "method": method, "params": params})
        while True:
            try:
                message = self.messages.get(timeout=30)
            except Empty:
                if self.process.poll() is not None:
                    raise RuntimeError("Lean server stopped before replying")
                continue
            if isinstance(message, Exception):
                raise message
            if "method" in message:
                if "id" in message:
                    self.write({"jsonrpc": "2.0", "id": message["id"], "result": None})
                if message["method"] == "textDocument/publishDiagnostics":
                    params = message["params"]
                    self.diagnostics[params["uri"]] = params["diagnostics"]
            elif message.get("id") == ident:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result")

    def close(self):
        try:
            if self.process.poll() is None:
                self.request("shutdown", {})
                self.notify("exit", {})
                self.process.stdin.close()
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()


def code_mask(text):
    """Mask comments and strings while preserving every source offset."""
    chars = list(text)
    index = depth = 0
    quoted = False
    while index < len(text):
        pair = text[index:index + 2]
        if depth:
            if pair == "/-":
                depth += 1
                chars[index:index + 2] = "  "
                index += 2
                continue
            if pair == "-/":
                depth -= 1
                chars[index:index + 2] = "  "
                index += 2
                continue
            if text[index] != "\n":
                chars[index] = " "
        elif quoted:
            chars[index] = " " if text[index] != "\n" else "\n"
            if text[index] == "\\" and index + 1 < len(text):
                chars[index + 1] = " "
                index += 2
                continue
            if text[index] == '"':
                quoted = False
        elif pair == "/-":
            depth = 1
            chars[index:index + 2] = "  "
            index += 2
            continue
        elif pair == "--":
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            chars[index:end] = " " * (end - index)
            index = end
            continue
        elif text[index] == '"':
            quoted = True
            chars[index] = " "
        index += 1
    return "".join(chars)


def check_document(server, text, version=1, path=SOURCE, gpu_check=False):
    uri = path.as_uri()
    checked = text + (GPU_CHECK if gpu_check else "") + "\n#check Nat\n"
    if version == 1:
        server.notify("textDocument/didOpen", {"textDocument": {
            "uri": uri, "languageId": "lean4", "version": version, "text": checked,
        }})
    else:
        server.diagnostics.pop(uri, None)
        server.notify("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": version},
            "contentChanges": [{"text": checked}],
        })
    lines = checked.splitlines(keepends=True)
    starts = []
    total = 0
    for line in lines:
        starts.append(total)
        total += len(line)
    # The appended command waits for elaboration through every declaration and proof.
    server.request("textDocument/hover", {
        "textDocument": {"uri": uri},
        "position": {"line": len(lines) - 1, "character": 8},
    })
    failures = [d["message"] for d in server.diagnostics.get(uri, []) if d["severity"] == 1]
    if failures:
        raise ValueError(f"{path.relative_to(ROOT)} did not check:\n" + "\n".join(failures))
    info = []
    info_ids = {}
    def intern(value):
        if value not in info_ids:
            info_ids[value] = len(info)
            info.append(value)
        return info_ids[value]
    spans, goals = [], []
    identifiers = re.compile(r"[^\W\d][\w'!?]*(?:\.[^\W\d][\w'!?]*)*|\d+(?:\.\d+)?")
    masked = code_mask(checked).splitlines(keepends=True)
    for line, source in enumerate(masked):
        if starts[line] >= len(text):
            break
        for token in identifiers.finditer(source):
            column = len(lines[line][:token.start()].encode("utf-16-le")) // 2
            hover = server.request("textDocument/hover", {
                "textDocument": {"uri": uri}, "position": {"line": line, "character": column},
            })
            value = (hover or {}).get("contents", {}).get("value", "")
            match = re.match(r"```lean\n(.*?)\n```", value, re.S)
            if match:
                spans.append({"start": starts[line] + token.start(),
                              "end": starts[line] + token.end(), "info": intern(match[1])})
        # Every code line is inspectable, including tactic keywords and the final 'no goals'.
        if source.strip():
            state = server.request("$/lean/plainGoal", {
                "textDocument": {"uri": uri}, "position": {
                    "line": line, "character": len(lines[line].rstrip('\n').encode('utf-16-le')) // 2,
                },
            })
            if state:
                value = "\n\n".join(state["goals"]) if state["goals"] else "no goals"
                goals.append({"start": starts[line],
                              "end": min(starts[line] + len(lines[line].rstrip('\n')), len(text)),
                              "info": intern(value)})
    diagnostics = server.diagnostics.get(uri, [])
    messages = [{"line": d["range"]["start"]["line"], "text": d["message"],
                 "severity": d["severity"]}
                for d in diagnostics if starts[d["range"]["start"]["line"]] < len(text)]
    for message in messages:
        line = message["line"]
        command = re.match(r"#(?:eval|check)\b", lines[line])
        if command:
            spans.append({"start": starts[line], "end": starts[line] + len(command[0]),
                          "info": intern(message["text"])})
    def source_offset(position):
        line = position['line']
        prefix = lines[line].encode('utf-16-le')[:position['character'] * 2].decode('utf-16-le')
        return min(starts[line] + len(prefix), len(text))
    symbols = {}
    def visit(items, namespace=''):
        for symbol in items:
            name = symbol['name']
            qualified = '.'.join(filter(None, [namespace, name])) if not name.startswith('<') else namespace
            if 'children' in symbol:
                visit(symbol['children'], qualified)
            elif name and not name.startswith('<'):
                symbols[qualified] = [source_offset(symbol['range']['start']),
                                      source_offset(symbol['range']['end'])]
    visit(server.request('textDocument/documentSymbol', {'textDocument': {'uri': uri}}))
    spans.sort(key=lambda span: span['start'])
    if not spans:
        raise ValueError(f"No compiler information recorded for {path}")
    return {'code': text, 'spans': spans, 'goals': goals, 'info': info,
            'messages': messages, 'symbols': symbols}


def source_files(blocks):
    return sorted({block["file"] for block in blocks.values()})


def start_server():
    server = Server()
    server.request('initialize', {'rootUri': ROOT.as_uri(), 'capabilities': {
        'textDocument': {'documentSymbol': {'hierarchicalDocumentSymbolSupport': True}},
    }})
    server.notify('initialized', {})
    return server


def complete_symbols(model):
    """Lean omits `omit … in` declarations from documentSymbol; their type hovers identify them."""
    masked = code_mask(model['code'])
    commands = list(re.finditer(
        r'^(?:(?:noncomputable|private|protected)\s+)*'
        r'(?:theorem|lemma|def|abbrev|instance|structure|inductive|class|namespace|end|'
        r'variable|omit|include|open|attribute|set_option)\b', masked, re.M))
    for span in model['spans']:
        prefix = masked[:span['start']]
        if not re.search(r'\b(?:theorem|lemma|def|abbrev)\s+$', prefix):
            continue
        name = re.match(r'([^\s{:(]+)', model['info'][span['info']])
        if not name:
            continue
        qualified = name[1].rstrip('.')
        if qualified in model['symbols']:
            continue
        # The hover's declaration identity is compiler supplied. Command boundaries only
        # choose which unchanged source text to display, never a type or proof.
        before = [command for command in commands if command.start() < span['start']]
        start = before[-1].start()
        if len(before) > 1 and before[-2][0].startswith('omit'):
            start = before[-2].start()
        next_command = next((command.start() for command in commands
                             if command.start() > span['end']), len(masked))
        end = len(masked[:next_command].rstrip())
        model['symbols'][qualified] = [start, end]
    return model


def record(only=None):
    before = fingerprint()
    blocks = json.loads((ROOT / 'site/blocks.json').read_text())
    cache = ROOT / 'build/lean-cache'
    cache.mkdir(parents=True, exist_ok=True)
    modules = {}
    for file in source_files(blocks):
        path = ROOT / file
        text = path.read_text()
        key = sha256(('hover-goal-v1' + fingerprint(compilation=True) + file).encode()).hexdigest()
        cached = cache / (key + '.json')
        if cached.exists():
            modules[file] = complete_symbols(json.loads(cached.read_text()))
        elif not only or file == only:
            server = start_server()
            try:
                module = check_document(server, text, path=path)
                cached.write_text(json.dumps(module, ensure_ascii=False, separators=(',', ':')))
                modules[file] = complete_symbols(module)
                print(f"Checked {file}: {len(module['spans'])} types, {len(module['goals'])} proof states.", flush=True)
            finally:
                server.close()
    if only:
        print(f"Cached {len(modules)}/{len(source_files(blocks))} source files.")
        return
    examples = {}
    server = start_server()
    try:
        pattern = r'^-- ANCHOR: (\w+)\n(.*?)^-- ANCHOR_END: \1$'
        for version, match in enumerate(re.finditer(pattern, SOURCE.read_text(), re.M | re.S), 1):
            name = match.group(1)
            code = f'import {IMPORTS[name]}\n\n' + match.group(2).rstrip() + '\n'
            # Only TorchLean's NN modules carry the GPU frontend (and the elaborator the check needs).
            gpu_check = 'NN.' in IMPORTS[name]
            examples[name] = check_document(server, code, version, gpu_check=gpu_check)
            print(f"Checked {name}: {len(examples[name]['spans'])} hover entries.", flush=True)
    finally:
        server.close()
    if before != fingerprint():
        raise ValueError('Lean sources changed during recording; no final output was written')
    if set(examples) != set(IMPORTS):
        raise ValueError('An example anchor or its import is missing')
    for name, block in blocks.items():
        for declaration in block['declarations']:
            if declaration not in modules[block['file']]['symbols']:
                raise ValueError(f'{name}: declaration {declaration} is missing')
    result = {'fingerprint': before, 'toolchain': (ROOT / 'lean-toolchain').read_text().strip(),
              'examples': examples, 'modules': modules}
    RECORD.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f"Recorded {len(examples)} examples, {len(blocks)} excerpts, {len(modules)} complete source files.")
def load():
    result = json.loads(RECORD.read_text())
    if result["fingerprint"] != fingerprint():
        raise ValueError("Lean recording is stale. Run: python3 site/lean.py")
    return result


# Pages inline the hover text of every token and proof line. A few proof states run to
# thousands of lines (a 15-line `simp only` showed one 165,000-character goal on each line),
# so a page keeps the start of each long state. The recording and the lazily loaded full
# source keep every state whole; the displayed code is unchanged.
HOVER_LIMIT = 4000


def hover_text(info):
    """Shorten a hover longer than HOVER_LIMIT, keeping the start of each goal."""
    if len(info) <= HOVER_LIMIT:
        return info
    goals = info.split('\n\n')
    budget = max(HOVER_LIMIT // len(goals), 400)
    shown = goals[:max(1, HOVER_LIMIT // budget)]
    parts = []
    def more(count, noun):
        return f'⋯ {count:,} more {noun}' + ('' if count == 1 else 's')
    for goal in shown:
        lines = goal.split('\n')
        kept, room = [], budget
        for line in lines:
            if room <= 0:
                break
            kept.append(line if len(line) < room else line[:room] + ' ⋯')
            room -= len(line) + 1
        if len(kept) < len(lines):
            kept.append(more(len(lines) - len(kept), 'line'))
        parts.append('\n'.join(kept))
    if len(shown) < len(goals):
        parts.append(more(len(goals) - len(shown), 'goal'))
    return '\n\n'.join(parts) + f'\n\n(Shortened from {len(info):,} characters.)'


def annotated(model, start=0, end=None):
    end = len(model['code']) if end is None else end
    spans = [span for span in model['spans'] if start <= span['start'] < span['end'] <= end]
    goals = {(goal['start'], goal['end']): goal for goal in model['goals']
             if start <= goal['start'] < goal['end'] <= end}
    rendered = []
    line_start = start
    previous = None
    for line in model['code'][start:end].splitlines(keepends=True):
        line_end = line_start + len(line.rstrip('\n'))
        parts, cursor = [], line_start
        for span in spans:
            if not line_start <= span['start'] < span['end'] <= line_end:
                continue
            if span['start'] < cursor:
                raise ValueError('Overlapping compiler tokens')
            parts.append(escape(model['code'][cursor:span['start']]))
            info = hover_text(model['info'][span['info']])
            parts.append(f'<span class="lean-token" tabindex="0" '
                         f'data-lean-info="{escape(info, quote=True)}">'
                         f'{escape(model["code"][span["start"]:span["end"]])}</span>')
            cursor = span['end']
        parts.append(escape(model['code'][cursor:line_end]))
        content = ''.join(parts)
        goal = goals.get((line_start, line_end))
        if goal:
            state = model['info'][goal['info']]
            # A multi-line tactic repeats its state on every line; inline a long one once.
            if goal['info'] == previous and len(state) > HOVER_LIMIT:
                state = 'Same proof state as the line above.'
            previous = goal['info']
            state = escape(hover_text(state), quote=True)
            content = (f'<span class="lean-step" tabindex="0" data-lean-goal="{state}" '
                       f'data-lean-info="{state}">{content}</span>')
        else:
            previous = None
        rendered.append(content + ('\n' if line.endswith('\n') else ''))
        line_start += len(line)
    return ''.join(rendered)


def render_examples(html, recording):
    """Every displayed block comes from a compiler-checked document or declaration."""
    blocks = json.loads((ROOT / 'site/blocks.json').read_text())
    def replace(match):
        attrs, kind, name = match.group(1), match.group(2), match.group(3)
        attrs = re.sub(r'\s*data-lean-(?:example|block)="[^"]+"', '', attrs)
        if kind == 'example':
            model = recording['examples'][name]
            code = annotated(model)
            output = '\n\n'.join(message['text'] for message in model['messages'])
            footer = (f'<details class="lean-output" open><summary>Lean output</summary>'
                      f'<pre><code>{escape(output)}</code></pre></details>') if output else ''
        else:
            block = blocks[name]
            model = recording['modules'][block['file']]
            code = '\n\n'.join(annotated(model, *model['symbols'][declaration])
                               for declaration in block['declarations'])
            footer = (f'<details class="lean-context" data-lean-source="{escape(block["file"])}">'
                      f'<summary>Full source</summary><pre class="lean-code" hidden><code></code></pre>'
                      f'</details>')
        return (f'<div class="lean-example" data-example="{name}">'
                f'<pre class="lean-code"{attrs}><code>{code}</code></pre>{footer}</div>')
    return re.sub(r'<pre\b([^>]*\bdata-lean-(example|block)="(\w[\w-]*)"[^>]*)>\s*</pre>', replace, html)


def publish_sources(destination, recording):
    """Keep full-file metadata lazy-loaded rather than duplicating it in every excerpt."""
    folder = destination / 'assets' / 'lean'
    folder.mkdir(parents=True, exist_ok=True)
    index = {}
    for path, model in recording['modules'].items():
        name = sha256(path.encode()).hexdigest()[:16] + '.json'
        (folder / name).write_text(json.dumps(model, ensure_ascii=False, separators=(',', ':')))
        index[path] = name
    (folder / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', help='Check and cache one source file, then stop')
    record(parser.parse_args().source)
