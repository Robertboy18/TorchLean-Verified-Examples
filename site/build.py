#!/usr/bin/env python3
"""Build the example site from READMEs, existing readers, and their checked-in sources.

Only website assets, source text, and small evidence files are published. Checkpoints,
local caches, and training data never enter the site. No Lean compilation is needed.
"""

from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import unquote, urlsplit

import markdown
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from lean import annotated, code_mask, load as load_lean, publish_sources, render_examples


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "build" / "site"
GITHUB = "https://github.com/Robertboy18/TorchLean-Verified-Examples/blob/master/"
SITE_URL = os.environ.get(
    "VERIFIED_EXAMPLES_SITE_URL",
    "https://robertboy18.github.io/TorchLean-Verified-Examples",
).rstrip('/')
PROJECTS = (
    ("week-01-batch-invariant-inference", "Thinking Machines Lab’s batch-invariance problem, formalized", "Inference"),
    ("week-02-verifiable-transformer-checkpoint", "Checking a small transformer in Lean", "Circuit verification"),
    ("week-03-gpt-training", "Training a GPT-2-style model in Lean", "Training"),
    ("torchlean-tensor-programming", "Writing tensor programs in Lean", "CPU and GPU programming"),
    ("week-04-kimi-k3-specification", "Kimi K3: the paper and proofs side by side", "Annotated report"),
    ("week-05-nemotron-specification", "Nemotron 3 Ultra: replaying a rounded state", "Recurrent state"),
)
TEXT_SUFFIXES = {".lean", ".md", ".py", ".sh", ".cpp", ".c", ".h", ".cu", ".json", ".svg", ".toml"}
LEAN_LEXER = get_lexer_by_name('lean4')


def syntax_spans(source):
    """Presentation only: keep compiler hovers and proof states unchanged."""
    categories = ((Token.Comment, 'comment'), (Token.String.Doc, 'comment'),
                  (Token.Keyword, 'keyword'), (Token.String, 'string'),
                  (Token.Number, 'number'), (Token.Name.Builtin, 'builtin'),
                  (Token.Operator, 'operator'))
    spans = []
    for start, token, value in LEAN_LEXER.get_tokens_unprocessed(source):
        kind = next((name for category, name in categories if token in category), None)
        if not kind:
            continue
        end = start + len(value)
        if spans and spans[-1][1] == start and spans[-1][2] == kind:
            spans[-1][1] = end
        else:
            spans.append([start, end, kind])
    return spans


class CodeText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def color_blocks(html):
    def annotate(match):
        text = CodeText()
        text.feed(match[2])
        spans = json.dumps(syntax_spans(''.join(text.parts)), separators=(',', ':'))
        return (match[1] + '<code data-lean-syntax="' + escape(spans, quote=True) + '">'
                + match[2] + '</code></pre>')
    return re.sub(r'(<pre\b[^>]*class="lean-code"[^>]*>)<code>(.*?)</code></pre>',
                  annotate, html, flags=re.S)


def proof_sections(html, recording):
    """Separate long theorem statements from proofs, using unchanged checked source ranges."""
    blocks = json.loads((ROOT / 'site/blocks.json').read_text())
    def replace(match):
        name = match[2]
        if name not in blocks:
            return match[0]
        block = blocks[name]
        model = recording['modules'][block['file']]
        declarations = []
        for declaration in block['declarations']:
            start, end = model['symbols'][declaration]
            source = model['code'][start:end]
            masked = code_mask(source)
            depth = 0
            cut = None
            if re.search(r'\btheorem\b', masked):
                for i, char in enumerate(masked):
                    if char in '([{':
                        depth += 1
                    elif char in ')]}':
                        depth -= 1
                    elif depth == 0 and (proof := re.match(r':=\s*by\b', masked[i:])):
                        by = i + proof.end() - 2
                        if source[by:].count('\n') >= 6:
                            cut = (start + i, start + by)
                        break
            declarations.append((start, end, cut))
        if not any(cut for _, _, cut in declarations):
            return match[0]
        parts = []
        for i, (start, end, cut) in enumerate(declarations):
            attrs = match[3] if i == 0 else ' class="lean-code"'
            if cut:
                statement, proof = cut
                parts.append(f'<pre{attrs} data-lean-part="statement"><code>'
                             + annotated(model, start, statement) + '</code></pre>'
                             + '<details class="lean-proof"><summary>Read the proof</summary>'
                             + '<pre class="lean-code" data-lean-part="proof"><code>'
                             + annotated(model, proof, end) + '</code></pre></details>')
            else:
                parts.append(f'<pre{attrs}><code>' + annotated(model, start, end) + '</code></pre>')
        return match[1] + ''.join(parts)
    return re.sub(r'(<div class="lean-example" data-example="([^"]+)">)'
                  r'<pre([^>]*)><code>.*?</code></pre>', replace, html, flags=re.S)


def project_url(folder):
    return f"{folder}/site/" if (ROOT / folder / "site" / "index.html").is_file() else f"{folder}/"


def math_assets(root):
    vendor = f"{root}week-04-kimi-k3-specification/site/vendor/katex/"
    return (f'<link rel="stylesheet" href="{vendor}katex.min.css">'
            f'<script defer src="{vendor}katex.min.js"></script>'
            f'<script defer src="{vendor}auto-render.min.js"></script>'
            f'<script defer src="{root}assets/article.js"></script>')


def cite_article(html, folder, destination):
    """Give each essay a citation without inventing a DOI or publication date."""
    title = CodeText()
    title.feed(re.search(r'<h1\b[^>]*>(.*?)</h1>', html, re.S)[1])
    title = ' '.join(''.join(title.parts).split())
    key = 'george2026' + folder.split('-')[2]
    url = f'{SITE_URL}/{project_url(folder)}'
    bibtex = (f'@misc{{{key},\n'
              '  author = {Robert Joseph George},\n'
              f'  title = {{{{{title}}}}},\n'
              '  year = {2026},\n'
              '  howpublished = {TorchLean Verified Examples},\n'
              f'  url = {{{url}}}\n'
              '}\n')
    asset = destination / 'assets' / 'citations' / f'{folder}.bib'
    asset.parent.mkdir(exist_ok=True)
    asset.write_text(bibtex)
    root = '../../' if project_url(folder).endswith('/site/') else '../'
    panel = (f'<details class="article-citation" id="cite-article">'
             '<summary>Cite this article</summary>'
             f'<p>Robert Joseph George. <cite>{escape(title)}</cite>. '
             'TorchLean Verified Examples, 2026.</p>'
             f'<pre><code>{escape(bibtex)}</code></pre>'
             f'<a href="{root}assets/citations/{folder}.bib" download>Download BibTeX</a>'
             '</details>')
    # Markdown articles place their footer inside <article>; report readers use <main>.
    if '<article id="article"' in html:
        html = html.replace('<nav class="page-nav"', panel + '<nav class="page-nav"', 1)
    elif '<footer class="site-footer">' in html:
        html = html.replace('<footer class="site-footer">', panel + '<footer class="site-footer">', 1)
    else:
        html = html.replace('</main>', panel + '</main>', 1)
    metadata = (f'<meta name="citation_title" content="{escape(title, quote=True)}">'
                '<meta name="citation_author" content="Robert Joseph George">'
                f'<link rel="canonical" href="{escape(url, quote=True)}">')
    return html.replace('</head>', metadata + '</head>', 1)


def page(folder, title, kind, number, *, development=False):
    essay = ROOT / "site" / "articles" / f"{folder}.md"
    is_essay = essay.is_file() and not development
    renderer = markdown.Markdown(
        extensions=["fenced_code", "tables", "toc", "sane_lists", "md_in_html", "pymdownx.arithmatex"],
        extension_configs={"pymdownx.arithmatex": {"generic": True}},
    )
    body = renderer.convert((essay if is_essay else ROOT / folder / "README.md").read_text())
    toc = renderer.toc
    byline = '<p class="article-meta">Robert Joseph George · TorchLean Verified Examples</p>'
    if is_essay:
        body = re.sub(r"(<h1\b[^>]*>.*?</h1>)", lambda match: match.group(1) + byline, body, count=1)
    else:
        body = byline + body
    # Keep the source text as the canonical instructions; add navigation outside it.
    previous = PROJECTS[number - 2] if number > 1 else None
    following = PROJECTS[number] if number < len(PROJECTS) else None
    links = []
    for entry, arrow in ((previous, "←"), (following, "→")):
        if entry:
            links.append(f'<a href="../{project_url(entry[0])}">{arrow} {escape(entry[1])}</a>')
    reader = ('<a href="./">Read the article</a>' if development
              else '<a href="site/">Open the annotated report →</a>' if number >= 4 else "")
    source_links = '' if is_essay else (
        f'<div class="project-links">{reader}<a href="README.md">README source</a>'
        f'<a href="https://github.com/Robertboy18/TorchLean-Verified-Examples/tree/master/{folder}/">'
        'Source on GitHub ↗</a></div>')
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="{escape(title)}: explanation, Lean source, and reproduction instructions.">
<title>{escape(title)} · TorchLean Verified Examples</title><link rel="stylesheet" href="../assets/site.css">{math_assets('../')}</head>
<body><a class="skip" href="#article">Skip to the article</a>
<header class="site-header"><a class="wordmark" href="../">TorchLean <span>Verified Examples</span></a>
<nav aria-label="Main navigation"><a href="../#examples">All examples</a><a href="https://lean-dojo.github.io/TorchLean/">TorchLean ↗</a>
<a href="{GITHUB}{folder}/README.md">GitHub ↗</a></nav></header>
<main class="article-shell"><aside class="article-sidebar" aria-label="Article contents">
<p class="eyebrow">{number:02d} · {escape(kind)}</p><details open><summary>On this page</summary>{toc}</details></aside>
<article id="article" class="article-content">{body}
{source_links}
<nav class="page-nav" aria-label="Other examples">{''.join(links)}</nav></article></main>
<footer class="site-footer"><a href="../">All examples</a></footer>
<script>
const narrow = matchMedia('(max-width:850px)');
const contents = document.querySelector('.article-sidebar details');
const fitContents = () => {{ contents.open = !narrow.matches; }};
fitContents();
narrow.addEventListener('change', fitContents);
</script>
</body></html>'''


class Links(HTMLParser):
    """Read URL attributes without interpreting article scripts or Lean text."""

    def __init__(self):
        super().__init__()
        self.urls = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for name in ("href", "src"):
            if name in attrs:
                self.urls.append((tag, attrs[name]))
        # Responsive figures must have valid compact assets as well as a fallback.
        for candidate in attrs.get("srcset", "").split(","):
            if candidate.strip():
                self.urls.append((tag, candidate.split()[0]))


def check_links(destination):
    """Reject broken local URLs before deploying; leave remote references untouched."""
    pages = list(destination.rglob("*.html"))
    parsed = {}
    for path in pages:
        parser = Links()
        parser.feed(path.read_text())
        parsed[path] = parser
    for path, parser in parsed.items():
        for tag, url in parser.urls:
            parts = urlsplit(url)
            if parts.scheme or parts.netloc:
                continue
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if not target.is_relative_to(destination):
                raise ValueError(f"URL escapes the site: {path.relative_to(destination)}: {url}")
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                raise ValueError(f"Missing local target: {path.relative_to(destination)}: {url}")
            if parts.fragment and target in parsed and unquote(parts.fragment) not in parsed[target].ids:
                raise ValueError(f"Missing anchor: {path.relative_to(destination)}: {url}")
    print(f"Checked local links and anchors in {len(pages)} HTML pages.")


def build():
    public_url = urlsplit(SITE_URL)
    if public_url.scheme not in {'http', 'https'} or not public_url.netloc or public_url.query or public_url.fragment:
        raise ValueError('VERIFIED_EXAMPLES_SITE_URL must be an absolute HTTP(S) base URL without a query or fragment.')
    lean_recording = load_lean()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verified-site-", dir=OUTPUT.parent) as temporary:
        destination = Path(temporary).resolve()
        (destination / "assets").mkdir()
        shutil.copyfile(ROOT / "site" / "index.html", destination / "index.html")
        shutil.copyfile(ROOT / "site" / "site.css", destination / "assets" / "site.css")
        shutil.copyfile(ROOT / "site" / "article.js", destination / "assets" / "article.js")
        # Hand-selected local fonts and illustrations; their licenses travel with them.
        shutil.copytree(ROOT / "site" / "assets", destination / "assets", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("lean.json"))
        publish_sources(destination, lean_recording)
        syntax_folder = destination / 'assets' / 'lean'
        for path, asset in json.loads((syntax_folder / 'index.json').read_text()).items():
            model = dict(lean_recording['modules'][path])
            model['syntax'] = syntax_spans(model['code'])
            (syntax_folder / asset).write_text(json.dumps(model, ensure_ascii=False,
                                                        separators=(',', ':')) + '\n')
        (destination / ".nojekyll").touch()
        for number, (folder, title, kind) in enumerate(PROJECTS, 1):
            source = ROOT / folder
            for path in source.rglob("*"):
                relative = path.relative_to(source)
                if any(part in {"data", ".lake", "__pycache__", "build", "checkpoint-final"} for part in relative.parts):
                    continue
                if path.is_symlink() or not path.is_file():
                    continue
                # Readers retain all their diagrams, scripts, KaTeX fonts, and license files.
                is_reader = relative.parts[0] == "site"
                if not is_reader and (path.suffix not in TEXT_SUFFIXES or path.stat().st_size > 2_000_000):
                    continue
                target = destination / folder / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            # Every README gets a readable page at its own directory, so its relative source
            # links keep working. The report readers remain at their established /site/ URLs.
            (destination / folder / "index.html").write_text(page(folder, title, kind, number))
            if (ROOT / "site" / "articles" / f"{folder}.md").is_file():
                (destination / folder / "development.html").write_text(page(folder, title, kind, number, development=True))
            for readme in (destination / folder).rglob("README.md"):
                if readme.parent == destination / folder:
                    continue
                renderer = markdown.Markdown(
                    extensions=["fenced_code", "tables", "toc", "pymdownx.arithmatex"],
                    extension_configs={"pymdownx.arithmatex": {"generic": True}},
                )
                content = renderer.convert(readme.read_text())
                root = "../" * len(readme.parent.relative_to(destination).parts)
                (readme.parent / "index.html").write_text(
                    f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                    f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                    f'<title>{escape(title)} · Development notes</title>'
                    f'<link rel="stylesheet" href="{root}assets/site.css">{math_assets(root)}</head><body>'
                    f'<main class="home"><p class="project-links"><a href="{root}">All examples</a></p>'
                    f'<article class="article-content">{content}</article></main></body></html>'
                )
        # An optional artifact omitted from the website still has a precise repository link.
        # Rewrite only links whose targets really exist in the source checkout.
        for html in destination.rglob("*.html"):
            def repository_link(match):
                url = match.group(2)
                parts = urlsplit(url)
                if parts.scheme or parts.netloc or not parts.path:
                    return match.group(0)
                target = (html.parent / unquote(parts.path)).resolve()
                if not target.is_relative_to(destination) or target.exists():
                    return match.group(0)
                relative = target.relative_to(destination)
                if (ROOT / relative).exists():
                    reference = GITHUB + relative.as_posix()
                    if parts.query:
                        reference += '?' + parts.query
                    if parts.fragment:
                        reference += '#' + parts.fragment
                    return f'{match.group(1)}="{escape(reference, quote=True)}"'
                return match.group(0)
            html.write_text(re.sub(r'(href|src)="([^"]+)"', repository_link, html.read_text()))
            html.write_text(color_blocks(proof_sections(
                render_examples(html.read_text(), lean_recording), lean_recording)))
        for folder, _, _ in PROJECTS:
            article = destination / project_url(folder) / 'index.html'
            article.write_text(cite_article(article.read_text(), folder, destination))
        check_links(destination)
        # Replace only this builder's generated output, never source or unrelated build caches.
        if OUTPUT.is_symlink():
            raise ValueError("Refusing to replace a symlink at build/site.")
        if OUTPUT.exists():
            shutil.rmtree(OUTPUT)
        destination.rename(OUTPUT)
    print(f"Site ready: {OUTPUT}")


if __name__ == "__main__":
    build()
