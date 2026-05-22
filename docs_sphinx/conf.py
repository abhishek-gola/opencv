"""Sphinx wrapper for opencv/doc/.

The wrapper lives in opencv/docs_sphinx/ as a single conf.py. Sphinx is
invoked with config-dir / source-dir separation so the wrapper never
duplicates the legacy tree. Build via the CMake `sphinx` target:

    cmake --build <build> --target sphinx
    # output -> <build>/docs_sphinx/html/

opencv/doc/ stays untouched: Doxygen-flavored directives in the .markdown
sources are translated to MyST in a `source-read` hook below.

To enable additional tutorial modules, append their directory names (the
folder under opencv/doc/tutorials/) to DOC_MODULES below. The root index
(tutorials/tutorials.markdown) lists every module, but only modules in
DOC_MODULES are actually compiled — entries for the rest are dropped
from toctrees automatically.
"""

from __future__ import annotations
import pathlib, re

HERE = pathlib.Path(__file__).parent.resolve()
DOC_ROOT = (HERE.parent / "doc").resolve()
OPENCV_ROOT = HERE.parent.resolve()

# ---------------------------------------------------------------------------
# SCOPE — add module folder names from opencv/doc/tutorials/ here.
# Override via env var to avoid editing this file:
#     OPENCV_DOC_MODULES=photo,imgproc cmake --build <build> --target sphinx
# ---------------------------------------------------------------------------
import os as _os
DOC_MODULES = [
    m.strip()
    for m in (_os.environ.get("OPENCV_DOC_MODULES") or "photo,objdetect,imgproc,3d,app,ios").split(",")
    if m.strip()
]
DOC_JS_MODULES = [
    m.strip()
    for m in (_os.environ.get("OPENCV_DOC_JS_MODULES") or "js_gui,js_core").split(",")
    if m.strip()
]

# -- Project ----------------------------------------------------------------
project = "OpenCV"
author = "OpenCV Team"
release = "5.x"

# -- Sphinx core ------------------------------------------------------------
extensions = ["myst_parser"]
for _ext in ("sphinx_design", "sphinx_copybutton"):
    try:
        __import__(_ext)
        extensions.append(_ext)
    except ImportError:
        pass
HAVE_SPHINX_DESIGN = "sphinx_design" in extensions

source_suffix = {".md": "markdown", ".markdown": "markdown"}

# Root tutorial index (lists all modules via @subpage). Stays the master
# regardless of how many modules are in DOC_MODULES.
master_doc = "tutorials/tutorials"

# Source dir is opencv/doc/ — scope to the master + enabled modules only.
include_patterns = ["tutorials/tutorials.markdown"] + [
    f"tutorials/{m}/**" for m in DOC_MODULES
] + (["js_tutorials/js_tutorials.markdown"] + [
    f"js_tutorials/{m}/**" for m in DOC_JS_MODULES
] if DOC_JS_MODULES else [])
exclude_patterns = ["**/Thumbs.db", "**/.DS_Store", "tutorials/app/_old/**"]

myst_enable_extensions = [
    "colon_fence", "deflist", "dollarmath", "amsmath",
    "attrs_inline", "attrs_block", "smartquotes",
]
myst_heading_anchors = 4
suppress_warnings = ["myst.header", "myst.xref_missing", "toc.not_included"]

# -- Doxygen integration -----------------------------------------------------
# External links in the navbar and unbuilt-module sidebar entries point at
# the existing Doxygen build. Override the base URL or tagfile via env vars.
DOXYGEN_BASE_URL = (
    _os.environ.get("OPENCV_DOXYGEN_BASE_URL", "https://docs.opencv.org/5.x/")
    .rstrip("/") + "/")
_TAG_FILE = pathlib.Path(_os.environ.get(
    "OPENCV_DOXYGEN_TAGFILE",
    str(HERE.parent.parent / "build" / "doc" / "doxygen" / "html" / "opencv.tag"),
))

# anchor -> doxygen URL filename (from opencv.tag if available).
_TAG_FILENAMES: dict[str, str] = {}
if _TAG_FILE.is_file():
    try:
        import xml.etree.ElementTree as _ET
        for _c in _ET.parse(str(_TAG_FILE)).getroot().iter("compound"):
            if _c.get("kind") == "page":
                _n, _f = _c.findtext("name"), _c.findtext("filename")
                if _n and _f:
                    _TAG_FILENAMES[_n] = _f if _f.endswith(".html") else _f + ".html"
    except Exception:
        pass

def _doxygen_url(page: str) -> str:
    return DOXYGEN_BASE_URL + _TAG_FILENAMES.get(page, page)

# -- HTML / PyData theme ----------------------------------------------------
try:
    import pydata_sphinx_theme  # noqa: F401
    html_theme = "pydata_sphinx_theme"
except ImportError:
    html_theme = "alabaster"

html_title = "OpenCV Tutorials"
html_show_sourcelink = False
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500&display=swap",
    "custom.css",
]
html_theme_options = {
    "logo": {"text": f"OpenCV {release}"},
    # Show all 7 Doxygen-style external links inline (no "More" dropdown).
    "header_links_before_dropdown": 7,
    # Doxygen-style top-level nav (the legacy site's MAIN PAGE / RELATED
    # PAGES / NAMESPACES / CLASSES / FILES / EXAMPLES / JAVA DOCUMENTATION).
    # All external — they target the existing Doxygen build.
    "external_links": [
        {"url": _doxygen_url("index.html"),       "name": "Main Page"},
        {"url": _doxygen_url("pages.html"),       "name": "Related Pages"},
        {"url": _doxygen_url("namespaces.html"),  "name": "Namespaces"},
        {"url": _doxygen_url("annotated.html"),   "name": "Classes"},
        {"url": _doxygen_url("files.html"),       "name": "Files"},
        {"url": _doxygen_url("examples.html"),    "name": "Examples"},
        {"url": DOXYGEN_BASE_URL + "javadoc/",    "name": "Java Documentation"},
    ],
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_prev_next": True,
    "show_nav_level": 2,
    "navigation_depth": 4,
    "secondary_sidebar_items": ["page-toc"],
    "back_to_top_button": True,
    "show_version_warning_banner": False,
    "icon_links": [{"name": "GitHub",
                    "url": "https://github.com/opencv/opencv",
                    "icon": "fa-brands fa-github"}],
}

# Doxygen defines \fork{a}{b}{c}{d} as a piecewise-function shorthand.
# Define it as a MathJax macro so threshold.markdown renders correctly.
mathjax3_config = {
    "tex": {
        "macros": {
            "fork": [r"\left\{ \begin{array}{ll} #1 & \mbox{#2}\\ #3 & \mbox{#4}\end{array} \right.", 4],
        }
    }
}

# ===========================================================================
#  Doxygen-flavored .markdown  ->  MyST translation via source-read.
#  Nothing on disk under opencv/doc/ is modified.
# ===========================================================================

# Build anchor maps. Two kinds:
#   _ANCHOR_TO_DOC      anchor -> docname  (internal — for enabled modules)
#   _ANCHOR_TO_EXTERNAL anchor -> (title, url)  (external — for the rest)
# Disabled modules still appear in the master toctree as external links to
# the Doxygen build, so the left sidebar shows the full module list.
_ANCHOR_TO_DOC: dict[str, str] = {}
_ANCHOR_TO_EXTERNAL: dict[str, tuple[str, str]] = {}

_HEAD_RE = re.compile(
    r"^(?P<title1>[^\n]+?)\s*\{#(?P<anchor1>[\w-]+)\}\s*\n[=\-]{3,}\s*$"
    r"|"
    r"^#+\s+(?P<title2>[^\n]+?)\s*\{#(?P<anchor2>[\w-]+)\}\s*$",
    re.MULTILINE)

def _scan_internal(path: pathlib.Path) -> None:
    """Add every {#anchor} in `path` (file or dir) to _ANCHOR_TO_DOC."""
    if path.is_file() and path.suffix in (".markdown", ".md"):
        files = [path]
    elif path.is_dir():
        files = list(path.rglob("*.markdown")) + list(path.rglob("*.md"))
    else:
        files = []
    for md in files:
        try:
            head = md.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:
            continue
        rel = md.relative_to(DOC_ROOT).with_suffix("").as_posix()
        for m in re.finditer(r"\{#([\w-]+)\}", head):
            _ANCHOR_TO_DOC[m.group(1)] = rel

def _scan_external(toc_file: pathlib.Path) -> None:
    """Pull the top heading's (title, anchor) from a module's table_of_content
    file and add it to _ANCHOR_TO_EXTERNAL with a URL into the Doxygen build."""
    try:
        head = toc_file.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return
    m = _HEAD_RE.search(head)
    if not m:
        return
    anchor = m.group("anchor1") or m.group("anchor2")
    title = (m.group("title1") or m.group("title2") or "").strip()
    if not anchor:
        return
    url = DOXYGEN_BASE_URL + _TAG_FILENAMES.get(anchor, "index.html")
    _ANCHOR_TO_EXTERNAL[anchor] = (title, url)

# Internal scan: master + every enabled module subtree.
_scan_internal(DOC_ROOT / "tutorials" / "tutorials.markdown")
for _m in DOC_MODULES:
    _scan_internal(DOC_ROOT / "tutorials" / _m)

# External scan: every OTHER module's top-level table_of_content_*.markdown.
for _toc in (DOC_ROOT / "tutorials").glob("*/table_of_content_*.markdown"):
    if _toc.parent.name not in DOC_MODULES:
        _scan_external(_toc)

_scan_internal(DOC_ROOT / "js_tutorials" / "js_tutorials.markdown")
for _m in DOC_JS_MODULES:
    _scan_internal(DOC_ROOT / "js_tutorials" / _m)
for _toc in (DOC_ROOT / "js_tutorials").glob("*/js_table_of_contents_*.markdown"):
    if _toc.parent.name not in DOC_JS_MODULES:
        _scan_external(_toc)

# Doxygen flattens IMAGE_PATH across every `images/` folder under the tutorial
# tree, so a tutorial can reference `images/foo.png` even when `foo.png` lives
# in a sibling module's `images/` directory. Mirror that behavior by building
# a basename -> doc-root-relative-path index once at import time.
_IMAGE_INDEX: dict[str, str] = {}
for _img in (DOC_ROOT / "tutorials").rglob("images/*"):
    if _img.is_file():
        _IMAGE_INDEX.setdefault(_img.name, _img.relative_to(DOC_ROOT).as_posix())

_TOGGLE_LABELS = {"cpp": "C++", "java": "Java", "python": "Python"}


# Mirror of Doxygen's EXAMPLE_PATH (see opencv/doc/Doxyfile.in) — the bases a
# bare `@snippet some/path.cpp` is resolved against. OPENCV_ROOT comes first so
# fully-qualified paths like `samples/cpp/...` keep working.
_SNIPPET_BASES = [
    OPENCV_ROOT,
    OPENCV_ROOT / "samples",
    OPENCV_ROOT / "apps",
]


def _read_snippet(rel_path: str, label: str | None) -> tuple[str, str]:
    """Return (code_text, language) for an @include / @snippet directive."""
    p = next((b / rel_path for b in _SNIPPET_BASES
              if (b / rel_path).is_file()), None)
    if p is None:
        return f"// not found: {rel_path}\n", "text"
    text = p.read_text(encoding="utf-8", errors="replace")
    ext = p.suffix.lower()
    lang = {".cpp": "cpp", ".hpp": "cpp", ".h": "cpp", ".c": "c",
            ".py": "python", ".java": "java"}.get(ext, "text")
    if label is None:
        return text, lang
    # Doxygen matches `[label]` after any comment-style marker (//, //!, #, ##)
    # anywhere on a line — including labels wrapped in block-comments like
    # `/* //! [label]` or `//! [label] */`.
    pat = re.compile(r"^[^\[\n]*(?://!|//|##|#)[^\[\n]*\[" + re.escape(label)
                     + r"\][^\n]*$", re.MULTILINE)
    matches = list(pat.finditer(text))
    if len(matches) < 2:
        return f"// snippet not found: {rel_path} [{label}]\n", lang
    body = text[matches[0].end():matches[1].start()].strip("\n")
    lines = body.split("\n")
    indents = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
    if indents:
        dedent = min(indents)
        lines = [l[dedent:] if len(l) >= dedent else l for l in lines]
    return "\n".join(lines), lang


def _emit_toggles(tabs: list[tuple[str, str]], indent: str = "") -> str:
    def _dedent(body: str) -> str:
        # Fully dedent the tab-item body to column 0. The {tab-item} directive
        # resets the parser context, so directives (```{code-block}```) must be
        # at column 0 to be recognised — not at 4-space where CommonMark treats
        # them as indented code blocks. Using the actual minimum indent of all
        # non-empty lines (instead of exactly len(indent)) handles the case
        # where @snippet/@include is indented deeper than @add_toggle (e.g.
        # 8-space snippet inside a 4-space toggle block).
        lines = body.split("\n")
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            return body
        min_ind = min(len(l) - len(l.lstrip()) for l in non_empty)
        if min_ind == 0:
            return body
        return "\n".join(l[min_ind:] if l.strip() else l for l in lines)
    if HAVE_SPHINX_DESIGN:
        out = ["", "``````{tab-set}"]
        for lang, body in tabs:
            label = _TOGGLE_LABELS.get(lang, lang.title())
            out += [f"`````{{tab-item}} {label}", _dedent(body), "`````"]
        out += ["``````", ""]
        return "\n".join(out)
    # Fallback: render each toggle as a labeled section.
    out = [""]
    for lang, body in tabs:
        label = _TOGGLE_LABELS.get(lang, lang.title())
        out += [f"**{label}**", "", _dedent(body), ""]
    return "\n".join(out)


def _translate(text: str, docname: str | None = None) -> str:
    # 0. \htmlonly ... \endhtmlonly → raw HTML block; rewrite relative iframe
    #    src (../../foo.html) to absolute docs.opencv.org URL so demos load.
    def _htmlonly_repl(m: re.Match) -> str:
        body = re.sub(r'src="\.\.\/\.\.\/([\w/.-]+)"',
                      lambda mm: f'src="{DOXYGEN_BASE_URL}{mm.group(1)}"',
                      m.group(1))
        body = re.sub(r'\s*onload="[^"]*"', ' height="700px"', body)
        return f"\n```{{raw}} html\n{body}\n```\n"
    text = re.sub(r"^\\htmlonly\s*\n(.*?)\\endhtmlonly\s*$",
                  _htmlonly_repl, text, flags=re.DOTALL | re.MULTILINE)

    # 1. Heading anchors: "Title {#name}\n===" (setext) and "## Title {#name}" (ATX).
    #    Strip the anchor from the rendered heading and emit a MyST label
    #    "(name)=" immediately above. Setext converted to ATX for simplicity.
    def _setext_repl(m: re.Match) -> str:
        title = m.group("title").strip()
        level = 1 if m.group("bar") == "=" else 2
        return f"({m.group('anchor')})=\n{'#' * level} {title}"
    text = re.sub(
        r"^(?P<title>[^\n]+?)\s*\{#(?P<anchor>[\w-]+)\}\s*\n(?P<bar>[=\-])[=\-]{2,}\s*$",
        _setext_repl, text, flags=re.MULTILINE)
    text = re.sub(
        r"^(?P<hashes>#+)\s+(?P<title>[^\n]+?)\s*\{#(?P<anchor>[\w-]+)\}\s*$",
        lambda m: f"({m.group('anchor')})=\n{m.group('hashes')} {m.group('title')}",
        text, flags=re.MULTILINE)

    # 1a-i. Over-indented list markers.
    #   A bullet marker (-/*/+) followed by 5+ spaces causes CommonMark to use
    #   marker_col+2 as the continuation indent (the extra spaces beyond 4 are
    #   treated as content), making the actual text 6+ spaces into the content —
    #   above the 4-space threshold for indented code blocks.  Normalise to
    #   exactly 3 spaces after the marker and reduce all continuation lines by
    #   the same delta so nested structure is preserved.
    def _normalize_over_indented_markers(src: str) -> str:
        lines_in = src.split("\n")
        out: list[str] = []
        i = 0
        while i < len(lines_in):
            m = re.match(r"^([ \t]*)([-*+])( {5,})(.*)", lines_in[i])
            if m:
                outer, marker, spaces, content = (
                    m.group(1), m.group(2), m.group(3), m.group(4))
                old_col = len(outer) + 1 + len(spaces)
                new_col = len(outer) + 1 + 3
                delta = old_col - new_col
                out.append(f"{outer}{marker}   {content}")
                i += 1
                while i < len(lines_in):
                    line = lines_in[i]
                    stripped = line.lstrip(" \t")
                    actual = len(line) - len(stripped)
                    if not stripped:
                        out.append(line); i += 1; continue
                    if actual >= old_col:
                        out.append(" " * (actual - delta) + stripped); i += 1
                    else:
                        break
            else:
                out.append(lines_in[i]); i += 1
        return "\n".join(out)
    text = _normalize_over_indented_markers(text)

    # 1a-ii. 4-space indented list items directly under a heading.
    #   Doxygen ignores visual indentation; CommonMark treats 4-space-indented
    #   lines as indented code blocks.  Strip the 4-space prefix when list items
    #   immediately follow a heading (with optional blank lines between).
    text = re.sub(
        r"(^#{1,6}[ \t][^\n]+\n(?:[ \t]*\n)*)((?:    [ \t]*[-*+][^\n]*\n)+)",
        lambda m: m.group(1) + re.sub(r"^    ", "", m.group(2), flags=re.MULTILINE),
        text, flags=re.MULTILINE)

    # 1a-iii. 4-space list items under a plain paragraph line → strip to fix lazy continuation.
    text = re.sub(
        r"(^(?![ \t#@`]|-#|[-*+]\s|\d+[.)]\s)[^\n]+\n)((?:    [-*+][ \t][^\n]*\n(?:[ \t]{5,}[^\n]*\n)*)+)",
        lambda m: m.group(1) + re.sub(r"^    ", "", m.group(2), flags=re.MULTILINE),
        text, flags=re.MULTILINE)

    # 1b. @note ... / @see ...  -> MyST admonitions.
    #     Runs BEFORE math conversion so that \f[...\f] inside a note body is
    #     still on one logical line and does not create a blank-line terminator
    #     that would cut the body short.
    #     Allow optional leading indent and bare @note (body on next line).
    #     Dedent the body so indented lines don't become code blocks inside
    #     the directive.
    _ADMON_KIND = {"note": "note", "see": "seealso", "warning": "warning", "sa": "seealso"}
    def _admon_repl(m: re.Match) -> str:
        indent = m.group("indent")
        kind = _ADMON_KIND[m.group("dir")]
        raw = m.group("body")
        lines = raw.split("\n")
        min_ind = min(
            (len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
        body = "\n".join(l[min_ind:] for l in lines).strip()
        return f"\n{indent}:::{{{kind}}}\n{indent}{body}\n{indent}:::\n"
    text = re.sub(
        r"^(?P<indent>[ \t]*)@(?P<dir>note|see|warning|sa)[ \t]*\n?(?P<body>.+?)(?=\n[ \t]*\n|\n[ \t]*@[A-Za-z]|\Z)",
        _admon_repl, text, flags=re.DOTALL | re.MULTILINE)

    # 1c. @param name desc / @return desc → MyST definition list.
    def _param_block_repl(m: re.Match) -> str:
        items = []
        for line in m.group(0).strip().split("\n"):
            pm = re.match(r"@param\s+(\S+)\s+(.*)", line.strip())
            if pm:
                items.append(f"`{pm.group(1)}`\n: {pm.group(2).strip()}")
            rm = re.match(r"@return\s+(.*)", line.strip())
            if rm:
                items.append(f"*(return value)*\n: {rm.group(1).strip()}")
        return "\n\n".join(items) + "\n"
    text = re.sub(
        r"(^@(?:param\s+\S+|return)\s+[^\n]+\n)+",
        _param_block_repl, text, flags=re.MULTILINE)

    # 2. Doxygen LaTeX math markers.
    #    Block \f[...\f]: consume leading indent and re-emit it on the $$
    #    fence lines so the block stays inside any enclosing list item and
    #    the text that follows (at the same indent) is not misread as a code block.
    #
    #    Preprocess: when two \f[...\f] blocks are adjacent on the same source
    #    line (e.g. \end{bmatrix}\f]\f[G_{y}), split them onto separate lines
    #    and prefix the second \f[ with the line's own leading indent.  Without
    #    this the primary regex below would convert the first block correctly but
    #    leave the second \f[ at column 0 in the output; the fallback then emits
    #    that block at column 0, breaking any enclosing list structure.
    def _split_adj_math(m: re.Match) -> str:
        indent = m.group("indent")
        return m.group(0).replace("\\f]\\f[", f"\\f]\n{indent}\\f[")
    text = re.sub(r"^(?P<indent>[ \t]*)[^\n]*\\f\]\\f\[",
                  _split_adj_math, text, flags=re.MULTILINE)

    def _block_math_repl(m: re.Match) -> str:
        ind = m.group("indent")
        return f"\n{ind}$$\n{m.group('body').strip()}\n{ind}$$\n"
    text = re.sub(r"^(?P<indent>[ \t]*)\\f\[(?P<body>.+?)\\f\]",
                  _block_math_repl, text, flags=re.DOTALL | re.MULTILINE)
    # Fallback for any \f[...\f] not at line-start (e.g. two adjacent blocks).
    text = re.sub(r"\\f\[(.+?)\\f\]",
                  lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n",
                  text, flags=re.DOTALL)
    # Inline math.
    text = re.sub(r"\\f\$(.+?)\\f\$", lambda m: f"${m.group(1)}$",
                  text, flags=re.DOTALL)

    # 2b. Normalise unknown Pygments lexer names: plaintext/bash/sh → text.
    text = re.sub(r"^([ \t]*)```plaintext\b", r"\1```text", text, flags=re.MULTILINE)
    text = re.sub(r"^([ \t]*)```(?:bash|sh)\b", r"\1```text", text, flags=re.MULTILINE)

    # 3. @code{.lang} ... @endcode
    # Capture leading indent so the fence stays inside any enclosing list item.
    def _code_repl(m: re.Match) -> str:
        indent = m.group("indent")
        lang = (m.group("lang") or "").strip(".") or "text"
        if lang in ("none", "plaintext"):
            lang = "text"
        if lang in ("bash", "sh"):
            lang = "text"
        if lang == "m":
            lang = "objc"
        if lang == "js":
            lang = "javascript"
        raw = m.group("body").split("\n")
        non_empty = [l for l in raw if l.strip()]
        min_ind = min((len(l) - len(l.lstrip()) for l in non_empty), default=0)
        lines = [l[min_ind:] if l.strip() else "" for l in raw]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        body = "\n".join(indent + l for l in lines)
        return f"\n{indent}```{{code-block}} {lang}\n{body}\n{indent}```\n"
    text = re.sub(
        r"^(?P<indent>[ \t]*)@code(?:\{(?P<lang>[^}]*)\})?\s*\n(?P<body>.*?)\n[ \t]*@endcode",
        _code_repl, text, flags=re.DOTALL | re.MULTILINE)

    # 4. @include path  /  @includelineno path
    #    Indent preserved so code blocks inside list items don't break the list.
    def _include_repl(m: re.Match) -> str:
        indent = m.group("indent")
        code, lang = _read_snippet(m.group("path"), None)
        body = "\n".join(indent + l for l in code.rstrip().split("\n"))
        return f"\n{indent}```{{code-block}} {lang}\n{body}\n{indent}```\n"
    text = re.sub(r"^(?P<indent>[ \t]*)@include(?:lineno)?\s+(?P<path>\S+)",
                  _include_repl, text, flags=re.MULTILINE)

    # 4b. Remove stray @snippet that immediately follows @end_toggle at the same
    #     indent (no blank line between them). These are Doxygen fallback snippets
    #     for non-toggle Doxygen mode; the Sphinx build already shows them inside
    #     the tab-set, so the stray duplicate must be dropped before step 5
    #     converts it to a {code-block} that would land at document level with an
    #     invalid 4-space closing fence.
    text = re.sub(
        r"(^([ \t]*)@end_toggle[ \t]*\n)\2@snippet[^\n]*\n",
        r"\1",
        text, flags=re.MULTILINE)

    # 5. @snippet path [Label]
    # Indent preserved so code blocks inside list items don't break the list.
    def _snippet_repl(m: re.Match) -> str:
        indent = m.group("indent")
        code, lang = _read_snippet(m.group("path"), m.group("label"))
        body = "\n".join(indent + l for l in code.rstrip().split("\n"))
        return f"\n{indent}```{{code-block}} {lang}\n{body}\n{indent}```\n"
    text = re.sub(r"^(?P<indent>[ \t]*)@snippet\s+(?P<path>\S+)\s+(?P<label>[^\n]+?)\s*$",
                  _snippet_repl, text, flags=re.MULTILINE)

    # 5b. @snippetlineno — same as @snippet with :linenos:.
    def _snippetlineno_repl(m: re.Match) -> str:
        indent = m.group("indent")
        code, lang = _read_snippet(m.group("path"), m.group("label"))
        body = "\n".join(indent + l for l in code.rstrip().split("\n"))
        return f"\n{indent}```{{code-block}} {lang}\n{indent}:linenos:\n{body}\n{indent}```\n"
    text = re.sub(r"^(?P<indent>[ \t]*)@snippetlineno\s+(?P<path>\S+)\s+(?P<label>[^\n]+?)\s*$",
                  _snippetlineno_repl, text, flags=re.MULTILINE)

    # 6. @add_toggle_LANG ... @end_toggle  (coalesce runs into one tab-set)
    #    Capture the leading indent of each toggle block and emit the tab-set
    #    fence lines at the same indent, so toggles inside list items stay as
    #    list-item continuation content (where 4-space fences are valid).
    #    Body content is dedented so code blocks at column 0 inside the
    #    directive body are parsed correctly by myst-parser.
    def _toggle_collapse(src: str) -> str:
        out, i = [], 0
        opener = re.compile(r"^([ \t]*)@add_toggle_(\w+)[ \t]*$", re.MULTILINE)
        while True:
            m = opener.search(src, i)
            if not m:
                out.append(src[i:]); break
            out.append(src[i:m.start()])
            block_ind, tabs, j = m.group(1), [], m.start()
            while True:
                m2 = re.match(
                    r"[ \t]*@add_toggle_(\w+)[ \t]*\n(.*?)\n[ \t]*@end_toggle[ \t]*\n?",
                    src[j:], flags=re.DOTALL)
                if not m2:
                    break
                tabs.append((m2.group(1), m2.group(2)))
                j += m2.end()
                k = re.match(r"\s*", src[j:])
                if not k or not re.match(r"@add_toggle_", src[j + k.end():]):
                    break
                j += k.end()
            if not tabs:
                out.append(src[m.start():m.start() + 1]); i = m.start() + 1; continue
            out.append(_emit_toggles(tabs, block_ind))
            i = j
        return "".join(out)
    text = _toggle_collapse(text)

    # 6b. Strip list-item continuation indent stranded after a col-0 tab-set close.
    def _strip_tabset_continuations(src: str) -> str:
        lines = src.split("\n")
        out: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line == "``````":
                out.append(line)
                i += 1
                while i < len(lines):
                    ln = lines[i]
                    if ln.startswith("    "):
                        out.append(ln[4:])
                        i += 1
                    elif not ln.strip():
                        out.append(ln)
                        i += 1
                    else:
                        break
            else:
                out.append(line)
                i += 1
        return "\n".join(out)
    text = _strip_tabset_continuations(text)

    # 7. @ref name [optional "Display Text"]
    def _ref_repl(m: re.Match) -> str:
        name = m.group("name"); disp = m.group("disp")
        target = _ANCHOR_TO_DOC.get(name)
        if target:
            return f"[{disp or name}]({'/' + target})"
        return f"[{disp or name}](#{name})"
    text = re.sub(r'@ref\s+(?P<name>[\w:-]+)(?:\s+"(?P<disp>[^"]+)")?',
                  _ref_repl, text)

    # 8. @cite KEY → [[KEY]](link to docs.opencv.org citelist)
    text = re.sub(
        r"@cite\s+([\w-]+)",
        lambda m: f"[[{m.group(1)}]](https://docs.opencv.org/5.x/d0/de3/citelist.html#CITEREF_{m.group(1)})",
        text)

    # 8b. @youtube{ID}  -> responsive embed (raw HTML, passed through by MyST).
    text = re.sub(
        r"^@youtube\{(?P<id>[\w-]+)\}\s*$",
        lambda m: (
            '\n<div class="opencv-youtube">'
            f'<iframe src="https://www.youtube-nocookie.com/embed/{m.group("id")}" '
            'title="YouTube video player" frameborder="0" '
            'allow="accelerometer; autoplay; clipboard-write; encrypted-media; '
            'gyroscope; picture-in-picture" allowfullscreen></iframe></div>\n'
        ),
        text, flags=re.MULTILINE)

    # 9. @subpage NAME  (collected blocks -> real toctree).
    #    Enabled modules' anchors become internal toctree entries.
    #    Disabled modules' anchors become external links into the Doxygen
    #    build, so the left sidebar still shows the full module list.
    def _subpage_list_to_toctree(src: str) -> str:
        pat = re.compile(
            r"((?:^[ \t]*-\s+@subpage\s+[\w-]+[^\n]*\n(?:[ \t]*\n[ \t]+[^\n]+\n)*)+)",
            re.MULTILINE)
        def repl(m: re.Match) -> str:
            block = m.group(1)
            entries = re.findall(r"@subpage\s+([\w-]+)", block)
            dm = re.search(
                r"@subpage\s+[\w-]+[^\n]*\n(?:[ \t]*\n([ \t]+[^\n]+)\n)?", block)
            desc = dm.group(1).strip() if dm and dm.group(1) else None
            lines = []
            for e in entries:
                if e in _ANCHOR_TO_DOC:
                    lines.append("/" + _ANCHOR_TO_DOC[e])
                elif e in _ANCHOR_TO_EXTERNAL:
                    title, url = _ANCHOR_TO_EXTERNAL[e]
                    lines.append(f"{title} <{url}>")
            if not lines:
                return ""
            body = "\n".join(lines)
            result = f"\n```{{toctree}}\n:maxdepth: 1\n\n{body}\n```\n"
            if desc:
                result += f"\n{desc}\n"
            return result
        return pat.sub(repl, src)
    text = _subpage_list_to_toctree(text)

    # 10. @next_tutorial / @prev_tutorial  -> drop
    text = re.sub(r"^@(?:next|prev)_tutorial\{[^}]*\}\s*$", "",
                  text, flags=re.MULTILINE)

    # 10b. Doxygen ordered-list marker: "-#" -> "1."
    #      Doxygen uses -# for numbered lists; MyST uses standard 1. notation.
    text = re.sub(r"^([ \t]*)-#([ \t]+)", r"\g<1>1.\g<2>",
                  text, flags=re.MULTILINE)

    # 11. @tableofcontents -> drop (PyData right sidebar replaces it)
    text = re.sub(r"^@tableofcontents\s*$", "", text, flags=re.MULTILINE)

    # 11b. @cond NAME ... @endcond  -> strip just the markers; if the
    #      enclosed @subpage points to a disabled module it gets dropped
    #      by _subpage_list_to_toctree above.
    text = re.sub(r"^@cond\s+\S+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^@endcond\s*$", "", text, flags=re.MULTILINE)

    # 11c. Wrap bare http(s) URLs in <> for CommonMark autolink.
    def _autolink_repl(m: re.Match) -> str:
        url = m.group(0)
        trail = ""
        while url and url[-1] in ".,;:!?)":
            trail = url[-1] + trail
            url = url[:-1]
        return f"<{url}>{trail}" if url else m.group(0)
    text = re.sub(
        r'(?<!\]\()(?<![<"])https?://\S+',
        _autolink_repl, text)

    # 12. Image paths "images/foo.png" — resolve like Doxygen's flat IMAGE_PATH:
    #     prefer the doc's own "images/" sibling, then fall back to a global
    #     basename lookup across every tutorial `images/` folder. As a final
    #     fallback, point at the consolidated `tutorials/others/images/` dir
    #     (where modules like `photo` store their assets).
    def _img_repl(m: re.Match) -> str:
        rel = m.group("rel")
        if docname:
            local = DOC_ROOT / pathlib.Path(docname).parent / "images" / rel
            if local.is_file():
                return m.group(0)
        hit = _IMAGE_INDEX.get(pathlib.Path(rel).name)
        if hit:
            return f'{m.group("pre")}/{hit})'
        return f'{m.group("pre")}/tutorials/others/images/{rel})'
    text = re.sub(
        r'(?P<pre>!\[[^\]]*\]\()images/(?P<rel>[^)]+)\)',
        _img_repl, text)

    # 12b. Bare image filenames with no directory prefix (e.g. "psf.png") that
    #      Doxygen resolves via IMAGE_PATH but Sphinx cannot find as-is.
    #      Redirect to images/<name> when the file lives in the doc's own
    #      images/ sibling, otherwise fall back to the global index.
    def _bare_img_repl(m: re.Match) -> str:
        rel = m.group("rel")
        if docname:
            local = DOC_ROOT / pathlib.Path(docname).parent / "images" / rel
            if local.is_file():
                return f'{m.group("pre")}images/{rel})'
        hit = _IMAGE_INDEX.get(rel)
        if hit:
            return f'{m.group("pre")}/{hit})'
        return m.group(0)
    text = re.sub(
        r'(?P<pre>!\[[^\]]*\]\()(?P<rel>[A-Za-z0-9_.-]+\.[A-Za-z]{2,4})\)',
        _bare_img_repl, text)

    # 12c. Standalone ![Caption](path) → {figure} so alt text is a visible caption.
    text = re.sub(
        r"^([ \t]*)!\[(?P<alt>[^\]]+)\]\((?P<path>[^)\n]+)\)[ \t]*$",
        lambda m: (
            f"\n{m.group(1)}```{{figure}} {m.group('path')}\n"
            f"{m.group(1)}{m.group('alt').strip()}\n"
            f"{m.group(1)}```\n"
        ),
        text, flags=re.MULTILINE)

    # 12d. Doxygen ^ rowspan cell → merge into row above via <hr class="cv-rowdiv">.
    def _merge_caret_rows(src: str) -> str:
        lines = src.split("\n")
        out: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and "|" in stripped[1:]:
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if "^" in cells:
                    prev_idx = next(
                        (k for k in range(len(out) - 1, -1, -1)
                         if out[k].strip().startswith("|") and
                            not re.match(r"^\|[\s|:-]+\|$", out[k].strip())),
                        None)
                    if prev_idx is not None:
                        prev_cells = [c.strip() for c in
                                      out[prev_idx].strip().strip("|").split("|")]
                        merged = []
                        for j, cell in enumerate(cells):
                            pv = prev_cells[j] if j < len(prev_cells) else ""
                            if cell == "^":
                                merged.append(pv)
                            else:
                                sep = '<hr class="cv-rowdiv">'
                                merged.append(f"{pv}{sep}{cell}" if pv else cell)
                        out[prev_idx] = "| " + " | ".join(merged) + " |"
                        continue
            out.append(line)
        return "\n".join(out)
    text = _merge_caret_rows(text)

    # 13. Front-matter table: OpenCV tutorials use the "| -: | :- |"
    #     alignment pattern for the Original-author/Compatibility block.
    #     Wrap it in a {div} carrying .opencv-meta-table so custom.css
    #     can pin the rounded card + label-column styling without us
    #     modifying the .markdown source.
    def _wrap_front_matter(src: str) -> str:
        pat = re.compile(
            r"(^\|[^\n]*\|[ \t]*\n"     # header row (often empty)
            r"\|[ \t]*-:[ \t]*\|[ \t]*:-[ \t]*\|[ \t]*\n"  # alignment row
            r"(?:\|[^\n]*\|[ \t]*\n)+)",  # one or more body rows
            re.MULTILINE)
        def repl(m: re.Match) -> str:
            return f":::{{div}} opencv-meta-table\n\n{m.group(1)}\n:::\n"
        return pat.sub(repl, src, count=1)
    text = _wrap_front_matter(text)

    return text


def _source_read(app, docname, source):
    if not (docname.startswith("tutorials/") or docname.startswith("js_tutorials/")):
        return
    source[0] = _translate(source[0], docname)
    if docname == "tutorials/tutorials" and DOC_JS_MODULES:
        source[0] += (
            "\n\n```{toctree}\n:maxdepth: 1\n:caption: JavaScript Tutorials\n\n"
            "/js_tutorials/js_tutorials\n```\n"
        )


def setup(app):
    app.connect("source-read", _source_read)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
