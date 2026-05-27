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
import pathlib, re, textwrap as _textwrap

HERE = pathlib.Path(__file__).parent.resolve()
DOC_ROOT = (HERE.parent / "doc").resolve()
OPENCV_ROOT = HERE.parent.resolve()

# ---------------------------------------------------------------------------
# Main modules from opencv/doc/tutorials/. Override via OPENCV_DOC_MODULES.
# ---------------------------------------------------------------------------
import os as _os
DOC_MODULES = [
    m.strip()
    for m in (_os.environ.get("OPENCV_DOC_MODULES") or "photo,objdetect,core,calib3d,features,introduction").split(",")
    if m.strip()
]

# ---------------------------------------------------------------------------
# Contrib modules from opencv_contrib/modules/. Override via OPENCV_CONTRIB_MODULES.
# ---------------------------------------------------------------------------
CONTRIB_MODULES = [
    m.strip()
    for m in (_os.environ.get("OPENCV_CONTRIB_MODULES") or "ml,bgsegm,bioinspired,cannops,ccalib,cnn_3dobj,cvv,dnn_objdetect,dnn_superres,gapi,hdf,julia,line_descriptor,phase_unwrapping,structured_light").split(",")
    if m.strip()
]
CONTRIB_ROOT = pathlib.Path(
    _os.environ.get("OPENCV_CONTRIB_ROOT")
    or str(HERE.parent.parent / "opencv_contrib" / "modules")
).resolve()

# ---------------------------------------------------------------------------
# API reference modules. Override via OPENCV_API_MODULES.
# ---------------------------------------------------------------------------
API_MODULES = [
    m.strip()
    for m in (_os.environ.get("OPENCV_API_MODULES") or "core").split(",")
    if m.strip()
]

# Sphinx source directory. Defaults to DOC_ROOT for direct sphinx-build runs.
SPHINX_INPUT_ROOT = pathlib.Path(
    _os.environ.get("OPENCV_SPHINX_INPUT_ROOT") or str(DOC_ROOT)
).resolve()

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

# -- Breathe (Doxygen XML -> Sphinx C++ domain) -----------------------------
# Gated on API_MODULES and breathe. Falls back to canonical build_doc layout.
_API_XML_DIR = pathlib.Path(
    _os.environ.get("OPENCV_DOXYGEN_XML_DIR")
    or str(HERE.parent.parent / "build_doc" / "doc" / "doxygen" / "xml")
).resolve()
if API_MODULES:
    try:
        import breathe  # noqa: F401
        extensions.append("breathe")
        breathe_projects = {"opencv": str(_API_XML_DIR)}
        breathe_default_project = "opencv"
        breathe_default_members = ("members",)
    except ImportError:
        API_MODULES = []

source_suffix = {".md": "markdown", ".markdown": "markdown"}

# Master document for tutorials
master_doc = "tutorials/tutorials"

# Define source inclusions for enabled modules
include_patterns = ["tutorials/tutorials.markdown"] + [
    f"tutorials/{m}/**" for m in DOC_MODULES
]
if CONTRIB_MODULES and (SPHINX_INPUT_ROOT / "tutorials_contrib").is_dir():
    include_patterns.append("tutorials_contrib/contrib_root.markdown")
    include_patterns += [f"tutorials_contrib/{m}/**" for m in CONTRIB_MODULES]
if API_MODULES:
    # Include generated API stubs recursively.
    include_patterns.append("api/**")
exclude_patterns = [
    "**/Thumbs.db", "**/.DS_Store",
    "tutorials/core/how_to_use_OpenCV_parallel_for_/**",
    "tutorials/introduction/load_save_image/**",
]

myst_enable_extensions = [
    "colon_fence", "deflist", "dollarmath", "amsmath",
    "attrs_inline", "attrs_block", "smartquotes",
]
myst_heading_anchors = 4
suppress_warnings = [
    "myst.header", "myst.xref_missing", "toc.not_included",
    "misc.highlighting_failure",
    "image.not_readable",
]

# -- Doxygen integration -----------------------------------------------------
# Links to external Doxygen build. Override base URL or tagfile via env vars.
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
    # External top-level nav links targeting Doxygen build
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

# ===========================================================================
#  Translate Doxygen-flavored .markdown to MyST dynamically
# ===========================================================================

# Maps for resolving anchors to internal docs or external Doxygen URLs.
_ANCHOR_TO_DOC: dict[str, str] = {}
_ANCHOR_TO_EXTERNAL: dict[str, tuple[str, str]] = {}

_HEAD_RE = re.compile(
    r"^(?P<title1>[^\n]+?)\s*\{#(?P<anchor1>[\w-]+)\}\s*\n[=\-]{3,}\s*$"
    r"|"
    r"^#+\s+(?P<title2>[^\n]+?)\s*\{#(?P<anchor2>[\w-]+)\}\s*$",
    re.MULTILINE)

def _scan_internal(path: pathlib.Path, base: pathlib.Path | None = None) -> None:
    """Map internal anchors to docnames relative to the staging root."""
    base = base or SPHINX_INPUT_ROOT
    _md_exts = (".markdown", ".md")
    files = [path] if (path.is_file() and path.suffix in _md_exts) \
        else (list(path.rglob("*.markdown")) + list(path.rglob("*.md"))
              if path.is_dir() else [])
    for md in files:
        try:
            body = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Resolve paths relative to staging root.
        rel = md.relative_to(base).with_suffix("").as_posix()
        for m in re.finditer(r"\{#([\w-]+)\}", body):
            _ANCHOR_TO_DOC[m.group(1)] = rel
        for m in re.finditer(r"^@anchor\s+([\w-]+)\s*$", body, re.MULTILINE):
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

# Map internal anchors for master and enabled modules.
_scan_internal(SPHINX_INPUT_ROOT / "tutorials" / "tutorials.markdown")
for _m in DOC_MODULES:
    _scan_internal(SPHINX_INPUT_ROOT / "tutorials" / _m)
_contrib_root_md = SPHINX_INPUT_ROOT / "tutorials_contrib" / "contrib_root.markdown"
if _contrib_root_md.is_file():
    _scan_internal(_contrib_root_md)
for _m in CONTRIB_MODULES:
    _scan_internal(SPHINX_INPUT_ROOT / "tutorials_contrib" / _m)
# Generate/scan API stubs mirroring Doxygen's group hierarchy.
def _itertext(el) -> str:
    """Flatten an XML element's inner text. None-safe."""
    return "".join(el.itertext()).strip() if el is not None else ""


# memberdef@kind → display section title. Mirrors Doxygen's group-page order.
_MEMBERDEF_SECTIONS = (
    ("typedef",  "Typedefs"),
    ("enum",     "Enumerations"),
    ("function", "Functions"),
    ("variable", "Variables"),
    ("define",   "Macros"),
)


def _read_class_brief(refid: str, xml_dir: pathlib.Path,
                      _cache: dict = {}) -> str:
    """Read brief description from a class/struct's compound XML. Cached."""
    if refid in _cache:
        return _cache[refid]
    import xml.etree.ElementTree as _ET
    xml_path = xml_dir / f"{refid}.xml"
    brief = ""
    if xml_path.is_file():
        try:
            ccd = _ET.parse(xml_path).getroot().find("compounddef")
            if ccd is not None:
                brief = _itertext(ccd.find("briefdescription"))
        except _ET.ParseError:
            pass
    _cache[refid] = brief
    return brief


def _build_api_hierarchy(refid: str, xml_dir: pathlib.Path,
                         _seen: set | None = None) -> dict | None:
    """Walk a group XML's <innergroup> children recursively.
    Returns {name, title, detailed, innerclasses, sections, children} or None.
    `_seen` guards against the rare case of cycles in the group graph."""
    import xml.etree.ElementTree as _ET
    _seen = _seen if _seen is not None else set()
    if refid in _seen:
        return None
    _seen.add(refid)
    xml_path = xml_dir / f"{refid}.xml"
    if not xml_path.is_file():
        return None
    try:
        root = _ET.parse(xml_path).getroot()
    except _ET.ParseError:
        return None
    cd = root.find("compounddef")
    if cd is None:
        return None
    name = (cd.findtext("compoundname") or "").strip()
    title = (cd.findtext("title") or name).strip()
    # Extract detailed description for context-display.
    detailed_el = cd.find("detaileddescription")
    detailed = ""
    if detailed_el is not None:
        paras = [_itertext(p) for p in detailed_el.findall("para")]
        detailed = "\n\n".join(p for p in paras if p)
    # Inner classes (public only). One read per class's XML for its brief.
    innerclasses = []
    for ic in cd.findall("innerclass"):
        if ic.get("prot") != "public":
            continue
        ic_refid = ic.get("refid", "")
        innerclasses.append({
            "refid": ic_refid,
            "name": (ic.text or "").strip(),
            "kind": "struct" if ic_refid.startswith("struct") else "class",
            "brief": _read_class_brief(ic_refid, xml_dir),
        })
    # Section members (typedefs, enums, functions, variables, macros).
    sections: dict[str, list[dict]] = {}
    for sd in cd.findall("sectiondef"):
        for md in sd.findall("memberdef"):
            kind = md.get("kind", "")
            section_title = dict(_MEMBERDEF_SECTIONS).get(kind)
            if not section_title:
                continue
            sections.setdefault(section_title, []).append({
                "id":    md.get("id", ""),
                "name":  (md.findtext("name") or "").strip(),
                "type":  _itertext(md.find("type")),
                "args":  (md.findtext("argsstring") or "").strip(),
                "brief": _itertext(md.find("briefdescription")),
            })
    # Recurse into subgroups.
    children = []
    for ig in cd.findall("innergroup"):
        child = _build_api_hierarchy(ig.get("refid"), xml_dir, _seen)
        if child is not None:
            children.append(child)
    return {"name": name, "title": title, "detailed": detailed,
            "innerclasses": innerclasses, "sections": sections,
            "children": children}


def _md_escape_cell(text: str) -> str:
    """Make `text` safe for a single Markdown table cell."""
    # Newlines collapse to spaces, pipes escape, angle brackets stay.
    return (text or "").replace("\n", " ").replace("\r", " ") \
                       .replace("|", "\\|").strip()


def _write_api_stub(node: dict, out_dir: pathlib.Path) -> None:
    """Write API stubs: index pages for parent groups, summary pages for leaves."""
    name = node["name"]
    title = node["title"]
    out = out_dir / f"{name}.md"

    if node["children"]:
        # List children for navigation index pages.
        lines = [f"# {title} {{#api_{name}}}", ""]
        if node["detailed"]:
            lines += [node["detailed"], ""]
        lines += ["## Topics", ""]
        for child in node["children"]:
            lines.append(f"- @subpage api_{child['name']}")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for child in node["children"]:
            _write_api_stub(child, out_dir)
        return

    # ---- Leaf page ----------------------------------------------------------
    lines = [f"# {title} {{#api_{name}}}", ""]

    # Summary tables mapping class links to breathe anchors.
    if node["innerclasses"]:
        lines += ["## Classes", "",
                  "| Name | Description |", "|---|---|"]
        for c in node["innerclasses"]:
            link = f"[`{c['kind']} {c['name']}`](#{c['refid']})"
            lines.append(f"| {link} | {_md_escape_cell(c['brief'])} |")
        lines.append("")

    # Section tables in Doxygen's order.
    for _, section_title in _MEMBERDEF_SECTIONS:
        items = node["sections"].get(section_title, [])
        if not items:
            continue
        lines.append(f"## {section_title}")
        lines.append("")
        if section_title == "Functions":
            lines += ["| Return | Name | Description |", "|---|---|---|"]
            for m in items:
                ret = _md_escape_cell(m["type"]) or "&nbsp;"
                sig_link = f"[`{m['name']}{_md_escape_cell(m['args'])}`](#{m['id']})"
                lines.append(
                    f"| `{ret}` | {sig_link} | {_md_escape_cell(m['brief'])} |")
        elif section_title in ("Typedefs", "Variables"):
            lines += ["| Type | Name | Description |", "|---|---|---|"]
            for m in items:
                t = _md_escape_cell(m["type"]) or "&nbsp;"
                name_link = f"[`{m['name']}`](#{m['id']})"
                lines.append(f"| `{t}` | {name_link} | {_md_escape_cell(m['brief'])} |")
        else:  # Enumerations, Macros
            lines += ["| Name | Description |", "|---|---|"]
            for m in items:
                name_link = f"[`{m['name']}`](#{m['id']})"
                lines.append(f"| {name_link} | {_md_escape_cell(m['brief'])} |")
        lines.append("")

    # Inject breathe directive for detailed descriptions.
    lines += ["## Detailed Description", "",
              f"```{{doxygengroup}} {name}",
              ":project: opencv",
              "```"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_api_stubs(modules, xml_dir, out_dir):
    """Generate the full api/ stub tree. Idempotent — wipes and regenerates
    on every sphinx-build so stale stubs from removed modules disappear."""
    if not modules:
        return
    if not xml_dir.is_dir():
        return  # No XML yet (sphinx-xml not run); degrade silently.
    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root_lines = [
        "API Reference {#api_root}",
        "=============",
        "",
        "Sphinx-rendered API reference for OpenCV main modules. Each entry",
        "below is a module's umbrella `@defgroup`; sub-pages mirror the",
        "Doxygen subgroup hierarchy.",
        "",
    ]
    for m in modules:
        tree = _build_api_hierarchy(
            "group__" + m.replace("_", "__"), xml_dir)
        if tree is None:
            continue
        root_lines.append(f"- @subpage api_{tree['name']}")
        _write_api_stub(tree, out_dir)
    (out_dir / "api_root.markdown").write_text(
        "\n".join(root_lines) + "\n", encoding="utf-8")


if API_MODULES:
    _generate_api_stubs(API_MODULES, _API_XML_DIR, SPHINX_INPUT_ROOT / "api")
    # Recursive scan picks up api_root.markdown + every group stub.
    _scan_internal(SPHINX_INPUT_ROOT / "api")

# Scan external modules directly from DOC_ROOT.
for _toc in (DOC_ROOT / "tutorials").glob("*/table_of_content_*.markdown"):
    if _toc.parent.name not in DOC_MODULES:
        _scan_external(_toc)

# Build flat index for resolve image paths from source trees.
_IMAGE_INDEX: dict[str, str] = {}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp", ".webp"}
for _img in (DOC_ROOT / "tutorials").rglob("images/*"):
    if _img.is_file():
        _IMAGE_INDEX.setdefault(_img.name,
                                _img.relative_to(DOC_ROOT).as_posix())
for _m in CONTRIB_MODULES:
    # Support contrib images structured like main modules.
    _tut = CONTRIB_ROOT / _m / "tutorials"
    if _tut.is_dir():
        for _img in _tut.rglob("images/*"):
            if _img.is_file():
                _rel = _img.relative_to(_tut).as_posix()
                _IMAGE_INDEX.setdefault(_img.name,
                                        f"tutorials_contrib/{_m}/{_rel}")
    # Support remaining contrib images externally without copying via html_extra_path.
    for _sub in ("doc", "samples"):
        _src = CONTRIB_ROOT / _m / _sub
        if _src.is_dir():
            for _img in _src.rglob("*"):
                if _img.is_file() and _img.suffix.lower() in _IMAGE_EXTS:
                    _rel = _img.relative_to(CONTRIB_ROOT).as_posix()
                    _IMAGE_INDEX.setdefault(_img.name,
                                            f"contrib_modules/{_rel}")


# Create symlinks for html_extra_path to serve contrib assets directly.
html_extra_path: list[str] = []
def _in_source_tree(p: pathlib.Path) -> bool:
    for _root in (DOC_ROOT, CONTRIB_ROOT):
        try: p.relative_to(_root); return True
        except ValueError: pass
    return False
if not _in_source_tree(SPHINX_INPUT_ROOT):
    _extras = SPHINX_INPUT_ROOT.parent / "contrib_extras"
    _prefix = _extras / "contrib_modules"
    _prefix.mkdir(parents=True, exist_ok=True)
    for _m in CONTRIB_MODULES:
        _src, _link = CONTRIB_ROOT / _m, _prefix / _m
        if _src.is_dir() and not _link.exists():
            try: _os.symlink(_src, _link, target_is_directory=True)
            except (OSError, NotImplementedError): pass
    html_extra_path = [str(_extras)]

_TOGGLE_LABELS = {"cpp": "C++", "java": "Java", "python": "Python"}


# Resolution bases for snippet paths, mirroring Doxygen's EXAMPLE_PATH.
_SNIPPET_BASES = [
    OPENCV_ROOT,
    OPENCV_ROOT / "samples",
    OPENCV_ROOT / "apps",
] + [CONTRIB_ROOT / _m / "samples" for _m in CONTRIB_MODULES]

# Map basenames to recursive paths for simple snippet lookups.
_SNIPPET_EXTENSIONS = {
    ".cpp", ".hpp", ".h", ".c", ".cc", ".cxx",
    ".py", ".java", ".kt", ".scala", ".clj", ".groovy",
    ".sh", ".bash", ".bat", ".ps1",
    ".cmake", ".gradle",
    ".xml", ".yaml", ".yml", ".json", ".html", ".css",
    ".js", ".ts", ".rb",
}
_SNIPPET_INDEX: dict[str, pathlib.Path] = {}
_snippet_scan_roots = [OPENCV_ROOT / "samples", OPENCV_ROOT / "apps"] + [
    CONTRIB_ROOT / _m / "samples" for _m in CONTRIB_MODULES]
for _root in _snippet_scan_roots:
    if _root.is_dir():
        for _f in _root.rglob("*"):
            if _f.is_file() and _f.suffix.lower() in _SNIPPET_EXTENSIONS:
                _SNIPPET_INDEX.setdefault(_f.name, _f)


# Map unrecognized language labels to safe Pygments equivalents.
_LANG_ALIASES = {
    "none": "text",
    "unparsed": "text",
    "guess": "text",
    "gradle": "groovy",
    # Map tutorial `run` commands to shell styling.
    "run": "bash",
}

def _normalize_lang(lang: str) -> str:
    lang = (lang or "").strip(".").strip().lower() or "text"
    return _LANG_ALIASES.get(lang, lang)


def _read_snippet(rel_path: str, label: str | None) -> tuple[str, str]:
    """Return (code_text, language) for an @include / @snippet directive."""
    # Strip leading slash to retain snippet base resolution.
    rel_norm = rel_path.lstrip("/")
    p = next((b / rel_norm for b in _SNIPPET_BASES
              if (b / rel_norm).is_file()), None)
    # Fallback to basename lookup if direct path fails.
    if p is None:
        hit = _SNIPPET_INDEX.get(pathlib.Path(rel_norm).name)
        if hit and hit.is_file():
            p = hit
    if p is None:
        return f"// not found: {rel_path}\n", "text"
    text = p.read_text(encoding="utf-8", errors="replace")
    ext = p.suffix.lower()
    lang = {".cpp": "cpp", ".hpp": "cpp", ".h": "cpp", ".c": "c",
            ".py": "python", ".java": "java",
            ".xml": "xml", ".html": "html",
            ".sh": "bash", ".bash": "bash"}.get(ext, "text")
    if label is None:
        return text, lang
    # Match inline labels after comment markers (//, #, <!--, etc.).
    pat = re.compile(r"^[^\[\n]*(?://!|//|##|#|<!--)[^\[\n]*\[" + re.escape(label)
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


def _emit_toggles(tabs: list[tuple[str, str]]) -> str:
    if HAVE_SPHINX_DESIGN:
        out = ["", "``````{tab-set}"]
        for lang, body in tabs:
            label = _TOGGLE_LABELS.get(lang, lang.title())
            out += [f"`````{{tab-item}} {label}", body, "`````"]
        out += ["``````", ""]
        return "\n".join(out)
    # Fallback: render each toggle as a labeled section.
    out = [""]
    for lang, body in tabs:
        label = _TOGGLE_LABELS.get(lang, lang.title())
        out += [f"**{label}**", "", body, ""]
    return "\n".join(out)


def _translate(text: str, docname: str | None = None) -> str:
    # Stash @verbatim regions safely from further processing.
    _verbatim_stash: dict[str, str] = {}
    def _verbatim_save(body: str, inline: bool) -> str:
        key = f"VERBATIM_{len(_verbatim_stash)}"
        if inline:
            _verbatim_stash[key] = f"`{body.strip()}`"
        else:
            _verbatim_stash[key] = f"\n```text\n{body.strip()}\n```\n"
        return key
    # Block form (markers on separate lines) — run first; DOTALL across body.
    text = re.sub(
        r"@verbatim[ \t]*\n(?P<body>.*?)\n[ \t]*@endverbatim",
        lambda m: _verbatim_save(m.group("body"), inline=False),
        text, flags=re.DOTALL)
    # Inline form (both markers on the same line).
    text = re.sub(
        r"@verbatim[ \t]+(?P<body>[^\n]+?)[ \t]+@endverbatim",
        lambda m: _verbatim_save(m.group("body"), inline=True),
        text)

    # Convert header anchors to MyST label format.
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

    # Fix trailing setext headings at EOF explicitly.
    text = re.sub(
        r"^(?P<title>[^\n#=\-][^\n]*?)[ \t]*\n(?P<bar>[=\-])[=\-]{2,}[ \t]*$\s*\Z",
        lambda m: f"{'#' if m.group('bar') == '=' else '##'} {m.group('title').strip()}\n",
        text, flags=re.MULTILINE)

    # 1c. Convert remaining mid-doc setext H1s to ATX so 1d can see them.
    text = re.sub(
        r"^(?P<title>[^\n#=\-][^\n]*?)[ \t]*\n=[=]{2,}[ \t]*$",
        lambda m: f"# {m.group('title').strip()}",
        text, flags=re.MULTILINE)

    # Demote extra H1 headings to ensure proper single-title outlines.
    def _demote_extra_h1s(src: str) -> str:
        fence_open_re = re.compile(r'^[ \t]*(?:`{3,}|~{3,})')
        atx_h1_re = re.compile(r'^#\s')
        h1_count = 0
        in_fence = False
        out = []
        for line in src.split('\n'):
            if fence_open_re.match(line):
                in_fence = not in_fence
                out.append(line)
                continue
            if in_fence:
                out.append(line)
                continue
            if atx_h1_re.match(line):
                h1_count += 1
                if h1_count > 1:
                    line = '#' + line   # H1 → H2
            out.append(line)
        return '\n'.join(out)
    text = _demote_extra_h1s(text)

    # 2. Doxygen LaTeX math markers
    text = re.sub(r"\\f\[(.+?)\\f\]",
                  lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n",
                  text, flags=re.DOTALL)
    text = re.sub(r"\\f\$(.+?)\\f\$", lambda m: f"${m.group(1)}$",
                  text, flags=re.DOTALL)

    # Translate \bordermatrix{...} to widely supported `matrix` blocks.
    text = re.sub(
        r"\\bordermatrix\s*\{([^}]*)\}",
        lambda m: r"\begin{matrix}" + m.group(1).replace(r"\cr", r"\\")
                  + r"\end{matrix}",
        text)

    # Convert @code to fenced blocks while preserving indentation.
    def _code_repl(m: re.Match) -> str:
        indent = m.group("indent") or ""
        lang = _normalize_lang(m.group("lang") or "")
        body = m.group("body")
        if indent:
            body = _textwrap.dedent(body).strip("\n")
            body = "\n".join((indent + line) if line else line
                             for line in body.split("\n"))
            return f"\n{indent}```{lang}\n{body}\n{indent}```\n"
        return f"\n```{lang}\n{body.strip()}\n```\n"
    text = re.sub(
        r"^(?P<indent>[ \t]*)@code(?:\{(?P<lang>[^}]*)\})?\s*\n(?P<body>.*?)\n[ \t]*@endcode",
        _code_repl, text, flags=re.DOTALL | re.MULTILINE)

    # Convert plain Markdown fences with Doxygen language spec.
    text = re.sub(
        r"^(?P<fence>`{3,})(?P<lang>\.?[\w-]+)[ \t]*$",
        lambda m: f"{m.group('fence')}{_normalize_lang(m.group('lang'))}",
        text, flags=re.MULTILINE)

    # Plain backtick fence with leading indent applied to every body
    # line so the fence stays inside an enclosing list-item scope.
    # ({code-block} and `:::` colon-fence forms break inside tab-items.)
    def _emit_codeblock(indent: str, lang: str, body: str) -> str:
        body_lines = body.rstrip().splitlines()
        body_indented = "\n".join(indent + line for line in body_lines)
        return f"\n{indent}```{lang}\n{body_indented}\n{indent}```\n"

    # 4. @include path  /  @includelineno path
    #    (line numbering hint is dropped — MyST fenced blocks don't take :linenos:
    #    and PyData's code-block styling is already legible without it.)
    def _include_repl(m: re.Match) -> str:
        code, lang = _read_snippet(m.group("path"), None)
        return _emit_codeblock(m.group("indent") or "", lang, code)
    text = re.sub(r"^(?P<indent>[ \t]*)@include(?:lineno)?\s+(?P<path>\S+)",
                  _include_repl, text, flags=re.MULTILINE)

    # 4b. Remove stray @snippet that immediately follows @end_toggle at the same
    #     indent (no blank line between them). These are Doxygen fallback snippets
    #     for non-toggle Doxygen mode; the Sphinx build already shows them inside
    #     the tab-set, so the stray duplicate must be dropped before step 5
    #     would otherwise emit a second copy at document level.
    text = re.sub(
        r"(^([ \t]*)@end_toggle[ \t]*\n)\2@snippet[^\n]*\n",
        r"\1",
        text, flags=re.MULTILINE)

    # 5. @snippet path [Label]
    def _snippet_repl(m: re.Match) -> str:
        code, lang = _read_snippet(m.group("path"), m.group("label"))
        return _emit_codeblock(m.group("indent") or "", lang, code)
    text = re.sub(
        r"^(?P<indent>[ \t]*)@snippet\s+(?P<path>\S+)\s+(?P<label>[^\n]+?)\s*$",
        _snippet_repl, text, flags=re.MULTILINE)

    # 6. @add_toggle_LANG ... @end_toggle  (coalesce runs into one tab-set)
    def _toggle_collapse(src: str) -> str:
        out, i = [], 0
        opener = re.compile(r"^\s*@add_toggle_(\w+)\s*$", re.MULTILINE)
        while True:
            m = opener.search(src, i)
            if not m:
                out.append(src[i:]); break
            out.append(src[i:m.start()])
            tabs, j = [], m.start()
            while True:
                m2 = re.match(
                    r"\s*@add_toggle_(\w+)\s*\n(.*?)\n\s*@end_toggle\s*\n?",
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
            out.append(_emit_toggles(tabs))
            i = j
        return "".join(out)
    text = _toggle_collapse(text)

    # 7. @ref name [optional "Display Text"]
    def _ref_repl(m: re.Match) -> str:
        name = m.group("name"); disp = m.group("disp")
        target = _ANCHOR_TO_DOC.get(name)
        if target:
            return f"[{disp or name}]({'/' + target})"
        return f"[{disp or name}](#{name})"
    # Allow qualified C++ identifiers (with `:`) in `@ref` targets.
    text = re.sub(r'@ref\s+(?P<name>[\w:-]+)(?:\s+"(?P<disp>[^"]+)")?',
                  _ref_repl, text)

    # 8. @cite KEY -> [KEY]
    text = re.sub(r"@cite\s+([\w-]+)", r"[\1]", text)

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

    # Map @note, @see, @warning to MyST admonitions following Doxygen paragraph rules.
    _ADMON_KIND = {"note": "note", "see": "seealso", "warning": "warning"}
    def _admon_repl(m: re.Match) -> str:
        kind = _ADMON_KIND[m.group("dir")]
        body = m.group("body").strip()
        return f"\n:::{{{kind}}}\n{body}\n:::\n"
    text = re.sub(
        r"^@(?P<dir>note|see|warning)\s+(?P<body>.+?)(?=\n[ \t]*\n|\n@[A-Za-z]|\Z)",
        _admon_repl, text, flags=re.DOTALL | re.MULTILINE)

    # Dedent descriptions after `- @subpage X` to prevent accidental code blocks.
    def _dedent_subpage_descriptions(src: str) -> str:
        # Accept CommonMark continuation indents (4 spaces, tabs, etc.).
        pat = re.compile(
            r"^(?P<bullet>[ \t]*-\s+[^\n]*@subpage\s+[\w-]+[^\n]*)\n"
            r"(?P<desc>(?:[ \t]*\n|(?:\t|[ \t]{4,})[^\n]+(?:\n|$))+)",
            re.MULTILINE)
        def repl(m: re.Match) -> str:
            desc = _textwrap.dedent(m.group("desc")).strip("\n")
            # Preserve existing line spacing for empty descriptions.
            if not desc.strip():
                return m.group(0)
            return f"{m.group('bullet')}\n\n{desc}\n\n"
        return pat.sub(repl, src)
    text = _dedent_subpage_descriptions(text)

    # Convert `@subpage` lists to toctrees, linking internal/external docs appropriately.
    def _subpage_list_to_toctree(src: str) -> str:
        pat = re.compile(
            r"((?:^[ \t]*-\s+[^\n]*?@subpage\s+[\w-]+(?:[^\n]*)\n)+)",
            re.MULTILINE)
        def repl(m: re.Match) -> str:
            entries = re.findall(r"@subpage\s+([\w-]+)", m.group(1))
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
            return f"\n```{{toctree}}\n:maxdepth: 1\n\n{body}\n```\n"
        return pat.sub(repl, src)
    text = _subpage_list_to_toctree(text)

    # 10. @next_tutorial / @prev_tutorial  -> drop
    text = re.sub(r"^@(?:next|prev)_tutorial\{[^}]*\}\s*$", "",
                  text, flags=re.MULTILINE)

    # Drop @tableofcontents as PyData theme includes a native sidebar TOC.
    text = re.sub(r"^(?:@tableofcontents|\[TOC\])\s*$", "",
                  text, flags=re.MULTILINE)

    # Strip unnecessary @cond, @endcond, @parblock, @endparblock markers.
    text = re.sub(r"^@cond\s+\S+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^@endcond\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*@parblock\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*@endparblock\s*$", "", text, flags=re.MULTILINE)

    # Replace @anchor with MyST labels `(NAME)=` for cross-referencing.
    text = re.sub(r"^@anchor\s+(?P<name>[\w-]+)\s*$",
                  lambda m: f"({m.group('name')})=",
                  text, flags=re.MULTILINE)

    # Convert Doxygen `-#` numbered lists to Markdown `1.`, preserving indent.
    text = re.sub(r"^(?P<indent>[ \t]*)-#(?P<sp>[ \t]+)",
                  lambda m: f"{m.group('indent')}1.{m.group('sp')}",
                  text, flags=re.MULTILINE)

    # Normalize over-indented bullets to prevent unintentional code blocks in MyST.
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

    # Strip invalid 4-space indents from bullet lists that follow headings.
    text = re.sub(
        r"(^#{1,6}[ \t][^\n]+\n(?:[ \t]*\n)*)((?:    [ \t]*[-*+][^\n]*\n)+)",
        lambda m: m.group(1) + re.sub(r"^    ", "", m.group(2), flags=re.MULTILINE),
        text, flags=re.MULTILINE)

    # Calculate depth-relative path for html_extra_path contrib_modules assets.
    _depth = docname.count("/") if docname else 0
    _contrib_url_prefix = ("../" * _depth) + "contrib_modules/"

    def _emit_contrib_img(rel_url: str, alt: str) -> str:
        """Emit raw HTML <img> to bypass Sphinx image processing for contrib assets."""
        src = _contrib_url_prefix + rel_url
        img = f'<img src="{src}" alt="{alt}"/>'
        if alt.startswith("Figure "):
            return (f'<figure>{img}'
                    f'<figcaption>{alt}</figcaption></figure>')
        return img

    # Resolve image paths locally first, then by basename, then via fallback.
    def _img_repl(m: re.Match) -> str:
        alt, rel = m.group("alt"), m.group("rel")
        if docname:
            parts = pathlib.Path(docname).parent.parts
            local = None
            if parts and parts[0] == "tutorials":
                local = DOC_ROOT / pathlib.Path(docname).parent / "images" / rel
            elif len(parts) >= 2 and parts[0] == "tutorials_contrib":
                # Contrib doc → resolve under <m>/tutorials/<rest>/images/.
                rest = pathlib.Path(*parts[2:]) if len(parts) > 2 else pathlib.Path()
                local = CONTRIB_ROOT / parts[1] / "tutorials" / rest / "images" / rel
            if local is not None and local.is_file():
                return f'![{alt}](images/{rel})'
        hit = _IMAGE_INDEX.get(pathlib.Path(rel).name)
        if hit:
            if hit.startswith("contrib_modules/"):
                return _emit_contrib_img(hit[len("contrib_modules/"):], alt)
            return f'![{alt}](/{hit})'
        return f'![{alt}](/tutorials/others/images/{rel})'
    text = re.sub(
        r'!\[(?P<alt>[^\]]*)\]\((?:[^)]*?/)?images/(?P<rel>[^)]+)\)',
        _img_repl, text)

    # Resolve cross-tree contrib image refs natively via html_extra_path.
    def _img_xtree(m: re.Match) -> str:
        alt, rel = m.group("alt"), m.group("rel")
        if rel.startswith("/") or "://" in rel:
            return m.group(0)
        if rel.startswith("./"):
            rel = rel[2:]
        if not docname or not docname.startswith("tutorials_contrib/"):
            return m.group(0)
        parts = pathlib.Path(docname).parent.parts
        if len(parts) < 2:
            return m.group(0)
        module = parts[1]
        for cand in (f"{module}/doc/{rel}",
                     f"{module}/{rel}",
                     rel):
            if (CONTRIB_ROOT / cand).is_file():
                return _emit_contrib_img(cand, alt)
        return m.group(0)
    text = re.sub(
        r'!\[(?P<alt>[^\]]*)\]\((?P<rel>[^)]+)\)',
        _img_xtree, text)

    # Force single paragraphs for consecutive image inclusions. Skip tables.
    text = re.sub(
        r"^(?P<line>(?!\|)[^\n]*!\[[^\]]*\]\([^)]+\)[^\n]*)\n"
        r"(?=(?!\|)[^\n]*!\[[^\]]*\]\([^)]+\))",
        r"\g<line>\n\n", text, flags=re.MULTILINE)

    # Convert Figure-prefixed images to MyST `{figure}` directives.
    text = re.sub(
        r"^(?P<indent>[ \t]*)!\[(?P<caption>Figure\s[^\]]+)\]\((?P<url>[^)]+)\)\s*$",
        lambda m: (f"{m.group('indent')}:::{{figure}} {m.group('url')}\n"
                   f"{m.group('indent')}{m.group('caption')}\n"
                   f"{m.group('indent')}:::"),
        text, flags=re.MULTILINE)

    # Wrap front-matter in `.opencv-meta-table` div.
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

    # Fallback autolink logic for bare URLs. Skips markdown links, fences, syntax.
    _fence_open_re = re.compile(r"^[ \t]*(`{3,}|~{3,})")
    _inline_code_re = re.compile(r"`[^`\n]+`")
    _bare_url_re = re.compile(
        r"(?<!\]\()(?<!<)(?<!=\")(?<!=')"
        r"https?://[^\s<>\[\]()\"']+"
    )
    def _wrap_one_url(m: re.Match) -> str:
        u = m.group(0)
        trailing = ""
        while u and u[-1] in ".,;:!?":
            trailing = u[-1] + trailing
            u = u[:-1]
        return f"<{u}>{trailing}" if u else m.group(0)
    def _wrap_outside_inline(line: str) -> str:
        # Avoid linkifying inside backticks.
        parts = _inline_code_re.split(line)
        codes = _inline_code_re.findall(line)
        result = []
        for i, p in enumerate(parts):
            result.append(_bare_url_re.sub(_wrap_one_url, p))
            if i < len(codes):
                result.append(codes[i])
        return "".join(result)
    _autolink_out, _in_fence = [], False
    for _line in text.split("\n"):
        if _fence_open_re.match(_line):
            _in_fence = not _in_fence
            _autolink_out.append(_line)
        elif _in_fence:
            _autolink_out.append(_line)
        else:
            _autolink_out.append(_wrap_outside_inline(_line))
    text = "\n".join(_autolink_out)

    # Finally, restore un-processed @verbatim blocks.
    for _vk, _vv in _verbatim_stash.items():
        text = text.replace(_vk, _vv)

    return text


def _source_read(app, docname, source):
    # Process enabled markdown documents via _translate(), leaving API docs intact.
    if not (docname.startswith("tutorials/")
            or docname.startswith("tutorials_contrib/")
            or docname.startswith("api/")):
        return
    text = source[0]
    # Append dynamic links to contrib/API modules to the master index page.
    if docname == "tutorials/tutorials":
        if CONTRIB_MODULES and "tutorial_contrib_root" in _ANCHOR_TO_DOC:
            text = text.rstrip() + "\n\n- @subpage tutorial_contrib_root\n"
        if API_MODULES and "api_root" in _ANCHOR_TO_DOC:
            text = text.rstrip() + "\n\n- @subpage api_root\n"
    source[0] = _translate(text, docname)


def setup(app):
    app.connect("source-read", _source_read)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
