"""Shared configuration & state for the OpenCV Sphinx wrapper.

Owns everything the doc-build engine reads in common: env-derived paths and
module lists, the Doxygen tag-file maps, the anchor indexes, and the small
constant tables. The sibling engines (xml_render, stubs, translate) pull these
via ``from .state import *``; conf.py imports the handful it needs for Sphinx
settings.
"""
from __future__ import annotations
import pathlib, re, os as _os, shutil as _shutil, textwrap as _textwrap

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
    for m in (_os.environ.get("OPENCV_DOC_JS_MODULES") or "js_gui,js_core,js_imgproc,js_video,js_dnn").split(",")
    if m.strip()
]

DOC_PY_MODULES = [
    m.strip()
    for m in (_os.environ.get("OPENCV_DOC_PY_MODULES") or "py_gui,py_features,py_calib3d,py_ml,py_bindings").split(",")
    if m.strip()
]

# ---------------------------------------------------------------------------
# SCOPE — contrib tree.  Folder names under opencv_contrib/modules/.
# Override via env var to avoid editing this file:
#     OPENCV_CONTRIB_MODULES=ml,bgsegm cmake --build <build> --target sphinx
# Empty list = main-only build (legacy behavior, no contrib site).
# Default list is the UNION of both merge sides — modules enabled in either
# branch stay enabled here (merge resolution: keep contributions from both).
# ---------------------------------------------------------------------------
CONTRIB_MODULES = [
    m.strip()
    for m in (_os.environ.get("OPENCV_CONTRIB_MODULES") or "ml,bgsegm,alphamat,face,bioinspired,cannops,ccalib,cnn_3dobj,cvv,dnn_objdetect,dnn_superres,gapi,xobjdetect,xstereo,xfeatures2d,xphoto,ximgproc,text,hdf,julia,line_descriptor,phase_unwrapping,structured_light").split(",")
    if m.strip()
]
CONTRIB_ROOT = pathlib.Path(
    _os.environ.get("OPENCV_CONTRIB_ROOT")
    or str(HERE.parent.parent / "opencv_contrib" / "modules")
).resolve()

# ---------------------------------------------------------------------------
# API reference modules. Override via OPENCV_API_MODULES (comma-separated).
# Default "all" auto-discovers every top-level module group from the Doxygen
# XML (see _discover_api_modules below, resolved once _API_XML_DIR is known).
# Pass an explicit list (e.g. "core,imgproc") to scope the API reference.
# ---------------------------------------------------------------------------
API_MODULES = [
    m.strip()
    for m in (_os.environ.get("OPENCV_API_MODULES") or "all").split(",")
    if m.strip()
]

# Sphinx srcdir as seen by conf.py.  CMake stages a merged tree at
# ${CMAKE_BINARY_DIR}/docs_sphinx_input/ and forwards this env var.
# Default = DOC_ROOT so ad-hoc sphinx-build runs keep working. The `or`
# idiom (rather than dict.get's default) treats an empty-string env var
# the same as unset — CMake forwards "" when contrib is disabled.
SPHINX_INPUT_ROOT = pathlib.Path(
    _os.environ.get("OPENCV_SPHINX_INPUT_ROOT") or str(DOC_ROOT)
).resolve()

# Expose enabled contrib modules via symlinks + html_extra_path.

# sphinx_design availability. Extension wiring lives in conf.py; _emit_toggles
# uses this flag to choose a {tab-set} vs a plain labeled-section fallback.
try:
    import sphinx_design as _sphinx_design  # noqa: F401
    HAVE_SPHINX_DESIGN = True
except ImportError:
    HAVE_SPHINX_DESIGN = False

# -- Breathe (Doxygen XML -> Sphinx C++ domain) -----------------------------
# Gated on API_MODULES and breathe. Falls back to canonical build_doc layout.
_API_XML_DIR = pathlib.Path(
    _os.environ.get("OPENCV_DOXYGEN_XML_DIR")
    or str(HERE.parent.parent / "build_doc" / "doc" / "doxygen" / "xml")
).resolve()
# Patched XML dir (see `_patch_namespace_xml_for_breathe` below). breathe is
# pointed at this rather than the raw Doxygen output so that functions defined
# inside `@addtogroup` regions (which Doxygen lists only as `<member refid>`
# in namespace XML, not full `<memberdef>`) become findable. The dir is built
# at sphinx-build time, mirrors the original XML via symlinks, and only the
# affected namespace XMLs are rewritten in place.
_PATCHED_XML_DIR = _API_XML_DIR.parent / "xml_for_sphinx"

def _discover_api_modules(xml_dir):
    """Return the sorted list of top-level OpenCV module names found in the
    Doxygen XML at `xml_dir`. Used to expand OPENCV_API_MODULES=all.

    A module is a Doxygen group that is (a) not nested as an <innergroup> of
    any other group (i.e. a hierarchy root) and (b) whose module name has no
    internal underscore. Rule (b) drops the low-level HAL-interface groups
    (`group__core__hal__interface__*`, `group__video__hal__interface`, …) and
    orphaned detail stubs (`group__photo__segmentation`, `group__tracking__detail`,
    `group__highgui__winrt`), which are internal and have no umbrella page —
    while keeping real modules including `3d` (emitted as `group____3d`).

    The generator maps a module name `m` back to its group via
    `"group__" + m.replace("_", "__")`, so the inverse here is
    `refid[len("group__"):].replace("__", "_")`."""
    import re as _re
    if not xml_dir.is_dir():
        return []
    groups, children = set(), set()
    for f in xml_dir.glob("group__*.xml"):
        groups.add(f.stem)  # "group__<name>"
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        children.update(_re.findall(r'<innergroup refid="(group__[A-Za-z0-9_]+)"', txt))
    modules = []
    for g in groups - children:                       # hierarchy roots only
        name = g[len("group__"):].replace("__", "_")  # inverse of generator map
        if _re.fullmatch(r"_?[A-Za-z0-9]+", name):    # no internal underscore
            modules.append(name)
    return sorted(modules)


# Expand the "all" sentinel now that the XML location is known. Falls back to
# core-only if discovery turns up nothing (e.g. XML not built yet).
if [m.lower() for m in API_MODULES] == ["all"]:
    API_MODULES = _discover_api_modules(_API_XML_DIR) or ["core"]

# Python enum/constant signatures; built by: cmake --build --target gen_opencv_python_source
_PY_SIGNATURES: dict = {}
for _pysigs_candidate in [
    _API_XML_DIR.parents[2] / "modules/python_bindings_generator/pyopencv_signatures.json",
    _os.environ.get("OPENCV_PYTHON_SIGNATURES_FILE", ""),
]:
    _pysigs_path = pathlib.Path(str(_pysigs_candidate)) if _pysigs_candidate else None
    if _pysigs_path and _pysigs_path.is_file():
        import json as _json
        _PY_SIGNATURES = _json.loads(_pysigs_path.read_text(encoding="utf-8"))
        break
del _pysigs_candidate, _pysigs_path

# -- Breathe availability ----------------------------------------------------
# Extension registration + breathe_projects config live in conf.py; here we
# only detect breathe (its absence empties API_MODULES) and apply the renderer
# fix-up breathe needs for Doxygen 1.12 XML.
HAVE_BREATHE = False
if API_MODULES:
    try:
        import breathe  # noqa: F401
        HAVE_BREATHE = True
        # Breathe 4.36 doesn't handle optional title in Doxygen 1.12 docSect2TypeSub
        from breathe.renderer import sphinxrenderer as _bsr
        _orig_visit = _bsr.SphinxRenderer.methods["docsect1"]
        def _visit_docsectN(self, node):
            if not getattr(node, "title", None):
                return self.render_iterable(node.content_)
            return _orig_visit(self, node)
        _bsr.SphinxRenderer.methods["docsect1"] = _visit_docsectN
        _bsr.SphinxRenderer.methods["docsect2"] = _visit_docsectN
        _bsr.SphinxRenderer.methods["docsect3"] = _visit_docsectN
    except ImportError:
        API_MODULES = []

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
if not _TAG_FILE.is_file():
    _TAG_FILE = HERE.parent.parent / "build" / "doc" / "opencv.tag"

# anchor -> doxygen URL filename (from opencv.tag if available).
_TAG_FILENAMES: dict[str, str] = {}
# anchor -> human-readable page title (from opencv.tag).
_TAG_PAGE_TITLES: dict[str, str] = {}
_CV_API: dict[str, str] = {}
if _TAG_FILE.is_file():
    try:
        import xml.etree.ElementTree as _ET
        for _c in _ET.parse(str(_TAG_FILE)).getroot().iter("compound"):
            if _c.get("kind") == "page":
                _n, _f = _c.findtext("name"), _c.findtext("filename")
                _t = _c.findtext("title", "")
                if _n and _f:
                    _TAG_FILENAMES[_n] = _f if _f.endswith(".html") else _f + ".html"
                    if _t:
                        _TAG_PAGE_TITLES[_n] = _t
            if _c.get("kind") == "class":
                _cn = (_c.findtext("name") or "").split("::")[-1]
                _cf = _c.findtext("filename", "")
                if _cn and _cf:
                    _CV_API.setdefault(_cn, DOXYGEN_BASE_URL + (_cf if _cf.endswith(".html") else _cf + ".html"))
            for _m in _c.findall("member"):
                _n = _m.findtext("name", "")
                _af = _m.findtext("anchorfile", "")
                _an = _m.findtext("anchor", "")
                if _n and _af and _an:
                    _CV_API.setdefault(_n, DOXYGEN_BASE_URL + _af + "#" + _an)
    except Exception:
        pass

def _doxygen_url(page: str) -> str:
    return DOXYGEN_BASE_URL + _TAG_FILENAMES.get(page, page)


# -- Live (docs.opencv.org) tagfile for API stub URL construction ----------
# The local Doxygen build runs with CREATE_SUBDIRS=NO (Breathe XML can't
# handle subdirs), so its tagfile filenames are flat like
# `group__core__basic.html` — which 404 on docs.opencv.org, where pages are
# served under hash-based subdirectories (e.g. `dc/d84/group__core__basic.html`).
# The live tagfile published at https://docs.opencv.org/5.x/opencv.tag has the
# subdir prefixes baked in. Fetch it once:
#     curl https://docs.opencv.org/5.x/opencv.tag \
#       -o <build>/doc/doxygen/opencv-live.tag
# and the API stub link rewriter (steps 8a/8b in _translate) will pick it up.
# Falls back silently to the flat URL form when not present.
_LIVE_TAG_FILE = pathlib.Path(_os.environ.get(
    "OPENCV_DOXYGEN_LIVE_TAGFILE",
    str(HERE.parent.parent / "build" / "doc" / "doxygen" / "opencv-live.tag"),
))
if not _LIVE_TAG_FILE.is_file():
    for _alt in (
        HERE.parent.parent / "build" / "build_contrib" / "build_contrib"
            / "doc" / "doxygen" / "opencv-live.tag",
    ):
        if _alt.is_file():
            _LIVE_TAG_FILE = _alt
            break

_LIVE_GROUP_URL: dict[str, str] = {}   # 'group__core__basic' -> live URL
_LIVE_CLASS_URL: dict[str, str] = {}   # 'Matx' -> live URL
_LIVE_TYPEDEF_URL: dict[str, str] = {} # 'uchar' -> live URL (group anchor)
if _LIVE_TAG_FILE.is_file():
    try:
        import xml.etree.ElementTree as _ET
        for _c in _ET.parse(str(_LIVE_TAG_FILE)).getroot().iter("compound"):
            _kind = _c.get("kind")
            _n = _c.findtext("name") or ""
            _f = _c.findtext("filename") or ""
            if not (_n and _f):
                continue
            _fn = _f if _f.endswith(".html") else _f + ".html"
            if _kind == "group":
                # Source-markdown anchors use the Doxygen FILENAME style for
                # the group identifier (every `_` in the name is doubled —
                # e.g. tagfile name 'core_basic' becomes filename
                # 'group__core__basic.html'). Key by the filename's basename
                # so the anchor pattern `group__<name>_1<hash>` looks up
                # cleanly.
                _basename = pathlib.PurePosixPath(_fn).name[:-5]  # strip .html
                _LIVE_GROUP_URL[_basename] = DOXYGEN_BASE_URL + _fn
            elif _kind == "class":
                _short = _n.split("::")[-1]
                _LIVE_CLASS_URL.setdefault(_short, DOXYGEN_BASE_URL + _fn)
            # Collect typedef members from any compound (group, namespace,
            # file). Maps `uchar` -> live anchor URL, used by the api/
            # core_basic Type-column linkification to make tokens like
            # `uchar` inside `Vec< uchar, 2 >` clickable just like on the
            # original Doxygen page.
            for _mem in _c.findall("member"):
                if _mem.get("kind") != "typedef":
                    continue
                _mn = (_mem.findtext("name") or "").strip()
                _maf = (_mem.findtext("anchorfile") or "").strip()
                _man = (_mem.findtext("anchor") or "").strip()
                if _mn and _maf and _man:
                    _LIVE_TYPEDEF_URL.setdefault(
                        _mn, f"{DOXYGEN_BASE_URL}{_maf}#{_man}")
    except Exception:
        pass


# -- Class template-parameter display map -----------------------------------
# Maps a class short name (e.g. 'Mat_', 'Vec', 'Matx') to its template
# parameter list as Doxygen would render it (e.g. '< _Tp >', '< _Tp, cn >').
# Read from the LOCAL Doxygen XML (which contains `<templateparamlist>` per
# class). Empty `declname` on a `typename`/`class` param defaults to `_Tp`
# (OpenCV-wide convention — every untemplated `template<typename>` class
# names its param `_Tp` in the source). Used only by the api/core_basic
# Classes-table rewrite — see _translate step 8d.
_CLASS_TEMPLATE_DISPLAY: dict[str, str] = {}
if _API_XML_DIR.is_dir():
    try:
        import xml.etree.ElementTree as _ET
        for _xml in _API_XML_DIR.glob("classcv_1_1*.xml"):
            try:
                _cd = _ET.parse(_xml).getroot().find("compounddef")
            except _ET.ParseError:
                continue
            if _cd is None:
                continue
            _tpl = _cd.find("templateparamlist")
            if _tpl is None:
                continue
            _names = []
            for _p in _tpl.findall("param"):
                _decl = (_p.findtext("declname")
                         or _p.findtext("defname") or "").strip()
                _type = (_p.findtext("type") or "").strip()
                if _decl:
                    _names.append(_decl)
                elif _type in ("typename", "class"):
                    _names.append("_Tp")
                elif _type:
                    _names.append(_type)
            if _names:
                _name = (_cd.findtext("compoundname") or "").split("::")[-1]
                _CLASS_TEMPLATE_DISPLAY[_name] = f"< {', '.join(_names)} >"
    except Exception:
        pass


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

def _scan_internal(path: pathlib.Path, base: pathlib.Path | None = None) -> None:
    """Add every {#anchor} and standalone `@anchor NAME` in `path` (file or
    dir) to _ANCHOR_TO_DOC. Docname is computed relative to `base` (default
    SPHINX_INPUT_ROOT) so the same scanner serves both main and contrib.
    Accepts both `.markdown` and `.md` extensions."""
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
        # Use the unresolved path so symlinks in the staged input tree
        # produce docnames relative to the staging root, not to their
        # real source location (opencv/doc/ or opencv_contrib/modules/).
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

# Basename indexes populated at import time by the build module (kept here so
# the translation engine and the build orchestrator share the same dict
# objects).
_IMAGE_INDEX: dict[str, str] = {}
_SNIPPET_INDEX: dict[str, pathlib.Path] = {}

_TOGGLE_LABELS = {"cpp": "C++", "java": "Java", "python": "Python"}


# Mirror of Doxygen's EXAMPLE_PATH (see opencv/doc/Doxyfile.in) — the bases a
# bare `@snippet some/path.cpp` is resolved against. OPENCV_ROOT comes first so
# fully-qualified paths like `samples/cpp/...` keep working.
_SNIPPET_BASES = [
    OPENCV_ROOT,
    OPENCV_ROOT / "samples",
    OPENCV_ROOT / "apps",
] + [CONTRIB_ROOT / _m / "samples" for _m in CONTRIB_MODULES]

# Doxygen accepts language names that Pygments doesn't recognize (or wraps
# them with a leading `.` in the `@code{.lang}` and ```.lang fenced forms).
# Strip the dot and remap a few aliases so Pygments stays warning-free.
_LANG_ALIASES = {
    "none": "text",
    "unparsed": "text",
    "guess": "text",
    "gradle": "groovy",
    "csv": "text",
    # `run` is a custom convention some contrib tutorials use to mean
    # "this is a shell command you run" (e.g. dnn_superres/upscale_image_*).
    # Pygments has no `run` lexer — map to bash so it highlights as shell.
    "run": "bash",
}

__all__ = [
    "HERE", "DOC_ROOT", "OPENCV_ROOT",
    "DOC_MODULES", "DOC_JS_MODULES", "DOC_PY_MODULES",
    "CONTRIB_MODULES", "CONTRIB_ROOT", "SPHINX_INPUT_ROOT", "API_MODULES",
    "_API_XML_DIR", "_PATCHED_XML_DIR",
    "HAVE_SPHINX_DESIGN", "HAVE_BREATHE",
    "DOXYGEN_BASE_URL", "_doxygen_url",
    "_TAG_FILE", "_TAG_FILENAMES", "_TAG_PAGE_TITLES", "_CV_API",
    "_LIVE_GROUP_URL", "_LIVE_CLASS_URL", "_LIVE_TYPEDEF_URL",
    "_CLASS_TEMPLATE_DISPLAY", "_PY_SIGNATURES",
    "_ANCHOR_TO_DOC", "_ANCHOR_TO_EXTERNAL", "_HEAD_RE",
    "_scan_internal", "_scan_external",
    "_IMAGE_INDEX", "_SNIPPET_INDEX", "_SNIPPET_BASES",
    "_TOGGLE_LABELS", "_LANG_ALIASES",
]
