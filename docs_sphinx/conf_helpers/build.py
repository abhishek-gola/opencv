"""Import-time orchestration: populate the shared indexes.

Importing this module runs the one-time build steps in their original order:
scan tutorial / contrib anchors, generate the API stub tree, then build the
image and snippet basename indexes. All results land in the shared maps owned
by ``state``. conf.py imports this module purely for its import-time effect.
"""
from __future__ import annotations
import pathlib, re, os as _os, shutil as _shutil, textwrap as _textwrap
from .state import *
from .xml_render import _patch_namespace_xml_for_breathe
from .stubs import _generate_api_stubs

# Internal scan: master + every enabled main and contrib module subtree.
_scan_internal(SPHINX_INPUT_ROOT / "tutorials" / "tutorials.markdown")
for _m in DOC_MODULES:
    _scan_internal(SPHINX_INPUT_ROOT / "tutorials" / _m)
_contrib_root_md = SPHINX_INPUT_ROOT / "tutorials_contrib" / "contrib_root.markdown"
if _contrib_root_md.is_file():
    _scan_internal(_contrib_root_md)
for _m in CONTRIB_MODULES:
    _scan_internal(SPHINX_INPUT_ROOT / "tutorials_contrib" / _m)

if API_MODULES:
    # 1) Build a patched XML tree breathe will read (inlines group-only
    #    <memberdef>s into namespace XML so name lookups succeed).
    if _API_XML_DIR.is_dir():
        _patch_namespace_xml_for_breathe(_API_XML_DIR, _PATCHED_XML_DIR)
    # 2) Generate the api/ stub tree from the ORIGINAL XML — the stub
    #    generator only reads group XML, which is unchanged.
    _generate_api_stubs(API_MODULES, _API_XML_DIR, SPHINX_INPUT_ROOT / "api")
    # Recursive scan picks up api_root.markdown + every group stub.
    _scan_internal(SPHINX_INPUT_ROOT / "api")

# External scan: every OTHER module's top-level table_of_content_*.markdown.
for _toc in (DOC_ROOT / "tutorials").glob("*/table_of_content_*.markdown"):
    if _toc.parent.name not in DOC_MODULES:
        _scan_external(_toc)

_scan_internal(DOC_ROOT / "js_tutorials" / "js_tutorials.markdown", base=DOC_ROOT)
for _m in DOC_JS_MODULES:
    _scan_internal(DOC_ROOT / "js_tutorials" / _m, base=DOC_ROOT)
for _toc in (DOC_ROOT / "js_tutorials").glob("*/js_table_of_contents_*.markdown"):
    if _toc.parent.name not in DOC_JS_MODULES:
        _scan_external(_toc)

_scan_internal(DOC_ROOT / "py_tutorials" / "py_tutorials.markdown", base=DOC_ROOT)
for _m in DOC_PY_MODULES:
    _scan_internal(DOC_ROOT / "py_tutorials" / _m, base=DOC_ROOT)
for _toc in (DOC_ROOT / "py_tutorials").glob("*/py_table_of_contents_*.markdown"):
    if _toc.parent.name not in DOC_PY_MODULES:
        _scan_external(_toc)

for _m in CONTRIB_MODULES:
    _tut_dir = CONTRIB_ROOT / _m / "tutorials"
    if not _tut_dir.is_dir():
        continue
    for _md in list(_tut_dir.rglob("*.markdown")) + list(_tut_dir.rglob("*.md")):
        try:
            _head = _md.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:
            continue
        _rel = "tutorials_contrib/" + _m + "/" + _md.relative_to(_tut_dir).with_suffix("").as_posix()
        for _mm in re.finditer(r"\{#([\w-]+)\}", _head):
            _ANCHOR_TO_DOC[_mm.group(1)] = _rel

# Image basename index (mirrors Doxygen's flat IMAGE_PATH lookup).
for _img_tree in (DOC_ROOT / "tutorials", DOC_ROOT / "js_tutorials", DOC_ROOT / "py_tutorials"):
    for _img in _img_tree.rglob("images/*"):
        if _img.is_file():
            _IMAGE_INDEX.setdefault(_img.name, _img.relative_to(DOC_ROOT).as_posix())
for _img in (DOC_ROOT / "js_tutorials" / "js_assets").glob("*"):
    if _img.is_file():
        _IMAGE_INDEX.setdefault(_img.name, _img.relative_to(DOC_ROOT).as_posix())
for _img in (DOC_ROOT / "images").glob("*"):
    if _img.is_file():
        _IMAGE_INDEX.setdefault(_img.name, _img.relative_to(DOC_ROOT).as_posix())
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp", ".webp"}
for _m in CONTRIB_MODULES:
    _tut_dir = CONTRIB_ROOT / _m / "tutorials"
    if _tut_dir.is_dir():
        for _img in _tut_dir.rglob("*"):
            if _img.is_file() and _img.suffix.lower() in _IMAGE_EXTS:
                _rel_str = "tutorials_contrib/" + _m + "/" + _img.relative_to(_tut_dir).as_posix()
                _IMAGE_INDEX.setdefault(_img.name, _rel_str)
    # Contrib images outside <m>/tutorials/. URL is contrib_modules/<m>/<rest>.
    # Files served via html_extra_path — no copies in srcdir.
    for _sub in ("doc", "samples"):
        _src = CONTRIB_ROOT / _m / _sub
        if _src.is_dir():
            for _img in _src.rglob("*"):
                if _img.is_file() and _img.suffix.lower() in _IMAGE_EXTS:
                    _rel = _img.relative_to(CONTRIB_ROOT).as_posix()
                    _IMAGE_INDEX.setdefault(_img.name,
                                            f"_contrib_images/{_rel}")


# Snippet basename index (mirrors Doxygen EXAMPLE_RECURSIVE lookup).
_SNIPPET_EXTENSIONS = {
    ".cpp", ".hpp", ".h", ".c", ".cc", ".cxx",
    ".py", ".java", ".kt", ".scala", ".clj", ".groovy",
    ".sh", ".bash", ".bat", ".ps1",
    ".cmake", ".gradle",
    ".xml", ".yaml", ".yml", ".json", ".html", ".css",
    ".js", ".ts", ".rb",
}
_snippet_scan_roots = [OPENCV_ROOT / "samples", OPENCV_ROOT / "apps"] + [
    CONTRIB_ROOT / _m / "samples" for _m in CONTRIB_MODULES]
for _root in _snippet_scan_roots:
    if _root.is_dir():
        for _f in _root.rglob("*"):
            if _f.is_file() and _f.suffix.lower() in _SNIPPET_EXTENSIONS:
                _SNIPPET_INDEX.setdefault(_f.name, _f)
