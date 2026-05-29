"""Doxygen XML -> Markdown primitives for the API-reference stubs.

Low-level readers and renderers shared by the stub writers in ``stubs``:
flatten XML text, render <detaileddescription> to Markdown, walk the group
hierarchy, read per-class data, recolour collaboration SVGs, and patch the
namespace XML so breathe can resolve @addtogroup members. Shared state (paths,
anchor maps) comes from ``state``.
"""
from __future__ import annotations
import pathlib, re, os as _os, shutil as _shutil, textwrap as _textwrap
from .state import *


def _itertext(el) -> str:
    """Flatten an XML element's inner text. None-safe."""
    return "".join(el.itertext()).strip() if el is not None else ""


def _type_to_md(type_elem) -> str:
    """Render <type> XML as markdown; turns <ref> children into links."""
    if type_elem is None:
        return ""
    out = type_elem.text or ""
    for child in type_elem:
        if child.tag == "ref":
            word = (child.text or "").strip()
            refid = child.get("refid", "")
            kindref = child.get("kindref", "")
            if kindref == "compound" and refid in _ANCHOR_TO_DOC:
                fn = _ANCHOR_TO_DOC[refid].split("/")[-1]
                out += f"[{word}]({fn}.md)"
            elif refid:
                out += f"[{word}](#{refid})"
            else:
                out += word
        out += child.tail or ""
    return out.strip()


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


def _doxygen_desc_to_md(el, h_level: int = 3) -> str:
    """Convert a Doxygen <detaileddescription> (or similar) element to Markdown."""
    if el is None:
        return ""

    def _hl_text(hl_node) -> str:
        """Extract text from a <highlight> element, treating <sp/> as a space."""
        parts = []
        if hl_node.text:
            parts.append(hl_node.text)
        for child in hl_node:
            if child.tag == "sp":
                parts.append(" ")
            elif child.tag == "ref":
                parts.append("".join(child.itertext()))
            else:
                parts.append("".join(child.itertext()))
            if child.tail:
                parts.append(child.tail)
        return "".join(parts)

    def _programlisting(node) -> str:
        lines = []
        for codeline in node.findall("codeline"):
            lines.append("".join(_hl_text(hl) for hl in codeline.findall("highlight")))
        return "```cpp\n" + "\n".join(lines) + "\n```"

    def _inline(node) -> str:
        """Render inline content of a node (inline elements only)."""
        parts = []
        if node.text:
            parts.append(node.text)
        for child in node:
            t = child.tag
            if t in _BLOCK_TAGS:
                break  # stop at block-level — caller handles separately
            inner = "".join(child.itertext())
            if t == "ulink":
                url = child.get("url", "")
                parts.append(f"[{inner}]({url})" if url else inner)
            elif t in ("ref", "computeroutput"):
                parts.append(f"`{inner}`" if inner else "")
            elif t == "emphasis":
                parts.append(f"*{inner}*" if inner else "")
            elif t in ("bold", "strong"):
                parts.append(f"**{inner}**" if inner else "")
            elif t == "sp":
                parts.append(" ")
            elif t == "linebreak":
                parts.append("\n")
            else:
                parts.append(inner)
            if child.tail:
                parts.append(child.tail)
        return "".join(parts)

    _BLOCK_TAGS = {"orderedlist", "itemizedlist", "programlisting"}

    def _listitem_text(item) -> str:
        parts = []
        for child in item:
            if child.tag == "para":
                parts.append(_inline(child).strip())
        return " ".join(p for p in parts if p)

    def _emit_block(sub, result: list, level: int) -> None:
        t = sub.tag
        if t == "programlisting":
            result.append(_programlisting(sub))
        elif t == "orderedlist":
            for i, item in enumerate(sub.findall("listitem"), 1):
                result.append(f"{i}. {_listitem_text(item)}")
        elif t == "itemizedlist":
            for item in sub.findall("listitem"):
                result.append(f"- {_listitem_text(item)}")

    def _blocks(node, level: int) -> list[str]:
        result = []
        for child in node:
            t = child.tag
            if t == "title":
                continue
            elif t == "para":
                # Walk para children; flush text before each block-level child.
                pending: list[str] = []
                if child.text:
                    pending.append(child.text)
                for sub in child:
                    if sub.tag in _BLOCK_TAGS:
                        text = "".join(pending).strip()
                        if text:
                            result.append(text)
                        pending = []
                        _emit_block(sub, result, level)
                        if sub.tail and sub.tail.strip():
                            pending.append(sub.tail)
                    else:
                        inner = "".join(sub.itertext())
                        st = sub.tag
                        if st == "ulink":
                            url = sub.get("url", "")
                            pending.append(f"[{inner}]({url})" if url else inner)
                        elif st in ("ref", "computeroutput"):
                            pending.append(f"`{inner}`" if inner else "")
                        elif st == "emphasis":
                            pending.append(f"*{inner}*" if inner else "")
                        elif st in ("bold", "strong"):
                            pending.append(f"**{inner}**" if inner else "")
                        elif st == "sp":
                            pending.append(" ")
                        elif st == "linebreak":
                            pending.append("\n")
                        else:
                            pending.append(inner)
                        if sub.tail:
                            pending.append(sub.tail)
                text = "".join(pending).strip()
                if text:
                    result.append(text)
            elif t in ("sect1", "sect2", "sect3"):
                offset = {"sect1": 0, "sect2": 1, "sect3": 2}[t]
                lv = level + offset
                title_text = child.findtext("title") or ""
                result.append(f"{'#' * lv} {title_text}")
                result.extend(_blocks(child, lv + 1))
            elif t in _BLOCK_TAGS:
                _emit_block(child, result, level)
            elif t == "simplesect":
                kind = child.get("kind", "")
                admon = {"note": "note", "warning": "warning",
                         "attention": "warning", "remark": "note"}.get(kind)
                body = "\n\n".join(_blocks(child, level))
                if admon:
                    result.append(f":::{{{admon}}}\n{body}\n:::")
                elif body:
                    result.append(body)
        return result

    return "\n\n".join(b for b in _blocks(el, h_level) if b.strip())


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
    # Detailed description (used on parent index pages; breathe handles it
    # on leaf pages, so we extract it for context-display only).
    detailed_el = cd.find("detaileddescription")
    detailed = _doxygen_desc_to_md(detailed_el, h_level=3)
    # Inner classes (public only). One read per class's XML for its brief.
    # `qualified` is what `{doxygenclass}` needs (e.g. cv::ocl::Context); the
    # innerclass element's text already carries that, but normalize spaces.
    innerclasses = []
    for ic in cd.findall("innerclass"):
        if ic.get("prot") != "public":
            continue
        ic_refid = ic.get("refid", "")
        qualified = " ".join((ic.text or "").split())
        innerclasses.append({
            "refid": ic_refid,
            "name": qualified,
            "qualified": qualified,
            "kind": "struct" if ic_refid.startswith("struct") else "class",
            "brief": _read_class_brief(ic_refid, xml_dir),
        })
    # Section members (typedefs, enums, functions, variables, macros).
    # `qualified` and `param_types` exist so we can emit per-member breathe
    # directives (doxygenenum / doxygenfunction / …) instead of one big
    # doxygengroup; the latter inlines every <innerclass> on the group page,
    # which is the opposite of Doxygen's group-page layout.
    sections: dict[str, list[dict]] = {}
    for sd in cd.findall("sectiondef"):
        for md in sd.findall("memberdef"):
            kind = md.get("kind", "")
            section_title = dict(_MEMBERDEF_SECTIONS).get(kind)
            if not section_title:
                continue
            qualified = (md.findtext("qualifiedname") or "").strip()
            if not qualified:
                qualified = (md.findtext("name") or "").strip()
            # Param types: <type> text plus any <array> suffix (Doxygen
            # stores `int foo[3]` as <type>int</type>...<array>[3]</array>;
            # without merging them our breathe disambiguator omits the `[3]`
            # and breathe reports "Unable to resolve function with
            # arguments (…, const double)" even though the function exists).
            def _param_type(p) -> str:
                t = _itertext(p.find("type"))
                arr = (p.findtext("array") or "").strip()
                return (t + arr) if arr else t
            param_types = [_param_type(p) for p in md.findall("param")]
            # Enum values + scoped-vs-unscoped flag (for Doxygen-style
            # synopsis rendering — one code block per enum with all values
            # inside `{ }`, instead of breathe's discrete signature blocks).
            enum_values = []
            is_strong = md.get("strong", "no") == "yes"
            if kind == "enum":
                for ev in md.findall("enumvalue"):
                    enum_values.append({
                        "name":        (ev.findtext("name") or "").strip(),
                        "initializer": (ev.findtext("initializer") or "").strip(),
                        "brief":       _itertext(ev.find("briefdescription")),
                    })
            sections.setdefault(section_title, []).append({
                "id":          md.get("id", ""),
                "kind":        kind,
                "name":        (md.findtext("name") or "").strip(),
                "qualified":   qualified,
                "type":        _itertext(md.find("type")),
                "args":        (md.findtext("argsstring") or "").strip(),
                "param_types": param_types,
                "brief":       _itertext(md.find("briefdescription")),
                "enum_values": enum_values,
                "strong":      is_strong,
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


# Per-member breathe directive selector. The full doxygengroup directive
# recursively inlines every <innerclass> + <innernamespace>, which is the
# *opposite* of how Doxygen's group page lays out (classes are links to
# separate pages there). Emitting one directive per member keeps each
# member's detail block scoped to itself and lets us push classes to their
# own per-class pages — see _write_class_stub.
_MEMBER_DIRECTIVE = {
    "enum":     "doxygenenum",
    "function": "doxygenfunction",
    "typedef":  "doxygentypedef",
    "variable": "doxygenvariable",
    "define":   "doxygendefine",
}
# Section header for each member kind's detail block. Mirrors what Doxygen
# emits on a group page (e.g. group__core__opencl.html has separate
# "Enumeration Type Documentation" and "Function Documentation" sections,
# not one collapsed "Detailed Description").
_MEMBER_DETAIL_SECTION = {
    "Typedefs":     "Typedef Documentation",
    "Enumerations": "Enumeration Type Documentation",
    "Functions":    "Function Documentation",
    "Variables":    "Variable Documentation",
    "Macros":       "Macro Definition Documentation",
}


def _enum_synopsis_lines(m: dict) -> list[str]:
    """Render an enum as a Doxygen-style code synopsis: one `enum {…}` block
    listing every value with its initializer. Used in place of breathe's
    `{doxygenenum}` directive, which emits a discrete signature box per
    enumerator (one box for the enum + one per value) — that's the layout
    in the user's "before" screenshot. Doxygen's group page renders the
    enum as a single code-style box; this helper reproduces that.

    Value-name qualification follows Doxygen:
      * Scoped (`enum class`) → values prefixed with the enum's own
        qualified name.
      * Unscoped → values prefixed with the enum's *parent* scope
        (namespace or enclosing class), so they look like the C++ name
        you'd actually write in code."""
    qualified = m.get("qualified") or m["name"]
    is_strong = bool(m.get("strong"))
    keyword = "enum class" if is_strong else "enum"
    if is_strong:
        prefix = qualified + "::"
    elif "::" in qualified:
        prefix = qualified.rsplit("::", 1)[0] + "::"
    else:
        prefix = ""
    out = [f"{keyword} {qualified} {{"]
    vals = m.get("enum_values", []) or []
    for i, v in enumerate(vals):
        comma = "," if i < len(vals) - 1 else ""
        init = (" " + v["initializer"]) if v["initializer"] else ""
        out.append(f"    {prefix}{v['name']}{init}{comma}")
    out.append("}")
    return out


def _function_signature(member: dict) -> str:
    """Disambiguator used after a qualified function name in `{doxygenfunction}`.
    Breathe expects `name(type1, type2, …)` with parameter names dropped (it
    matches against Doxygen's `<param><type>` text). Empty-arg functions get
    `()` — required for breathe to match correctly even for non-overloads."""
    types = ", ".join((t or "").strip() for t in member.get("param_types", []))
    sig = f"({types})"
    if member.get("const"):
        sig += " const"
    return sig


def _class_page_name(refid: str) -> str:
    """Filename (without extension) for the per-class page. We use the Doxygen
    refid verbatim so MyST cross-refs and internal links from breathe stay
    stable (breathe's class anchors are the C++-mangled `_CPPv4N…` form, not
    the refid — so there's no collision with the page name)."""
    return refid


def _read_class_data(refid: str, xml_dir: pathlib.Path) -> dict | None:
    """Walk a class/struct compound XML and return everything the per-class
    page needs: brief + detailed for the class itself, and a list of
    members grouped by sectiondef kind. Returns None if the XML file is
    missing or unparseable — callers should fall back to a bare
    `{doxygenclass}` directive in that case.

    Each member dict carries the same fields `_build_api_hierarchy`
    captures, plus the `protection`, `static`, `const`, `virtual`,
    `explicit` flags from memberdef attributes (needed to render
    Doxygen-style annotations in the summary table)."""
    import xml.etree.ElementTree as _ET
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

    def _param_type(p) -> str:
        t = _itertext(p.find("type"))
        arr = (p.findtext("array") or "").strip()
        return (t + arr) if arr else t

    sections: dict[str, list[dict]] = {}
    for sd in cd.findall("sectiondef"):
        skind = sd.get("kind", "")
        items: list[dict] = []
        for md in sd.findall("memberdef"):
            mkind = md.get("kind", "")
            qualified = (md.findtext("qualifiedname") or "").strip()
            name = (md.findtext("name") or "").strip()
            enum_values = []
            if mkind == "enum":
                for ev in md.findall("enumvalue"):
                    enum_values.append({
                        "name":        (ev.findtext("name") or "").strip(),
                        "initializer": (ev.findtext("initializer") or "").strip(),
                        "brief":       _itertext(ev.find("briefdescription")),
                    })
            items.append({
                "id":          md.get("id", ""),
                "kind":        mkind,
                "name":        name,
                "qualified":   qualified or name,
                "type":        _itertext(md.find("type")),
                "args":        (md.findtext("argsstring") or "").strip(),
                "param_types": [_param_type(p) for p in md.findall("param")],
                "brief":       _itertext(md.find("briefdescription")),
                "static":      md.get("static") == "yes",
                "virt":        md.get("virt", "non-virtual"),
                "const":       md.get("const") == "yes",
                "explicit":    md.get("explicit") == "yes",
                "enum_values": enum_values,
                "strong":      md.get("strong", "no") == "yes",
            })
        if items:
            sections[skind] = items

    detailed_el = cd.find("detaileddescription")
    detailed_paras = []
    if detailed_el is not None:
        for p in detailed_el.findall("para"):
            txt = _itertext(p)
            if txt:
                detailed_paras.append(txt)

    return {
        "name":     (cd.findtext("compoundname") or "").strip(),
        "brief":    _itertext(cd.find("briefdescription")),
        "detailed": "\n\n".join(detailed_paras),
        "sections": sections,
    }


def _find_collaboration_svg(refid: str, html_root: pathlib.Path) -> pathlib.Path | None:
    """Locate the Doxygen-generated collaboration-diagram SVG for a class.

    Our XML pipeline (CMakeLists.txt's `Doxyfile-xml`) sets
    `COLLABORATION_GRAPH = NO` and friends — graph elements in XML would be
    forwarded by breathe as `graphviz` docutils nodes, which need an
    extension we don't load. So the diagram never reaches the XML.

    The *legacy* Doxygen HTML build (the `doxygen` target, separate from
    `sphinx-xml`) still renders it as `<refid>__coll__graph.svg`, written
    into a content-addressed subdir because the legacy Doxyfile keeps
    `CREATE_SUBDIRS=YES`. The HTML tree sits next to the XML tree
    (`…/doxygen/html` ⟷ `…/doxygen/xml`). We read that asset read-only —
    nothing in the Doxygen output is modified. Returns None when the legacy
    HTML build hasn't run (graphs simply stay absent, no crash)."""
    if not html_root.is_dir():
        return None
    matches = sorted(html_root.rglob(f"{refid}__coll__graph.svg"))
    return matches[0] if matches else None


def _svg_make_transparent(text: str) -> str:
    """Light-mode variant: only the full-canvas backdrop is made transparent
    so the white page shows through (native Doxygen look). Graphviz paints the
    canvas as a single `fill="white" stroke="transparent"` polygon."""
    return text.replace('fill="white" stroke="transparent"',
                        'fill="none" stroke="transparent"', 1)


def _svg_dark_variant(text: str) -> str:
    """Dark-mode variant: recolour the (light) Doxygen SVG into a dark diagram
    matching docs.opencv.org — transparent canvas (page slate shows through),
    dark node fills, light borders/text, lightened connector arrows. We recolour
    the SVG itself (rather than a CSS `filter: invert`, which turns the large
    white node boxes solid black) so the result blends with the dark page.

    Order matters: blank the backdrop first, *then* repaint the remaining white
    node fills, so the two `fill="white"` cases don't collide."""
    text = _svg_make_transparent(text)              # backdrop → transparent
    text = text.replace('fill="white"', 'fill="#1c2128"')   # node box fills → dark slate
    text = text.replace('fill="#bfbfbf"', 'fill="#373e47"')  # header bar → darker grey
    text = text.replace('stroke="black"', 'stroke="#c9d1d9"')  # borders → light
    text = text.replace('stroke="#404040"', 'stroke="#768390"')  # arrows → lighter grey
    # Graphviz <text> has no fill attribute (defaults to black); inject a light
    # fill so labels are readable on the dark canvas.
    text = text.replace('<text ', '<text fill="#adbac7" ')
    return text


def _patch_namespace_xml_for_breathe(xml_dir: pathlib.Path,
                                     out_dir: pathlib.Path) -> None:
    """Mirror `xml_dir` into `out_dir` via symlinks, then rewrite every
    *non-group* compound XML to inline `<memberdef>` blocks that exist only
    in the group XML.

    Why: Doxygen lists functions declared inside `@addtogroup` regions as
    `<member refid="group__…">` in the parent namespace XML (for `cv::*`
    free functions) or the parent file XML (for global `hal_ni_*`-style
    functions) — *without* a full `<memberdef>` block. The memberdef lives
    only in the group XML file. breathe's function-by-name lookup walks
    `<memberdef>` blocks in namespace/file XMLs and ignores bare refs, so
    directives like `{doxygenfunction} cv::log` or
    `{doxygenfunction} hal_ni_merge8u` fail with "Cannot find function".

    Patching: for each `<member refid>` in a target compound's sectiondef
    whose id targets `group__…`, we open the group XML, find the matching
    `<memberdef id="…">`, and append it into the compound's sectiondef.
    The original XML on disk is untouched.

    Freshness guard: skip the whole rebuild if `out_dir/index.xml` is at
    least as new as `xml_dir/index.xml`. The patcher parses ~1500 compound
    XMLs on a clean run — rerunning that on every sphinx-build is what
    made incremental rebuilds feel sluggish. `sphinx-xml` updates the
    source `xml/` tree's mtimes, so a real Doxygen change always
    invalidates this cache."""
    import xml.etree.ElementTree as _ET
    import os as _osmod, shutil as _shutil
    src_index = xml_dir / "index.xml"
    # Use a dedicated stamp file (not dst_index) for the freshness check.
    # dst_index is symlinked to src_index, so stat() on it follows the
    # symlink and always returns src's mtime — making the previous
    # `dst_index.mtime >= src_index.mtime` guard ALWAYS true after the
    # first mirror, freezing the patched dir even when Doxygen regenerated
    # the source with new files. The stamp is a real file whose mtime
    # records when the LAST mirror+patch finished.
    stamp = out_dir / ".mirror_complete"
    if (src_index.is_file() and stamp.is_file()
            and stamp.stat().st_mtime >= src_index.stat().st_mtime):
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Mirror every file from xml_dir into out_dir as a symlink. Cleaning
    #    out_dir each time keeps this idempotent across rebuilds (Doxygen
    #    XML changes are picked up because the symlinks resolve fresh).
    for child in list(out_dir.iterdir()):
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            _shutil.rmtree(child)
    for src in xml_dir.iterdir():
        dst = out_dir / src.name
        try:
            _osmod.symlink(src, dst)
        except (OSError, NotImplementedError):
            _shutil.copy2(src, dst)

    # 2) Cache for parsed group XMLs (each is read once even if referenced
    #    by many compounds).
    _group_cache: dict[str, _ET.ElementTree] = {}

    def _load_group(group_id: str):
        if group_id in _group_cache:
            return _group_cache[group_id]
        gx = xml_dir / f"{group_id}.xml"
        if not gx.is_file():
            _group_cache[group_id] = None
            return None
        try:
            _group_cache[group_id] = _ET.parse(gx)
        except _ET.ParseError:
            _group_cache[group_id] = None
        return _group_cache[group_id]

    # 3) For each non-group compound XML (namespace, file, dir, …) patch
    #    `<member refid>` entries that point at group memberdefs.
    #    `index.xml` is the project index (not a compound) → skip it.
    #    `class*.xml`/`struct*.xml`/`union*.xml` already carry full
    #    memberdefs for their methods, but they may *also* have
    #    `<member refid>` from @addtogroup tagged methods — patch them too.
    # Fast pre-filter: most compound XMLs don't reference any group at all;
    # parsing+walking them is wasted work. A bytes-level substring scan is
    # ~100x faster than ET.parse and lets us short-circuit those files.
    _SKIP_PREFIXES = ("group", "index")
    _GROUP_REF_MARKER = b'refid="group__'
    for compound_file in xml_dir.glob("*.xml"):
        if any(compound_file.name.startswith(p) for p in _SKIP_PREFIXES):
            continue
        try:
            raw = compound_file.read_bytes()
        except OSError:
            continue
        if _GROUP_REF_MARKER not in raw:
            continue  # no <member refid="group__…"> → nothing to patch
        try:
            tree = _ET.ElementTree(_ET.fromstring(raw))
        except _ET.ParseError:
            continue
        root = tree.getroot()
        cd = root.find("compounddef")
        if cd is None:
            continue
        existing_ids = {md.get("id") for md in cd.iter("memberdef")}
        patched = False
        for sd in cd.findall("sectiondef"):
            for member in list(sd.findall("member")):
                refid = member.get("refid", "")
                if not refid or refid in existing_ids:
                    continue
                # Member refids inside groups look like
                # "group__core__utils__softfloat_1ga…". The compound id is
                # everything before "_1" (which separates member from
                # compound in Doxygen's id scheme).
                if not refid.startswith("group__"):
                    continue
                sep = refid.find("_1")
                if sep < 0:
                    continue
                group_id = refid[:sep]
                gtree = _load_group(group_id)
                if gtree is None:
                    continue
                for md in gtree.getroot().iter("memberdef"):
                    if md.get("id") == refid:
                        sd.append(md)
                        existing_ids.add(refid)
                        patched = True
                        break
        if patched:
            out_file = out_dir / compound_file.name
            if out_file.is_symlink() or out_file.is_file():
                out_file.unlink()
            tree.write(out_file, encoding="utf-8", xml_declaration=True)

    # 4) Record completion with a stamp file. Subsequent invocations
    #    compare this stamp's mtime against `src_index.xml` — if Doxygen
    #    has run since (added/removed/changed XMLs), the stamp is older
    #    and a full re-mirror is triggered.
    stamp.touch()


def _collect_all_nodes(node: dict) -> list[str]:
    return [node["name"]] + [n for c in node["children"] for n in _collect_all_nodes(c)]


def _build_ns_group_map(all_refids: list[str],
                        xml_dir: pathlib.Path) -> dict[str, set]:
    """Return namespace_name → set of group compound-names.
    Reads <innernamespace> directly from group XML (Doxygen 1.12+)."""
    import xml.etree.ElementTree as _ET
    ns_to_groups: dict[str, set] = {}
    for refid in all_refids:
        xml_path = xml_dir / f"{refid}.xml"
        if not xml_path.is_file():
            continue
        try:
            cd = _ET.parse(xml_path).getroot().find("compounddef")
        except _ET.ParseError:
            continue
        cname = (cd.findtext("compoundname") or "").strip()
        for inn in cd.findall("innernamespace"):
            ns_xml = xml_dir / f"{inn.get('refid', '')}.xml"
            if not ns_xml.is_file():
                continue
            try:
                ns_cd = _ET.parse(ns_xml).getroot().find("compounddef")
            except _ET.ParseError:
                continue
            if ns_cd is None:
                continue
            ns_name = (ns_cd.findtext("compoundname") or "").strip()
            if any(p in ns_name.split("::") for p in ("detail", "internal", "impl")):
                continue
            if not (ns_cd.findall("sectiondef") or
                    ns_cd.findall("innerclass") or
                    ns_cd.findall("innernamespace")):
                continue
            ns_to_groups.setdefault(ns_name, set()).add(cname)
    return ns_to_groups


__all__ = [
    "_itertext", "_type_to_md", "_MEMBERDEF_SECTIONS", "_read_class_brief",
    "_doxygen_desc_to_md", "_build_api_hierarchy", "_md_escape_cell",
    "_MEMBER_DIRECTIVE", "_MEMBER_DETAIL_SECTION", "_enum_synopsis_lines",
    "_function_signature", "_class_page_name", "_read_class_data",
    "_find_collaboration_svg", "_svg_make_transparent", "_svg_dark_variant",
    "_patch_namespace_xml_for_breathe", "_collect_all_nodes",
    "_build_ns_group_map",
]
