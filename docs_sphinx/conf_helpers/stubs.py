"""API-reference stub writers.

Emits one Markdown stub per Doxygen group / namespace / class, mirroring the
legacy Doxygen page layout (summary tables + per-member breathe directives).
``_generate_api_stubs`` is the entry point the build orchestrator calls. Builds
on the primitives in ``xml_render`` and the shared state in ``state``.
"""
from __future__ import annotations
import pathlib, re, os as _os, shutil as _shutil, textwrap as _textwrap
from .state import *
from .xml_render import *


def _write_api_stub(node: dict, out_dir: pathlib.Path,
                    classes_seen: dict, ns_map: dict | None = None) -> None:
    """Write one .md per group node. Recurses into children.

    Parent groups (have <innergroup> children) → navigation index pages with
    @subpage toctrees. Leaf groups → Doxygen-style summary tables (Classes /
    Typedefs / Enumerations / Functions / Variables / Macros) at top, then
    a per-member detail block per kind (one breathe directive per member, not
    the recursive `{doxygengroup}` which inlines every nested class). Inner
    classes get their own pages — emitted later by `_generate_api_stubs` from
    `classes_seen`, which this fn populates."""
    name = node["name"]
    title = node["title"]
    out = out_dir / f"{name}.md"

    if node["children"]:
        # Navigation index page — list children as @subpage entries; the
        # existing _subpage_list_to_toctree rule converts them to a real
        # toctree at translate time.
        lines = [f"# {title} {{#api_{name}}}", ""]
        lines += ["## Topics", ""]
        for child in node["children"]:
            lines.append(f"- @subpage api_{child['name']}")
        lines.append("")
        if node["detailed"]:
            lines += ["## Detailed Description", "", node["detailed"], ""]
        if ns_map and ns_map.get(name):
            lines += ["## Namespaces", ""]
            for _ns_name, anchor in ns_map[name]:
                lines.append(f"- @subpage {anchor}")
            lines.append("")
        lines += ["## Topics", ""]
        for child in node["children"]:
            lines.append(f"- @subpage api_{child['name']}")
        # out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _stub_write(out, "\n".join(lines) + "\n")
        for child in node["children"]:
            _write_api_stub(child, out_dir, classes_seen, ns_map)
        return

    # ---- Leaf page ----------------------------------------------------------
    lines = [f"# {title} {{#api_{name}}}", ""]
    if node["detailed"]:
        lines += ["## Detailed Description", "", node["detailed"], ""]

    if ns_map and ns_map.get(name):
        lines += ["## Namespaces", ""]
        for _ns_name, anchor in ns_map[name]:
            lines.append(f"- @subpage {anchor}")
        lines.append("")

    # Classes summary table — link to the per-class page that
    # `_generate_api_stubs` emits (one .md per refid, deduped across groups).
    if node["innerclasses"]:
        lines += ["## Classes", "",
                  "| Name | Description |", "|---|---|"]
        for c in node["innerclasses"]:
            classes_seen.setdefault(c["refid"], c)
            page = _class_page_name(c["refid"])
            link = f"[`{c['kind']} {c['name']}`]({page}.md)"
            lines.append(f"| {link} | {_md_escape_cell(c['brief'])} |")
        lines.append("")

    # Build a fast lookup of class qualified names known so far — used to
    # detect when a group's "Functions"/"Variables" sectiondef is actually
    # listing a class member (Doxygen groups span class boundaries). Such
    # members get rendered on the class page, not as standalone breathe
    # directives.
    class_qualifieds = {c.get("qualified") for c in classes_seen.values()
                        if c.get("qualified")}

    def _is_class_member(m: dict) -> bool:
        q = m.get("qualified") or ""
        if "::" not in q:
            return False
        parent = q.rsplit("::", 1)[0]
        return parent in class_qualifieds

    def _is_template_spec(m: dict) -> bool:
        # Explicit template specializations carry `<…>` in the name (Doxygen
        # stores `cv::saturate_cast< unsigned >`). breathe's C++ parser
        # rejects this as a function-name argument, so we can't emit a
        # `doxygenfunction` directive for them — the summary table still
        # lists them; only the per-member detail block is skipped.
        return "<" in (m.get("name") or "")


    # Section summary tables in Doxygen's order. For class-member items the
    # in-page anchor breathe would have emitted doesn't exist (we skip the
    # per-member directive below) — point the link at the parent class page
    # instead so the table stays clickable.
    def _member_anchor_link(m: dict, label: str) -> str:
        if _is_class_member(m):
            q = m["qualified"]
            parent_qualified = q.rsplit("::", 1)[0]
            for c in classes_seen.values():
                if c.get("qualified") == parent_qualified:
                    return f"[`{label}`]({_class_page_name(c['refid'])}.md)"
        return f"[`{label}`](#{m['id']})"

    for _, section_title in _MEMBERDEF_SECTIONS:
        items = node["sections"].get(section_title, [])
        if not items:
            continue
        lines.append(f"## {section_title}")
        lines.append("")
        if section_title == "Functions":
            # Added '.api-function-table' class selector here for specialized layout handling
            lines += ["{.api-reference-table .api-function-table}",
                      "| Return | Name | Description |", "|---|---|---|"]
            for m in items:
                # Dynamically extract and build specifier prefixes (static, virtual, etc.)
                prefixes = []
                if m.get("static") in ("yes", True): prefixes.append("static")
                if m.get("inline") in ("yes", True): prefixes.append("inline")
                if m.get("virtual") in ("yes", "virtual", True): prefixes.append("virtual")
                prefix_str = " ".join(prefixes) + " " if prefixes else ""

                ret = _md_escape_cell(m["type"]) or "&nbsp;"
                # Combine your code qualifiers with the return type variable
                full_return_type = f"{prefix_str}{ret}"

                tpl = m.get("template", "")
                tpl_html = f"<div class='api-template'>{_md_escape_cell(tpl)}</div>" if tpl else ""

                label = f"cv::{m['name']}{_md_escape_cell(m['args'])}"
                sig_link = _member_anchor_link(m, label)

                lines.append(f"| {tpl_html}{full_return_type} | {sig_link} | {_md_escape_cell(m['brief'])} |")

        elif section_title == "Typedefs":
            lines += ["{.api-typedef-table}",
                      "| Type | Name | Description |", "|---|---|---|"]
            for m in items:
                t = _md_escape_cell(m["type"]) or "&nbsp;"
                name_link = _member_anchor_link(m, f"cv::{m['name']}")
                lines.append(f"| typedef {t} | {name_link} | {_md_escape_cell(m['brief'])} |")
        elif section_title == "Variables":
            lines += ["{.api-reference-table}",
                      "| Type | Name | Description |", "|---|---|---|"]
            for m in items:
                t = _md_escape_cell(m["type"]) or "&nbsp;"
                name_link = _member_anchor_link(m, m["name"])
                lines.append(f"| `{t}` | {name_link} | {_md_escape_cell(m['brief'])} |")
        elif section_title == "Enumerations":
            # Code-style synopsis (Doxygen layout) instead of name/desc table.
            # Both summary and detail-block representations would duplicate
            # the same content — we only emit the synopsis here, and skip
            # enums in the detail-block loop below.
            for m in items:
                if m["brief"]:
                    lines.append(_md_escape_cell(m["brief"]))
                    lines.append("")
                lines.append("```cpp")
                lines.extend(_enum_synopsis_lines(m))
                lines.append("```")
                lines.append("")
            continue   # already appended trailing blank
        else:  # Macros
            lines += ["{.api-reference-table}",
                      "| Name | Description |", "|---|---|"]
            for m in items:
                name_link = _member_anchor_link(m, m["name"])
                lines.append(f"| {name_link} | {_md_escape_cell(m['brief'])} |")
        lines.append("")

    # Per-member detail blocks (Enumeration Type Documentation,
    # Function Documentation, …). One breathe directive per item — except
    # for enums, which are rendered as a single code-style synopsis (one
    # `enum {…}` block listing all values) to match Doxygen's group-page
    # layout. breathe's `{doxygenenum}` instead emits a discrete signature
    # box per enumerator, which looks fragmented and doesn't match the
    # reference rendering. Class members and template specializations are
    # skipped — see `_is_*`. Macros are deduped by name: Doxygen can emit
    # multiple <memberdef>s for an arity-overloaded macro, but breathe
    # renders the same C declaration each time and docutils complains
    # about duplicate IDs.
    seen_define_names: set[str] = set()
    for kind_key, section_title in _MEMBERDEF_SECTIONS:
        items = node["sections"].get(section_title, [])
        if not items:
            continue
        directive = _MEMBER_DIRECTIVE.get(kind_key)
        if not directive:
            continue
        # `rendered` collects entries with one of two shapes:
        #   ("breathe", spec, directive_name)  — emitted as a breathe block
        #   ("synopsis", brief, code_lines)    — emitted as a fenced block
        rendered = []
        for m in items:
            # Class members render on the class page; skip on the group page.
            if kind_key in ("function", "variable") and _is_class_member(m):
                continue
            # Explicit template specializations: breathe can't parse the
            # `<T>` in the function name, so we leave them out of the
            # detail section (table above still lists them).
            if _is_template_spec(m):
                continue
            if m["kind"] == "enum":
                # Enums are rendered as synopses in the "Enumerations"
                # summary section above; no separate detail block needed.
                continue
            qualified = m["qualified"] or m["name"]
            if m["kind"] == "function":
                # Always pass the param-types disambiguator. breathe sees
                # multiple matches for common names (e.g. cv::cos lives in
                # both core_quaternion and core_utils_softfloat) — without
                # a signature it can't pick. The XML patcher above
                # guarantees the matching <memberdef> is reachable for
                # breathe's lookup.
                spec = qualified + _function_signature(m)
            elif m["kind"] == "define":
                # Preprocessor macros aren't namespaced. Dedupe by name —
                # arity-overloaded macros (e.g. CV_LOG_VERBOSE) appear as
                # multiple memberdefs but render to the same C declaration.
                if m["name"] in seen_define_names:
                    continue
                seen_define_names.add(m["name"])
                spec = m["name"]
            else:
                spec = qualified
            rendered.append(("breathe", spec, directive))
        if not rendered:
            continue
        lines.append(f"## {_MEMBER_DETAIL_SECTION[section_title]}")
        lines.append("")
        for entry in rendered:
            if entry[0] == "synopsis":
                _, brief, code_lines = entry
                if brief:
                    lines.append(brief)
                    lines.append("")
                lines.append("```cpp")
                lines.extend(code_lines)
                lines.append("```")
                lines.append("")
            else:
                _, spec, dname = entry
                lines += [
                    f"```{{{dname}}} {spec}",
                    ":project: opencv",
                    "```",
                    "",
                ]

    # Hidden toctree for per-class pages — needed so Sphinx knows these
    # files exist and so the left sidebar lists them under this group.
    if node["innerclasses"]:
        lines += ["```{toctree}", ":hidden:", ":maxdepth: 1", ""]
        for c in node["innerclasses"]:
            lines.append(_class_page_name(c["refid"]))
        lines += ["```", ""]

    # out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _stub_write(out, "\n".join(lines) + "\n")


# Class-XML sectiondef kind → (summary heading, detail heading).
# Order in this mapping is the order Doxygen uses on a class page.
_CLASS_SUMMARY_SECTIONS = [
    ("public-type",             "Public Types"),
    ("public-func",             "Public Member Functions"),
    ("public-static-func",      "Static Public Member Functions"),
    ("public-attrib",           "Public Attributes"),
    ("public-static-attrib",    "Static Public Attributes"),
    ("protected-type",          "Protected Types"),
    ("protected-func",          "Protected Member Functions"),
    ("protected-static-func",   "Static Protected Member Functions"),
    ("protected-attrib",        "Protected Attributes"),
    ("protected-static-attrib", "Static Protected Attributes"),
    ("friend",                  "Friends"),
]


def _write_class_stub(cls: dict, out_dir: pathlib.Path,
                      xml_dir: pathlib.Path) -> None:
    """One .md per inner class. Mirrors Doxygen's class-page layout:
      * Brief + detailed description for the class itself
      * Summary tables, one per sectiondef kind (Public Member Functions,
        Static Public Member Functions, Protected Attributes, etc.)
      * Detail blocks per member, grouped Doxygen-style into Constructor &
        Destructor Documentation / Member Function Documentation /
        Member Data Documentation / etc.

    Detail blocks are per-member breathe directives (`{doxygenfunction}` /
    `{doxygenvariable}` / `{doxygentypedef}`), not the recursive
    `{doxygenclass} :members:` — the latter is breathe's discrete
    one-signature-per-method layout that the user wanted replaced.

    Falls back to a bare `{doxygenclass}` / `{doxygenstruct}` if the class
    XML can't be read (e.g. XML wasn't regenerated)."""
    page = _class_page_name(cls["refid"])
    out = out_dir / f"{page}.md"
    qualified = cls["qualified"] or cls["name"]
    kind_label = cls["kind"].title()  # "Class" / "Struct"
    title = f"{kind_label} {qualified}"
    # Note: no `{#refid}` anchor in the heading — duplicates the
    # docname-derived target. `_generate_api_stubs` seeds the
    # refid→docname mapping into `_ANCHOR_TO_DOC` for `@ref` resolution.
    lines = [f"# {title}", ""]

    # Collaboration diagram — surface the SVG the legacy Doxygen HTML build
    # already rendered (the XML pipeline disables graphs; see
    # `_find_collaboration_svg`). Copy it next to the stub so Sphinx's image
    # collector publishes it to `_images/`, then reference it relative to the
    # api/ doc. The `images/`/`js_assets/` rewrite in `_translate` doesn't
    # touch this path (no such dir segment), and `_img_xtree` leaves
    # non-contrib image refs unchanged. Absent SVG → section silently omitted.
    _svg = _find_collaboration_svg(cls["refid"], xml_dir.parent / "html")
    _light_name = _dark_name = None
    if _svg is not None:
        import hashlib as _hashlib
        try:
            _raw = _svg.read_text(encoding="utf-8")
            # Two theme variants: light = native Doxygen with a transparent
            # backdrop (white page shows through); dark = recoloured to
            # light-on-dark so it matches docs.opencv.org and blends with the
            # dark page. custom.css shows exactly one per active theme.
            #
            # Filenames are content-hashed: Doxygen names every diagram
            # `<refid>__coll__graph.svg`; if a browser cached an older copy
            # under that fixed name it would keep serving the stale image
            # (query-string busts don't always work — some caches key on path
            # only). A hashed filename is a brand-new URL whenever the SVG
            # content changes, so it can never be served stale.
            _light_txt = _svg_make_transparent(_raw)
            _dark_txt = _svg_dark_variant(_raw)
            _lh = _hashlib.md5(_light_txt.encode("utf-8")).hexdigest()[:10]
            _dh = _hashlib.md5(_dark_txt.encode("utf-8")).hexdigest()[:10]
            _light_name = f"{_svg.stem}.{_lh}.svg"
            _dark_name = f"{_svg.stem}.{_dh}.dark.svg"
            (out_dir / _light_name).write_text(_light_txt, encoding="utf-8")
            (out_dir / _dark_name).write_text(_dark_txt, encoding="utf-8")
        except OSError:
            _light_name = _dark_name = None
    if _light_name is not None:
        # `only-light` / `only-dark` are pydata-sphinx-theme's native
        # theme-aware image classes: the theme shows exactly one per active
        # colour mode (via `display:none !important`), and — critically —
        # exempts `.only-dark` images from its
        # `html[data-theme=dark] .bd-content img { background:#fff }` rule, so
        # our dark (transparent-backdrop) variant blends with the dark page
        # instead of getting a white card behind it.
        lines += [
            f"Collaboration diagram for {qualified}:",
            "",
            f"![Collaboration diagram for {qualified}]({_light_name})"
            "{.opencv-coll-graph .only-light}",
            "",
            f"![Collaboration diagram for {qualified}]({_dark_name})"
            "{.opencv-coll-graph .only-dark}",
            "",
        ]

    data = _read_class_data(cls["refid"], xml_dir)
    if data is None:
        # Fallback for missing XML.
        directive = "doxygenstruct" if cls["kind"] == "struct" else "doxygenclass"
        lines += [
            f"```{{{directive}}} {qualified}",
            ":project: opencv",
            ":members:",
            ":protected-members:",
            ":undoc-members:",
            "```",
            "",
        ]
        # out.write_text("\n".join(lines), encoding="utf-8")
        _stub_write(out, "\n".join(lines))
        return

    # 1) Summary tables in Doxygen's order.
    for sd_kind, summary_title in _CLASS_SUMMARY_SECTIONS:
        items = data["sections"].get(sd_kind, [])
        if not items:
            continue
        lines.append(f"## {summary_title}")
        lines.append("")
        # Type-bearing sections (functions, variables, typedefs) get a
        # Return/Type column. Enum-bearing public-type sections get
        # rendered as code-block synopses instead (matches the group-page
        # treatment).
        non_enum_items = [m for m in items if m["kind"] != "enum"]
        enum_items = [m for m in items if m["kind"] == "enum"]
        if non_enum_items:
            lines += ["{.api-reference-table}",
                      "| Return | Name | Description |", "|---|---|---|"]
            for m in non_enum_items:
                ret = _md_escape_cell(m["type"]) or "&nbsp;"
                if m["static"]:
                    ret = "static " + ret
                sig = f"{m['name']}{_md_escape_cell(m['args'])}"
                sig_link = f"[`{sig}`](#{m['id']})"
                lines.append(
                    f"| `{ret}` | {sig_link} | {_md_escape_cell(m['brief'])} |")
            lines.append("")
        for m in enum_items:
            if m["brief"]:
                lines.append(_md_escape_cell(m["brief"]))
                lines.append("")
            lines.append("```cpp")
            lines.extend(_enum_synopsis_lines(m))
            lines.append("```")
            lines.append("")

    # 2) "Detailed Description" section for the class itself.
    if data["detailed"]:
        lines += ["## Detailed Description", "", data["detailed"], ""]

    # 3) Per-member detail blocks. Functions split into ctor/dtor vs
    #    others (mirrors Doxygen's "Constructor & Destructor Documentation"
    #    + "Member Function Documentation"). Variables → "Member Data
    #    Documentation". Typedefs → "Member Typedef Documentation".
    #    Enums → "Member Enumeration Documentation" (synopsis code block).
    class_simple = qualified.rsplit("::", 1)[-1]

    # Doxygen leaves <qualifiedname> empty for the members of some classes
    # (cv::_InputArray is one), so `m["qualified"]` falls back to the bare
    # member name. A bare name makes breathe search the *whole* project: it
    # resolves only when the name+signature is unique across all documented
    # symbols (e.g. `channels(int) const`), but common methods shared with
    # other classes stay ambiguous — `copyTo`, `empty`, `getFlags`, `size`
    # all collide with Mat/UMat/etc. and render as "Unable to resolve".
    # Scope every member to this class so the lookup is unambiguous.
    def _scoped(m: dict) -> str:
        q = m.get("qualified") or m["name"]
        return q if "::" in q else f"{qualified}::{m['name']}"

    typedef_items: list[dict] = []
    enum_items_all: list[dict] = []
    ctor_dtor_items: list[dict] = []
    func_items: list[dict] = []
    var_items: list[dict] = []
    _detail_seen: set[str] = set()
    for sd_items in data["sections"].values():
        for m in sd_items:
            if m["id"] in _detail_seen:
                continue
            _detail_seen.add(m["id"])
            if m["kind"] == "typedef":
                typedef_items.append(m)
            elif m["kind"] == "enum":
                enum_items_all.append(m)
            elif m["kind"] == "function":
                if m["name"] == class_simple or m["name"] == f"~{class_simple}":
                    ctor_dtor_items.append(m)
                else:
                    func_items.append(m)
            elif m["kind"] == "variable":
                var_items.append(m)

    def _emit_member_directive(m: dict, directive: str, spec: str) -> list[str]:
        # docutils splits a directive's arguments on whitespace and caps
        # doxygentypedef/doxygenvariable/doxygendefine at their declared arg
        # count, so a spec carrying template-argument spaces (Doxygen emits
        # e.g. `cv::ParamType< String >::member_type`) parses as 3 arguments
        # and is rejected with "maximum N argument(s) allowed". Collapse the
        # whitespace for these name-only directives — the C++ domain matches
        # names whitespace-insensitively. doxygenfunction is exempt: it takes a
        # full signature (final_argument_whitespace) whose `(...)` parameter
        # types need their spaces.
        if directive != "doxygenfunction":
            spec = re.sub(r"\s+", "", spec)
        # MyST anchor label so the summary-table `#refid` link resolves.
        spec = spec.replace("< ", "<").replace(" >", ">")
        return [
            f"({m['id']})=",
            f"```{{{directive}}} {spec}",
            ":project: opencv",
            "```",
            "",
        ]

    if typedef_items:
        lines += ["## Member Typedef Documentation", ""]
        for m in typedef_items:
            lines += _emit_member_directive(m, "doxygentypedef", _scoped(m))

    if enum_items_all:
        # Enums render as code-block synopses (matches the group-page
        # treatment — breathe's `{doxygenenum}` gives the discrete
        # one-signature-per-value layout the user wanted replaced).
        lines += ["## Member Enumeration Documentation", ""]
        for m in enum_items_all:
            lines.append(f"({m['id']})=")
            if m["brief"]:
                lines.append(_md_escape_cell(m["brief"]))
                lines.append("")
            lines.append("```cpp")
            lines.extend(_enum_synopsis_lines(m))
            lines.append("```")
            lines.append("")

    # Dedupe functions: a method can appear in multiple sectiondefs (e.g.
    # the same memberdef appearing in `public-func` and again via a
    # `<member refid>` we inlined). The refid is unique per memberdef.
    def _dedupe(items: list[dict]) -> list[dict]:
        seen, out = set(), []
        for m in items:
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            out.append(m)
        return out

    if ctor_dtor_items:
        lines += ["## Constructor & Destructor Documentation", ""]
        for m in _dedupe(ctor_dtor_items):
            spec = _scoped(m) + _function_signature(m)
            lines += _emit_member_directive(m, "doxygenfunction", spec)

    if func_items:
        lines += ["## Member Function Documentation", ""]
        for m in _dedupe(func_items):
            spec = _scoped(m) + _function_signature(m)
            lines += _emit_member_directive(m, "doxygenfunction", spec)

    if var_items:
        lines += ["## Member Data Documentation", ""]
        for m in _dedupe(var_items):
            lines += _emit_member_directive(m, "doxygenvariable", _scoped(m))

    # out.write_text("\n".join(lines), encoding="utf-8")
    _stub_write(out, "\n".join(lines))


def _write_namespace_stub(ns: dict, out_dir: pathlib.Path,
                          xml_dir: pathlib.Path,
                          ns_group_map: dict | None = None,
                          group_info: dict | None = None) -> tuple[str, str]:
    """Write api/namespace_<slug>.md for one namespace. Returns (anchor, fname)."""
    import xml.etree.ElementTree as _ET
    slug = ns["name"].replace("::", "__")
    anchor = f"api_ns_{slug}"
    fname = f"namespace_{slug}.md"
    lines = [f"# {ns['name']} namespace {{#{anchor}}}", ""]
    if ns_group_map and group_info:
        crumbs: list[str] = []
        for grp in sorted(ns_group_map.get(ns["name"], set())):
            chain: list[str] = []
            cur: str | None = grp
            while cur and cur in group_info:
                chain.append(cur)
                cur = group_info[cur]["parent"]
            chain.reverse()
            parts = [f"[{group_info[g]['title']}]({g}.md)" for g in chain]
            if parts:
                crumbs.append(" » ".join(parts))
        if crumbs:
            lines += [" | ".join(crumbs), ""]
    if ns["brief"]:
        lines += [ns["brief"], ""]

    # Read member sections from patched XML (has inlined group memberdefs).
    ns_sections: dict[str, list[dict]] = {}
    ns_xml_path = _PATCHED_XML_DIR / f"{ns['refid']}.xml" if ns.get("refid") else None
    if ns_xml_path and ns_xml_path.is_file():
        try:
            cd_ns = _ET.parse(ns_xml_path).getroot().find("compounddef")
            if cd_ns is not None:
                for sd in cd_ns.findall("sectiondef"):
                    for md in sd.findall("memberdef"):
                        kind = md.get("kind", "")
                        section_title = dict(_MEMBERDEF_SECTIONS).get(kind)
                        if not section_title:
                            continue
                        qualified = (md.findtext("qualifiedname") or "").strip() or \
                                    (md.findtext("name") or "").strip()
                        # Skip class methods / sub-namespace members inlined via group patching.
                        _ns_pfx = ns["name"] + "::"
                        if qualified.startswith(_ns_pfx) and "::" in qualified[len(_ns_pfx):]:
                            continue
                        def _pt(p) -> str:
                            t = _itertext(p.find("type"))
                            arr = (p.findtext("array") or "").strip()
                            return (t + arr) if arr else t
                        enum_values = []
                        if kind == "enum":
                            for ev in md.findall("enumvalue"):
                                enum_values.append({
                                    "name":        (ev.findtext("name") or "").strip(),
                                    "initializer": (ev.findtext("initializer") or "").strip(),
                                    "brief":       _itertext(ev.find("briefdescription")),
                                })
                        ns_sections.setdefault(section_title, []).append({
                            "id":          md.get("id", ""),
                            "kind":        kind,
                            "name":        (md.findtext("name") or "").strip(),
                            "qualified":   qualified,
                            "type":        _itertext(md.find("type")),
                            "type_elem":   md.find("type"),
                            "static":      md.get("static") == "yes",
                            "args":        (md.findtext("argsstring") or "").strip(),
                            "param_types": [_pt(p) for p in md.findall("param")],
                            "brief":       _itertext(md.find("briefdescription")),
                            "enum_values": enum_values,
                            "strong":      md.get("strong", "no") == "yes",
                        })
        except _ET.ParseError:
            pass

    # Sub-namespaces: read <innernamespace> directly from the namespace XML.
    ns_prefix = ns["name"] + "::"
    innernamespaces = []
    if ns_xml_path and ns_xml_path.is_file():
        try:
            cd_ns2 = _ET.parse(ns_xml_path).getroot().find("compounddef")
            if cd_ns2 is not None:
                for inn in cd_ns2.findall("innernamespace"):
                    iname = (inn.text or "").strip()
                    irefid = inn.get("refid", "")
                    if iname:
                        innernamespaces.append((iname, irefid))
        except _ET.ParseError:
            pass
    def _ns_has_content(refid: str) -> bool:
        f = xml_dir / f"{refid}.xml"
        if not f.is_file():
            return True  # can't check → keep
        try:
            cd3 = _ET.parse(f).getroot().find("compounddef")
            return cd3 is not None and bool(
                cd3.findall("sectiondef") or
                cd3.findall("innerclass") or
                cd3.findall("innernamespace"))
        except _ET.ParseError:
            return True
    nonempty_ns = [(n, r) for n, r in innernamespaces if _ns_has_content(r)]
    if nonempty_ns:
        lines += ["## Namespaces", "", "| Namespace |", "|---|"]
        for iname, irefid in sorted(nonempty_ns, key=lambda x: x[0].lower()):
            short = iname[len(ns_prefix):] if iname.startswith(ns_prefix) else iname
            islug = iname.replace("::", "__")
            lines.append(f"| [namespace {short}](namespace_{islug}.md) |")
        lines.append("")

    # Namespace XML lacks <innerclass> in Doxygen 1.12; glob by refid prefix.
    refid_prefix = ns["name"].replace("::", "_1_1") + "_1_1"
    innerclasses = []
    for kind in ("struct", "class"):
        for xml_file in sorted(xml_dir.glob(f"{kind}{refid_prefix}*.xml")):
            try:
                cd2 = _ET.parse(xml_file).getroot().find("compounddef")
                if cd2 is None:
                    continue
                cname = (cd2.findtext("compoundname") or "").strip()
                if "::" in cname[len(ns_prefix):]:  # skip sub-namespace classes
                    continue
                brief = _itertext(cd2.find("briefdescription"))
                innerclasses.append((xml_file.stem, cname, kind, brief))
            except _ET.ParseError:
                continue
    if innerclasses:
        lines += ["## Classes", "",
                  "{.api-reference-table}",
                  "| Name |", "|---|"]
        for ic_refid, ic_name, ic_kind, ic_brief in innerclasses:
            page = _class_page_name(ic_refid)
            short_name = ic_name[len(ns_prefix):]  # strip namespace prefix for display
            # Keyword `class`/`struct` as plain text; only the name is linked.
            lines.append(f"| {ic_kind} [`{short_name}`]({page}.md) |")
        lines.append("")

    # Member summary tables. Skip group-scoped members entirely — they're
    # documented on their owning group's page (and that's also where their
    # `#refid` anchor lives). Listing them here without a working anchor link
    # is worse than omitting them, and emitting directives for them below is
    # what was inflating Sphinx's cpp-domain `Symbol` tree to ~14M nodes
    # (1+ GB environment.pickle). Group-scoped refids start with `group__`;
    # namespace-original memberdefs use the `namespace*_1…` pattern.
    ns_sections = {
        section: [m for m in items if not (m.get("id") or "").startswith("group__")]
        for section, items in ns_sections.items()
    }
    for kind_key, section_title in _MEMBERDEF_SECTIONS:
        items = ns_sections.get(section_title, [])
        if not items:
            continue
        lines.append(f"## {section_title}")
        lines.append("")
        if section_title == "Functions":
            # Added '.api-function-table' class selector here as well
            lines += ["{.api-reference-table .api-function-table}",
                      "| Return | Name | Description |", "|---|---|---|"]
            for m in items:
                prefixes = []
                if m.get("static") in ("yes", True): prefixes.append("static")
                if m.get("inline") in ("yes", True): prefixes.append("inline")
                if m.get("virtual") in ("yes", "virtual", True): prefixes.append("virtual")
                prefix_str = " ".join(prefixes) + " " if prefixes else ""

                ret = _md_escape_cell(m["type"]) or "&nbsp;"
                full_return_type = f"{prefix_str}{ret}"

                tpl = m.get("template", "")
                tpl_html = f"<div class='api-template'>{_md_escape_cell(tpl)}</div>" if tpl else ""

                label = f"cv::{m['name']}{_md_escape_cell(m['args'])}"
                # FIXED: Removed the internal markdown backticks here to drop the gray pills
                lines.append(f"| {tpl_html}{full_return_type} | [{label}](#{m['id']}) | {_md_escape_cell(m['brief'])} |")

        elif section_title == "Typedefs":
            lines += ["{.api-typedef-table}",
                      "| Type | Name | Description |", "|---|---|---|"]
            for m in items:
                t = _md_escape_cell(m["type"]) or "&nbsp;"
                lines.append(f"| typedef {t} | [`cv::{m['name']}`](#{m['id']}) | {_md_escape_cell(m['brief'])} |")
        elif section_title == "Variables":
            lines += ["{.api-reference-table}",
                      "| Type | Name | Description |", "|---|---|---|"]
            for m in items:
                t = _md_escape_cell(m["type"]) or "&nbsp;"
                lines.append(f"| `{t}` | [`{m['name']}`](#{m['id']}) | {_md_escape_cell(m['brief'])} |")
        elif section_title == "Enumerations":
            for m in items:
                if m["brief"]:
                    lines.append(_md_escape_cell(m["brief"]))
                    lines.append("")
                lines.append("```cpp")
                lines.extend(_enum_synopsis_lines(m))
                lines.append("```")
                lines.append("")
            continue
        else:  # Macros
            lines += ["{.api-reference-table}",
                      "| Name | Description |", "|---|---|"]
            for m in items:
                lines.append(f"| [`{m['name']}`](#{m['id']}) | {_md_escape_cell(m['brief'])} |")
        lines.append("")

    if not ns_sections and not innerclasses:
        lines += [
            "## Detailed Description",
            "",
            f"```{{doxygennamespace}} {ns['name']}",
            ":project: opencv",
            "```",
        ]
    elif ns.get("detailed"):
        lines += ["## Detailed Description", "", ns["detailed"], ""]

    # Per-member detail blocks.
    seen_define_names: set[str] = set()
    for kind_key, section_title in _MEMBERDEF_SECTIONS:
        items = ns_sections.get(section_title, [])
        if not items:
            continue

        # Enums: heading + declaration + Enumerator table with Python names.
        if section_title == "Enumerations":
            enum_items = [m for m in items if "<" not in (m.get("name") or "")]
            if enum_items:
                lines.append(f"## {_MEMBER_DETAIL_SECTION[section_title]}")
                lines.append("")
                for m in enum_items:
                    qualified = m["qualified"] or m["name"]
                    keyword = "enum class" if m.get("strong") else "enum"
                    lines.append(f"({m['id']})=")
                    lines.append(f"### {m['name']}")
                    lines.append("")
                    lines += [f"`{keyword} {qualified}`", ""]
                    if m.get("brief"):
                        lines += [_md_escape_cell(m["brief"]), ""]
                    vals = m.get("enum_values") or []
                    if vals:
                        has_desc = any(v.get("brief") for v in vals)
                        if has_desc:
                            lines += ["| Enumerator | Description |", "|---|---|"]
                        else:
                            lines += ["| Enumerator |", "|---|"]
                        for v in vals:
                            scope = qualified if m.get("strong") else ns["name"]
                            cpp_key = f"{scope}::{v['name']}"
                            py_entries = _PY_SIGNATURES.get(cpp_key, [])
                            py_name = py_entries[0]["name"] if py_entries else None
                            cell = f"`{v['name']}`"
                            if py_name:
                                cell += f"<br>Python: `{py_name}`"
                            if has_desc:
                                desc = _md_escape_cell(v.get("brief") or "")
                                lines.append(f"| {cell} | {desc} |")
                            else:
                                lines.append(f"| {cell} |")
                        lines.append("")
            continue

        directive = _MEMBER_DIRECTIVE.get(kind_key)
        if not directive:
            continue
        rendered = []
        for m in items:
            if "<" in (m.get("name") or ""):  # skip template specializations
                continue
            if m["kind"] == "enum":
                continue
            qualified = m["qualified"] or m["name"]
            if m["kind"] == "function":
                spec = qualified + _function_signature(m)
            elif m["kind"] == "define":
                if m["name"] in seen_define_names:
                    continue
                seen_define_names.add(m["name"])
                spec = m["name"]
            else:
                spec = qualified
            rendered.append((spec, directive))
        if not rendered:
            continue
        lines.append(f"## {_MEMBER_DETAIL_SECTION[section_title]}")
        lines.append("")
        for spec, dname in rendered:
            short = spec.split("::")[-1].split("(")[0]
            suffix = "()" if dname == "doxygenfunction" else ""
            lines += [
                f"### {short}{suffix}",
                "",
                f"```{{{dname}}} {spec}",
                ":project: opencv",
                "```",
                "",
            ]

    # (out_dir / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _stub_write(out_dir / fname, "\n".join(lines) + "\n")
    return anchor, fname


def _namespaces_for_group(group_name: str, xml_dir: pathlib.Path,
                          ns_group_map: dict[str, set]) -> list[dict]:
    """Return namespace dicts for a group from ns_group_map."""
    import xml.etree.ElementTree as _ET, glob as _glob
    wanted = {ns for ns, grps in ns_group_map.items() if group_name in grps}
    candidates: list[dict] = []
    for ns_name in sorted(wanted):
        ns_file = xml_dir / ("namespace" + "_1_1".join(ns_name.split("::")) + ".xml")
        if not ns_file.is_file():
            for f in _glob.glob(str(xml_dir / "namespacecv*.xml")):
                try:
                    cd = _ET.parse(f).getroot().find("compounddef")
                    if cd is not None and (cd.findtext("compoundname") or "").strip() == ns_name:
                        ns_file = pathlib.Path(f)
                        break
                except _ET.ParseError:
                    continue
        if not ns_file.is_file():
            continue
        try:
            cd = _ET.parse(ns_file).getroot().find("compounddef")
        except _ET.ParseError:
            continue
        if cd is None:
            continue
        brief = _itertext(cd.find("briefdescription"))
        detailed_el = cd.find("detaileddescription")
        detailed = "\n\n".join(
            p for p in (_itertext(e) for e in (detailed_el.findall("para") if detailed_el is not None else []))
            if p)
        candidates.append({"name": ns_name, "refid": cd.get("id", ""), "brief": brief, "detailed": detailed})
    return candidates


_stub_written: set[pathlib.Path] = set()  # populated by _stub_write during generation


def _stub_write(path: pathlib.Path, content: str) -> None:
    """Write only when content changed; track path for stale-file cleanup."""
    if not (path.is_file() and path.read_text(encoding="utf-8") == content):
        path.write_text(content, encoding="utf-8")
    _stub_written.add(path)

def _generate_api_stubs(modules, xml_dir, out_dir):
    """Generate the full api/ stub tree. Write-if-changed so Sphinx incremental
    builds only reprocess pages whose content actually changed. Stale files
    from removed modules are deleted at the end."""
    if not modules:
        return
    if not xml_dir.is_dir():
        return  # No XML yet (sphinx-xml not run); degrade silently.
    src_index = xml_dir / "index.xml"
    root_md = out_dir / "api_root.markdown"
    if (src_index.is_file() and root_md.is_file()
            and root_md.stat().st_mtime >= src_index.stat().st_mtime):
        # Cache hit: stubs are current. Just reseed the refid → docname
        # map for every existing per-class page so cross-refs still work.
        has_ns_stubs = False
        for stub in out_dir.iterdir():
            n = stub.name
            if n.endswith(".md") and (n.startswith("class")
                                      or n.startswith("struct")):
                refid = n[:-3]
                _ANCHOR_TO_DOC[refid] = f"api/{refid}"
            elif n.endswith(".md") and n.startswith("namespace_"):
                has_ns_stubs = True
                if has_ns_stubs:
                    return

    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global _stub_written
    _stub_written = set()
    root_lines = [
        "API Reference {#api_root}",
        "=============",
        "",
        "Sphinx-rendered API reference for OpenCV main modules. Each entry",
        "below is a module's umbrella `@defgroup`; sub-pages mirror the",
        "Doxygen subgroup hierarchy.",
        "",
    ]
    classes_seen: dict[str, dict] = {}
    for m in modules:
        tree = _build_api_hierarchy(
            "group__" + m.replace("_", "__"), xml_dir)
        if tree is None:
            continue
        root_lines.append(f"- @subpage api_{tree['name']}")
        all_nodes = _collect_all_nodes(tree)
        all_refids = ["group__" + n.replace("_", "__") for n in all_nodes]
        ns_group_map = _build_ns_group_map(all_refids, _API_XML_DIR)
        # Flatten the group tree into {name: {title, parent}} for breadcrumbs.
        group_info: dict[str, dict] = {}
        def _flatten(node: dict, parent: str | None) -> None:
            group_info[node["name"]] = {"title": node["title"], "parent": parent}
            for child in node.get("children", []):
                _flatten(child, node["name"])
        _flatten(tree, None)
        ns_map: dict[str, list] = {}
        for group_name in all_nodes:
            for ns in _namespaces_for_group(group_name, _API_XML_DIR, ns_group_map):
                anchor, _ = _write_namespace_stub(ns, out_dir, _API_XML_DIR,
                                                  ns_group_map, group_info)
                ns_map.setdefault(group_name, []).append((ns["name"], anchor))
        _write_api_stub(tree, out_dir, classes_seen, ns_map)
    # Per-class pages (one per unique refid across all groups). We also
    # seed `_ANCHOR_TO_DOC` directly with refid -> docname so `@ref`
    # cross-references in tutorial markdown (and in any group page) keep
    # working — the per-class page no longer carries a `{#refid}` heading
    # anchor (would duplicate the docname-derived target).
    for cls in classes_seen.values():
        _write_class_stub(cls, out_dir, xml_dir)
        _ANCHOR_TO_DOC[cls["refid"]] = f"api/{_class_page_name(cls['refid'])}"
    # (out_dir / "api_root.markdown").write_text("\n".join(root_lines) + "\n", encoding="utf-8")
    _stub_write(out_dir / "api_root.markdown", "\n".join(root_lines) + "\n")
    for _p in list(out_dir.iterdir()):
        if _p not in _stub_written:
            _p.unlink(missing_ok=True)

