"""Doxygen-flavored .markdown -> MyST translation (the source-read engine).

``_translate`` rewrites a single document's text and ``_source_read`` is the
Sphinx hook conf.py registers. Also holds the snippet/toggle/contrib-image
helpers the translation passes rely on. Reads shared state (anchor maps, tag
URLs, image & snippet indexes, constants) from ``state``.

Note: ``_translate`` is one large sequential rewrite pass (~1200 lines). Its
steps are ordered and interdependent, so it is intentionally left as a single
function rather than split across files.
"""
from __future__ import annotations
import pathlib, re, os as _os, shutil as _shutil, textwrap as _textwrap
from .state import *


def _materialize_contrib_image(url: str) -> None:
    """Copy a contrib image into srcdir on first reference so Sphinx
    can find it. Idempotent; lazy."""
    if not url.startswith("_contrib_images/"):
        return
    rel = url[len("_contrib_images/"):]
    src = CONTRIB_ROOT / rel
    dest = SPHINX_INPUT_ROOT / "_contrib_images" / rel
    if src.is_file() and not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _shutil.copy2(src, dest)


def _stage_unique_contrib_image(src_abs: pathlib.Path, module: str) -> str | None:
    """Copy a contrib image into srcdir under a module-prefixed basename
    and return its srcdir-relative URL.

    Why uniqueness matters: Sphinx's `_images/` collection keys on the
    file's basename. Two contrib and main-tree files with the same
    basename (e.g. face's `2.jpg` vs calib3d's `2.jpg`) compete for the
    same `_images/2.jpg` destination. Sphinx's incremental
    `copy_asset_file` only overwrites when src.mtime != dest.mtime, so
    a `_images/2.jpg` left over from a prior build that referenced the
    other source survives — and the current page silently shows the
    wrong image. Staging under `<module>__<parent>__<basename>` makes
    the basename Sphinx sees unique across modules and across builds,
    so no slot is ever shared and no stale file can hijack the URL.

    Idempotent: only copies when the staged file is missing."""
    if not src_abs.is_file():
        return None
    # `<module>__<parent_dir>__<basename>` keeps the rendered file
    # name readable (e.g. `face__images__2.jpg`) while guaranteeing
    # uniqueness both across modules (different `module` prefix) and
    # within a module (different `parent` segment).  Use `__` as the
    # separator — extremely unlikely to appear in real filenames.
    parent = src_abs.parent.name
    unique_basename = f"{module}__{parent}__{src_abs.name}"
    staged_rel = f"_contrib_images/{module}/{unique_basename}"
    staged_abs = SPHINX_INPUT_ROOT / staged_rel
    if not staged_abs.exists():
        staged_abs.parent.mkdir(parents=True, exist_ok=True)
        _shutil.copy2(src_abs, staged_abs)
    return staged_rel


def _contrib_source_from_url(hit: str) -> tuple[str, pathlib.Path] | None:
    """Given a URL stored in `_IMAGE_INDEX` for a contrib image, derive
    `(module, source_abs_path)` so the caller can re-stage it uniquely.
    Returns None for main-tree URLs (no re-staging needed)."""
    if hit.startswith("tutorials_contrib/"):
        parts = pathlib.Path(hit).parts
        if len(parts) >= 3:
            # tutorials_contrib/<m>/<rel-relative-to-<m>/tutorials>
            module = parts[1]
            src = CONTRIB_ROOT / module / "tutorials" / pathlib.Path(*parts[2:])
            return module, src
    elif hit.startswith("_contrib_images/"):
        parts = pathlib.Path(hit).parts
        if len(parts) >= 2:
            # _contrib_images/<m>/<rest-relative-to-CONTRIB_ROOT/<m>>
            module = parts[1]
            src = CONTRIB_ROOT / pathlib.Path(*parts[1:])
            return module, src
    return None

def _normalize_lang(lang: str) -> str:
    lang = (lang or "").strip(".").strip().lower() or "text"
    return _LANG_ALIASES.get(lang, lang)


def _read_snippet(rel_path: str, label: str | None) -> tuple[str, str]:
    """Return (code_text, language) for an @include / @snippet directive."""
    # Some sources write the path with a leading slash (e.g. `@include
    # /samples/android/.../tutorial1_surface_view.xml`). pathlib's `/` would
    # treat that as absolute and lose the snippet base, so strip it.
    rel_norm = rel_path.lstrip("/")
    p = next((b / rel_norm for b in _SNIPPET_BASES
              if (b / rel_norm).is_file()), None)
    # Doxygen does a recursive basename lookup across EXAMPLE_PATH (see
    # opencv/doc/Doxyfile.in: EXAMPLE_RECURSIVE = YES). If the direct join
    # doesn't find the file, fall back to the prebuilt basename index.
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
    # Doxygen matches `[label]` after any comment-style marker anywhere on a
    # line: //, //! and // for C/C++/Java/Kotlin, # and ## for Python/shell,
    # <!-- for XML/HTML. Block-comment-wrapped labels like
    # `/* //! [label] */` are matched via the `//`-prefix branch too.
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
    # 0. @verbatim ... @endverbatim — stash content first so neither math
    #    markers, @code, nor any other rule below mangles the body. Used
    #    heavily in introduction/documenting_opencv/documentation_tutorial,
    #    which shows Doxygen syntax (so the body contains literal directives,
    #    `\f[...\f]` math, and code-fences as examples). Body is restored at
    #    the very end of this function with a private-use placeholder so the
    #    inserted text is safe from re-processing.
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

    # 1b. Convert a trailing setext heading at EOF to ATX. Otherwise
    #     docutils rejects the doc as ending with a transition.
    text = re.sub(
        r"^(?P<title>[^\n#=\-][^\n]*?)[ \t]*\n(?P<bar>[=\-])[=\-]{2,}[ \t]*$\s*\Z",
        lambda m: f"{'#' if m.group('bar') == '=' else '##'} {m.group('title').strip()}\n",
        text, flags=re.MULTILINE)

    # 1c. Convert remaining mid-doc setext H1s to ATX so 1d can see them.
    text = re.sub(
        r"^(?P<title>[^\n#=\-][^\n]*?)[ \t]*\n=[=]{2,}[ \t]*$",
        lambda m: f"# {m.group('title').strip()}",
        text, flags=re.MULTILINE)

    # 1d. Demote every H1 after the first to H2 so multi-H1 Doxygen docs
    #     (one `# Heading` per section) end up with a proper "1 title +
    #     N sections" outline. Without this, Sphinx's toctree lists every
    #     H1 as a separate entry on the parent TOC page.
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

    # 1e. Multi-line setext H2 splitter.  When `----` immediately follows
    #     a multi-line text block without an intervening blank line,
    #     CommonMark greedily folds the entire block into the heading
    #     title.  Some contrib sources omit that blank (e.g. xfeatures2d
    #     py_brief.markdown: the body of the "STAR(CenSurE)" section
    #     flows straight into the next "BRIEF in OpenCV\n---" without a
    #     separator, producing one giant 8-line <h2>).
    #
    #     A naive `re.sub` for this has bad false positives: it would
    #     split inside fenced code blocks (where `----` is just bash
    #     output) and inside `\f[...\f]` math (turning the `\f]` close
    #     marker into a heading).  Use a line-based scanner that
    #     (a) tracks ``` / ~~~ fence state so fenced runs are skipped
    #     wholesale, and (b) requires every line of the candidate body
    #     to start with an alphabetic character — this excludes math
    #     markers (`\`), `@code`/`@endcode` directives (`@`), indented
    #     code blocks (` `), and `$$` math fences.
    def _split_multiline_setext_h2(src: str) -> str:
        lines = src.split("\n")
        n = len(lines)
        fence_open = re.compile(r"^[ \t]*(`{3,}|~{3,})")
        setext = re.compile(r"^-{3,}[ \t]*$")
        text_ok = re.compile(r"^[A-Za-z][^\n]*$")
        out: list[str] = []
        in_fence = False
        fence_char: str | None = None
        i = 0
        while i < n:
            line = lines[i]
            m = fence_open.match(line)
            if m:
                ch = m.group(1)[0]
                if not in_fence:
                    in_fence, fence_char = True, ch
                elif ch == fence_char:
                    in_fence, fence_char = False, None
                out.append(line); i += 1; continue
            if in_fence:
                out.append(line); i += 1; continue
            if text_ok.match(line):
                # Walk the contiguous run of alpha-only text lines.
                j = i
                while j < n and text_ok.match(lines[j]):
                    j += 1
                # Need at least 3 lines (>=2 body + 1 title) AND the
                # next line must be the setext underline.
                if (j - i) >= 3 and j < n and setext.match(lines[j]):
                    out.extend(lines[i:j - 1])
                    out.append("")
                    out.append(f"## {lines[j - 1].strip()}")
                    i = j + 1   # skip the underline
                    continue
                out.extend(lines[i:j])
                i = j
                continue
            out.append(line); i += 1
        return "\n".join(out)
    text = _split_multiline_setext_h2(text)

    # 2. Doxygen LaTeX math markers
    text = re.sub(r"\\f\[(.+?)\\f\]",
                  lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n",
                  text, flags=re.DOTALL)
    text = re.sub(r"\\f\$(.+?)\\f\$", lambda m: f"${m.group(1)}$",
                  text, flags=re.DOTALL)

    # 2b. \bordermatrix{...} is a Plain-TeX macro (not LaTeX), so MathJax
    #     leaves it raw. Convert to a standard `matrix` environment and
    #     translate `\cr` row separators to `\\`. Loses the bracket lines
    #     of bordermatrix but the contents render correctly.
    text = re.sub(
        r"\\bordermatrix\s*\{([^}]*)\}",
        lambda m: r"\begin{matrix}" + m.group(1).replace(r"\cr", r"\\")
                  + r"\end{matrix}",
        text)

    # 3. @code{.lang} ... @endcode → fenced block. Preserve the indent
    #    so blocks nested under a bullet item stay inside the list; for
    #    col-0 @code keep the legacy .strip() form (byte-identical).
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

    # 3a. \dot ... \enddot → MyST `{graphviz}` fenced directive. Body is
    #     raw DOT source; the fenced form keeps it out of MyST's smartquotes
    #     and URL-autolink passes.
    def _dot_repl(m: re.Match) -> str:
        indent = m.group("indent") or ""
        body = _textwrap.dedent(m.group("body")).strip("\n")
        if indent:
            body = "\n".join((indent + line) if line else line
                             for line in body.split("\n"))
            return f"\n{indent}```{{graphviz}}\n{body}\n{indent}```\n"
        return f"\n```{{graphviz}}\n{body}\n```\n"
    text = re.sub(
        r"^(?P<indent>[ \t]*)\\dot[ \t]*\n(?P<body>.*?)\n[ \t]*\\enddot[ \t]*$",
        _dot_repl, text, flags=re.DOTALL | re.MULTILINE)

    # 3b. Plain Markdown fences with a Doxygen-flavored language spec
    #     (e.g. "```.sh") confuse Pygments — strip the leading dot and apply
    #     the same alias map as @code{.lang}.
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
    # Names may be qualified C++ identifiers like `cv::saturate_cast`, so
    # the character class allows `:` in addition to word chars and `-`.
    text = re.sub(r'@ref\s+(?P<name>[\w:-]+)(?:\s+"(?P<disp>[^"]+)")?',
                  _ref_repl, text)

    # 7b. cv.Name → [cv.Name](doxygen url) for names in the API index.
    #     Skips code spans and fenced blocks so identifiers inside `…` or
    #     ```…``` aren't rewritten.
    if _CV_API:
        def _cvlink_repl(m: re.Match) -> str:
            url = _CV_API.get(m.group(1))
            return f'[cv.{m.group(1)}]({url})' if url else m.group(0)
        _parts = re.split(r'(```.*?```|`[^`\n]+`)', text, flags=re.DOTALL)
        text = ''.join(
            p if i % 2 else re.sub(
                r'(?<!\[)(?<!\()cv\.([A-Za-z][A-Za-z0-9_]*)',
                _cvlink_repl, p)
            for i, p in enumerate(_parts))

    # 8. @cite KEY → [[KEY]](link to docs.opencv.org citelist)
    text = re.sub(
        r"@cite\s+([\w-]+)",
        lambda m: f"[[{m.group(1)}]](https://docs.opencv.org/5.x/d0/de3/citelist.html#CITEREF_{m.group(1)})",
        text)

    # 8a / 8b. API stub link rewriting — narrowly scoped to the
    # "Basic structures" page (api/core_basic) so other api/ pages are
    # untouched.
    #
    # 8a. Name-column typedef anchors: the stub emits
    #         [Vec2b](#group__core__basic_1ga595…)
    #     where the anchor is a Doxygen group-page in-page reference
    #     that has no matching element on the Sphinx-built page.
    #     Rewrite to the Sphinx cpp-domain v4 anchor of the typedef on
    #     this same page — the right-sidebar "On this page" TOC links
    #     to exactly this anchor, so the table entry and the sidebar
    #     entry now jump to the same Typedef Documentation section in
    #     the local build. Only simple identifiers (typedef names) are
    #     rewritten; function-signature entries are left alone (their
    #     in-page anchors include mangled parameter types that we can't
    #     reconstruct from the markdown without parsing the Doxygen
    #     XML).
    #
    # 8b. Type-column class names: stub cells like `Matx< double, 1, 2 >`
    #     are a single code span with nothing clickable. Rewrite to
    #     inline HTML where the class name is a link to the LOCAL
    #     Sphinx api class page (e.g. classcv_1_1Matx.html — a sibling
    #     file in the same api/ directory). The bare filename is read
    #     out of the live tagfile (only as a class-name → filename
    #     map; no docs.opencv.org URL ends up in the output).
    if docname == "api/core_basic":
        # 8f. Move Vec specialization rows out of the Typedefs table into
        #     their own H2 section "Shorter aliases for the most popular
        #     specializations of Vec<T,n>", placed between "## Functions"
        #     and "## Typedef Documentation" to match the live Doxygen
        #     page's ordering. Doxygen reorders user-defined sectiondefs
        #     after the standard typedef/enum/function sections, but our
        #     stub generator lumps everything into the main Typedefs
        #     table; this pass restores the separation.
        #
        # Implementation: extract every consecutive `| `Vec<…` | … |`
        # row from the typedef table (they're all together at the top of
        # the table because of how the stub generator orders members),
        # then re-emit them as a new H2 + table just before the
        # `## Typedef Documentation` heading. Runs BEFORE the other
        # api/core_basic rewrites so the moved rows still flow through
        # the typedef-anchor / template-linkification / token-linkifier
        # passes below.
        _vec_rows_re = re.compile(
            r"(?:^\| `Vec<[^`]*` \| [^\n]*\n)+", re.MULTILINE)
        _vm = _vec_rows_re.search(text)
        if _vm:
            _vec_rows = _vm.group(0)
            text = text[:_vm.start()] + text[_vm.end():]
            _shorter = (
                "## Shorter aliases for the most popular specializations of "
                "Vec<T,n>\n\n"
                "| Type | Name | Description |\n"
                "|---|---|---|\n"
                + _vec_rows + "\n")
            text = text.replace(
                "## Typedef Documentation",
                _shorter + "## Typedef Documentation",
                1)

        # 8c. Replace `{doxygentypedef} cv::Ptr` with a hand-rolled
        #     cpp:type directive. Breathe's doxygentypedef cannot render
        #     C++11 template aliases (`using cv::Ptr = std::shared_ptr<_Tp>`
        #     in the Doxygen XML) — it silently emits nothing, no warning,
        #     so the Ptr entry was missing from the Typedef Documentation
        #     section even though every other typedef rendered. Sphinx's
        #     native cpp:type directive does support alias templates;
        #     reach it via an eval-rst escape. Anchor `_CPPv4N2cv3PtrE`
        #     matches what step 8a generates for the Name-column link.
        text = re.sub(
            r"```\{doxygentypedef\} cv::Ptr\s*\n:project: opencv\s*\n```",
            "```{eval-rst}\n"
            ".. cpp:namespace:: cv\n"
            ".. cpp:type:: template<typename _Tp> Ptr = std::shared_ptr<_Tp>\n"
            "```",
            text)

        # 8d. Namespaces section — original Doxygen renders this as a
        #     two-column borderless table ("namespace" label on the left,
        #     class link on the right), not a heading-plus-bullet-list.
        #     The stub emits `## Namespaces\n\n- @subpage api_ns_<x>`
        #     which step 9 then folds into a visible toctree. Replace
        #     with a two-column table + hidden toctree (so the page's
        #     sidebar nav still picks up the namespace child).
        def _build_namespaces_table(m: re.Match) -> str:
            rows = []
            toc = []
            for sub in re.finditer(r"- @subpage api_ns_(?P<a>[A-Za-z0-9_]+)",
                                   m.group("body")):
                anchor = sub.group("a")
                # Doxygen mangles `::` -> `__` in filenames; reverse it
                # to recover the display name (cv__traits -> cv::traits).
                display = anchor.replace("__", "::")
                href = f"namespace_{anchor}.html"
                docref = f"namespace_{anchor}"
                rows.append(f"| namespace | [{display}]({href}) |")
                toc.append(docref)
            table = "\n".join(
                ["## Namespaces", "", "| | |", "|---|---|", *rows, ""])
            if toc:
                table += "\n```{toctree}\n:hidden:\n:maxdepth: 1\n\n"
                table += "\n".join(toc) + "\n```\n"
            return table
        text = re.sub(
            r"## Namespaces\n\n(?P<body>(?:- @subpage api_ns_[A-Za-z0-9_]+\n)+)",
            _build_namespaces_table, text)

        # 8e. Classes table — two changes per row:
        #     (i)  append the template-parameter list (`< _Tp >` etc.)
        #          to the class name so e.g. `class cv::Mat_` becomes
        #          `class cv::Mat_< _Tp >`, matching Doxygen.
        #     (ii) append a "More..." link to the description cell,
        #          pointing to the class's own api stub page.
        def _rewrite_class_row(m: re.Match) -> str:
            kind = m.group("kind")
            name = m.group("name")       # 'cv::Mat_'
            page = m.group("page")       # 'classcv_1_1Mat__'
            desc = m.group("desc").strip()
            short = name.split("::")[-1]
            tparams = _CLASS_TEMPLATE_DISPLAY.get(short, "")
            label = f"{kind} {name}{tparams}"
            more = f"[More...]({page}.md)"
            desc_out = f"{desc} {more}" if desc else more
            return f"| [`{label}`]({page}.md) | {desc_out} |"
        text = re.sub(
            r"\| \[`(?P<kind>class|struct) (?P<name>cv::[A-Za-z0-9_:]+)`\]"
            r"\((?P<page>(?:class|struct)cv_1_1[A-Za-z0-9_]+)\.md\)"
            r" \| (?P<desc>[^\n|]*?) \|",
            _rewrite_class_row, text)

        text = re.sub(
            r"\[`(?P<name>[A-Za-z_][A-Za-z0-9_]*)`\]"
            r"\(#group__[a-z0-9_]+?_1[a-z0-9]+\)",
            lambda m: (f"[`{m.group('name')}`]"
                       f"(#_CPPv4N2cv{len(m.group('name'))}"
                       f"{m.group('name')}E)"),
            text)

        if _LIVE_CLASS_URL:
            def _linkify_class_codespan(m: re.Match) -> str:
                cls = m.group("cls")
                rest = m.group("rest")
                full = _LIVE_CLASS_URL.get(cls)
                if not full:
                    return m.group(0)
                # Use the bare filename — same directory as core_basic.html.
                href = pathlib.PurePosixPath(full).name
                rest_esc = (rest.replace("&", "&amp;")
                                .replace("<", "&lt;")
                                .replace(">", "&gt;"))
                return (f'<code class="docutils literal notranslate">'
                        f'<a class="reference internal" href="{href}">{cls}</a>'
                        f'{rest_esc}</code>')
            text = re.sub(
                r"`(?P<cls>[A-Z][A-Za-z0-9_]*)(?P<rest><[^`\n]*>)`",
                _linkify_class_codespan, text)

        # 8g. Linkify class/typedef tokens in code spans that step 8b did
        #     not transform. Covers two cases the live Doxygen page makes
        #     clickable but our Sphinx output didn't:
        #       (a) inner template-parameter types: `uchar` in
        #           `Vec< uchar, 2 >` should link to its typedef definition
        #           (a group anchor on docs.opencv.org). Step 8b made
        #           `Vec` itself clickable but the inner `uchar` stayed
        #           plain text inside `<code>`.
        #       (b) non-template Type cells: `_InputArray` in
        #           `const _InputArray &` was never matched by step 8b
        #           (no `<>` template form) so the class name stayed
        #           unlinked.
        #     Two passes: process inner HTML of step-8b `<code>` blocks
        #     (skipping their existing `<a>`), then process remaining
        #     plain markdown code spans containing recognized tokens.
        if _LIVE_CLASS_URL or _LIVE_TYPEDEF_URL:
            def _token_url(tok: str) -> str | None:
                # Match live tagfile maps; ignore C++ primitives by absence.
                return _LIVE_CLASS_URL.get(tok) or _LIVE_TYPEDEF_URL.get(tok)
            _tok_re = re.compile(r"\b_?[A-Za-z][A-Za-z0-9_]*\b")
            def _linkify_html_segment(seg: str) -> str:
                """Linkify recognized tokens in a plain-text HTML segment."""
                def _sub(m: re.Match) -> str:
                    url = _token_url(m.group(0))
                    if not url:
                        return m.group(0)
                    return (f'<a class="reference external" '
                            f'href="{url}">{m.group(0)}</a>')
                return _tok_re.sub(_sub, seg)
            def _linkify_inside_code(m: re.Match) -> str:
                """Walk the inner HTML of an existing <code> block, skipping
                spans already inside <a>...</a> (which step 8b emitted)."""
                inner = m.group("inner")
                out, i = [], 0
                n = len(inner)
                while i < n:
                    if inner.startswith("<a ", i):
                        j = inner.find("</a>", i)
                        if j < 0:
                            out.append(inner[i:]); break
                        out.append(inner[i:j + 4])
                        i = j + 4
                    else:
                        # Take a chunk up to the next <a — process it.
                        k = inner.find("<a ", i)
                        if k < 0:
                            out.append(_linkify_html_segment(inner[i:]))
                            break
                        out.append(_linkify_html_segment(inner[i:k]))
                        i = k
                return m.group("open") + "".join(out) + m.group("close")
            text = re.sub(
                r'(?P<open><code class="docutils literal notranslate">)'
                r'(?P<inner>.*?)(?P<close></code>)',
                _linkify_inside_code, text, flags=re.DOTALL)

            def _linkify_markdown_codespan(m: re.Match) -> str:
                """Convert a markdown `…` code span to <code>…</code> with
                embedded <a> tags when its content contains a recognized
                token; otherwise leave unchanged."""
                content = m.group("content")
                hits = [(t.start(), t.end(), t.group(0)) for t in
                        _tok_re.finditer(content) if _token_url(t.group(0))]
                if not hits:
                    return m.group(0)
                # Build mixed-HTML version preserving non-token text.
                from html import escape as _esc
                parts, last = [], 0
                for s, e, tok in hits:
                    parts.append(_esc(content[last:s]))
                    url = _token_url(tok)
                    parts.append(f'<a class="reference external" '
                                 f'href="{url}">{tok}</a>')
                    last = e
                parts.append(_esc(content[last:]))
                return (f'<code class="docutils literal notranslate">'
                        f'{"".join(parts)}</code>')
            # Negative lookbehind `(?<!\[)` and negative lookahead `(?!\])`
            # prevent matching code spans that are already markdown link
            # text — e.g. the Name-column `[`Vec2b`](#_CPPv4N2cv5Vec2bE)`
            # link. Wrapping that with an inner <a href="…"> would create
            # nested anchors (the inner external URL would win the click,
            # defeating the step 8a in-page-anchor rewrite).
            text = re.sub(
                r"(?<!\[)`(?P<content>[^`\n]+?)`(?!\])",
                _linkify_markdown_codespan, text)

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

    # 8c. @note / @see / @warning  -> MyST admonitions.  Each directive body runs
    #     until the next blank line, the next @directive at start-of-line, or
    #     end of file (matches Doxygen's paragraph-level semantics).
    _ADMON_KIND = {"note": "note", "see": "seealso", "warning": "warning"}
    def _admon_repl(m: re.Match) -> str:
        kind = _ADMON_KIND[m.group("dir")]
        body = m.group("body").strip()
        return f"\n:::{{{kind}}}\n{body}\n:::\n"
    text = re.sub(
        r"^@(?P<dir>note|see|warning)\s+(?P<body>.+?)(?=\n[ \t]*\n|\n@[A-Za-z]|\Z)",
        _admon_repl, text, flags=re.DOTALL | re.MULTILINE)

    # 8d. Dedent indented description paragraphs after `- @subpage X`
    #     so they render as normal text, not as code blocks. Accept both
    #     4-space and tab-based indentation (CommonMark continuation forms).
    def _dedent_subpage_descriptions(src: str) -> str:
        pat = re.compile(
            r"^(?P<bullet>[ \t]*-\s+[^\n]*@subpage\s+[\w-]+[^\n]*)\n"
            r"(?P<desc>(?:[ \t]*\n|(?:\t|[ \t]{4,})[^\n]+(?:\n|$))+)",
            re.MULTILINE)
        def repl(m: re.Match) -> str:
            desc = _textwrap.dedent(m.group("desc")).strip("\n")
            # All-blank description (e.g. `- @subpage X\n\n##### Section`):
            # don't rewrite, or we'd accumulate extra blank lines.
            if not desc.strip():
                return m.group(0)
            return f"{m.group('bullet')}\n\n{desc}\n\n"
        return pat.sub(repl, src)
    text = _dedent_subpage_descriptions(text)

    # 9. Bullet `@subpage` lists → real toctree. Enabled modules become
    #    internal entries; disabled ones become external Doxygen links.
    #    Allows any text between `-` and `@subpage` to accept the
    #    `- <module>. @subpage <id>` form used by contrib_root.markdown.
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

    # 11. @tableofcontents / [TOC] -> drop. PyData's right sidebar
    #     already shows the per-page outline.
    text = re.sub(r"^(?:@tableofcontents|\[TOC\])\s*$", "",
                  text, flags=re.MULTILINE)

    # 11b. @cond NAME ... @endcond  -> strip just the markers; if the
    #      enclosed @subpage points to a disabled module it gets dropped
    #      by _subpage_list_to_toctree above.  Same treatment for @parblock /
    #      @endparblock — they exist only to let Doxygen accept multi-
    #      paragraph arguments to directives like @note, which Markdown
    #      already handles natively, so the markers can be dropped.
    text = re.sub(r"^@cond\s+\S+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^@endcond\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*@parblock\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*@endparblock\s*$", "", text, flags=re.MULTILINE)

    # 11c. @anchor NAME  ->  MyST label "(NAME)=" so the following block
    #      element (typically a heading) becomes the cross-reference target.
    text = re.sub(r"^@anchor\s+(?P<name>[\w-]+)\s*$",
                  lambda m: f"({m.group('name')})=",
                  text, flags=re.MULTILINE)

    # 11d. Doxygen numbered-list marker `-#` at line start -> Markdown `1.`.
    #      Markdown auto-increments numbered lists, so emitting `1.` for every
    #      item produces the right output. Preserves leading indent for nested
    #      lists.
    text = re.sub(r"^(?P<indent>[ \t]*)-#(?P<sp>[ \t]+)",
                  lambda m: f"{m.group('indent')}1.{m.group('sp')}",
                  text, flags=re.MULTILINE)

    # 11e. Bullet markers with 5+ spaces between marker and content cause MyST
    #      to treat the continuation lines as code blocks. Normalize such
    #      bullets to 3-space separation and re-flow the continuation column
    #      by the same delta so nested content stays aligned with the marker.
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

    # 11f. Bullet lists immediately after a heading are sometimes indented by
    #      4 spaces in Doxygen sources — Markdown would interpret that as a
    #      code block. Strip exactly one level of 4-space indent off such
    #      runs so MyST renders a proper list.
    text = re.sub(
        r"(^#{1,6}[ \t][^\n]+\n(?:[ \t]*\n)*)((?:    [ \t]*[-*+][^\n]*\n)+)",
        lambda m: m.group(1) + re.sub(r"^    ", "", m.group(2), flags=re.MULTILINE),
        text, flags=re.MULTILINE)

    # 1a-iii. 4-space list items under a plain paragraph line → strip to fix lazy continuation.
    text = re.sub(
        r"(^(?![ \t#@`]|-#|[-*+]\s|\d+[.)]\s)[^\n]+\n)((?:    [-*+][ \t][^\n]*\n(?:[ \t]{5,}[^\n]*\n)*)+)",
        lambda m: m.group(1) + re.sub(r"^    ", "", m.group(2), flags=re.MULTILINE),
        text, flags=re.MULTILINE)

    # 1b0. \b word → **word** (Doxygen bold macro — single next word only).
    text = re.sub(r"\\b\s+(\S+)", r"**\1**", text)

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

    # 1c. @param / @return → MyST definition list.
    def _param_block_repl(m: re.Match) -> str:
        items: list[list] = []
        for line in m.group(0).split("\n"):
            pm = re.match(r"^@param\s+(\S+)\s*(.*)", line)
            rm = re.match(r"^@return\s*(.*)", line)
            if pm:
                items.append([f"`{pm.group(1)}`", [pm.group(2).strip()]])
            elif rm:
                items.append(["*(return value)*", [rm.group(1).strip()]])
            elif items and line.strip():
                items[-1][1].append(line.strip())
        def _inline_block_math(s: str) -> str:
            if re.match(r"^\\f\[.+\\f\]$", s.strip()):
                return s
            return re.sub(r"\\f\[(.+?)\\f\]", lambda mm: f"${mm.group(1).strip()}$", s)
        result = []
        has_param = False
        for key, desc_lines in items:
            desc_lines = [l for l in desc_lines if l]
            if not desc_lines:
                continue
            if key != "*(return value)*":
                has_param = True
            entry = f"{key}\n: {_inline_block_math(desc_lines[0])}"
            for cont in desc_lines[1:]:
                entry += f"\n  {_inline_block_math(cont)}"
            result.append(entry)
        header = "\n**Parameters**\n\n" if has_param else "\n"
        return header + "\n\n".join(result) + "\n"
    text = re.sub(
        r"((?:^@(?:param\s+\S+|return)\s+[^\n]+\n(?:[ \t]+[^\n]+\n)*)+)",
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
        r"^(?P<indent>[ \t]*)@code(?:\{(?P<lang>[^}]*)\})?\s*\n(?P<body>.*?)\n?[ \t]*@endcode",
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

    # 7. toctree before @ref so list-item @ref tutorial_* entries aren't converted first
    def _subpage_list_to_toctree(src: str) -> str:
        if not src.endswith("\n"):
            src += "\n"
        pat = re.compile(
            r"((?:^[ \t]*-\s+@(?:subpage\s+[\w-]+|ref\s+tutorial_[\w-]+)[^\n]*\n(?:(?:[ \t]*\n)?[ \t]+[^\n]+\n)*)+)",
            re.MULTILINE)
        def repl(m: re.Match) -> str:
            block = m.group(1)
            entries = re.findall(r"@(?:subpage|ref)\s+([\w-]+)", block)
            dm = re.search(
                r"@(?:subpage|ref)\s+[\w-]+[^\n]*\n((?:(?:[ \t]*\n)?[ \t]+[^\n]+\n)*)", block)
            desc_raw = dm.group(1) if dm and dm.group(1) else ""
            _groups: list[list[str]] = [[]]
            for _l in desc_raw.splitlines():
                if _l.strip():
                    _groups[-1].append(_l.strip())
                elif _groups[-1]:
                    _groups.append([])
            desc = "\n\n".join(" ".join(g) for g in _groups if g) or None
            lines = []
            for e in entries:
                if e in _ANCHOR_TO_DOC:
                    lines.append("/" + _ANCHOR_TO_DOC[e])
                elif e in _ANCHOR_TO_EXTERNAL:
                    title, url = _ANCHOR_TO_EXTERNAL[e]
                    lines.append(f"{title} <{url}>")
                elif e in _TAG_FILENAMES:
                    title = _TAG_PAGE_TITLES.get(e, e)
                    lines.append(f"{title} <{DOXYGEN_BASE_URL + _TAG_FILENAMES[e]}>")
            if not lines:
                return ""
            body = "\n".join(lines)
            result = f"\n```{{toctree}}\n:maxdepth: 1\n\n{body}\n```\n"
            if desc:
                result += f"\n{desc}\n"
            return result
        return pat.sub(repl, src)
    text = _subpage_list_to_toctree(text)

    # 7b. @ref name [optional "Display Text"]
    def _ref_repl(m: re.Match) -> str:
        name = m.group("name"); disp = m.group("disp")
        target = _ANCHOR_TO_DOC.get(name)
        if target:
            return f"[{disp or name}]({'/' + target})"
        return f"[{disp or name}](#{name})"
    text = re.sub(r'@ref\s+(?P<name>[\w:-]+)(?:\s+"(?P<disp>[^"]+)")?',
                  _ref_repl, text)

    # 7b. cv.Name → [cv.Name](doxygen url) for names in the API index.
    if _CV_API:
        def _cvlink_repl(m: re.Match) -> str:
            url = _CV_API.get(m.group(1))
            return f'[cv.{m.group(1)}]({url})' if url else m.group(0)
        _parts = re.split(r'(```.*?```|`[^`\n]+`)', text, flags=re.DOTALL)
        text = ''.join(
            p if i % 2 else re.sub(
                r'(?<!\[)(?<!\()cv\.([A-Za-z][A-Za-z0-9_]*)',
                _cvlink_repl, p)
            for i, p in enumerate(_parts))

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

    # 9. @next_tutorial / @prev_tutorial  -> drop
    text = re.sub(r"^@(?:next|prev)_tutorial\{[^}]*\}\s*$", "",
                  text, flags=re.MULTILINE)

    # 10a. col-8 -# under a col-0 parent: move to col 5 (avoids CommonMark code-block threshold).
    def _fix_nested_ordered(src: str) -> str:
        lines = src.split("\n")
        out = []
        for i, line in enumerate(lines):
            if re.match(r"^        -#\s", line):
                nearest = ""
                for j in range(i - 1, max(i - 40, -1), -1):
                    m = re.match(r"^([ \t]*)-", lines[j])
                    if m and len(m.group(1)) < 8:
                        nearest = lines[j]
                        break
                out.append(line if (nearest and re.match(r"^    ", nearest)) else "     " + line[8:])
            else:
                out.append(line)
        return "\n".join(out)
    text = _fix_nested_ordered(text)

    # 10b. Doxygen ordered-list marker: "-#" -> "1."
    text = re.sub(r"^([ \t]*)-#([ \t]+)", r"\g<1>1.\g<2>",
                  text, flags=re.MULTILINE)

    # 11. @tableofcontents -> drop (PyData right sidebar replaces it)
    text = re.sub(r"^@tableofcontents\s*$", "", text, flags=re.MULTILINE)

    # 11b. @cond NAME ... @endcond  -> strip just the markers; if the
    #      enclosed @subpage points to a disabled module it gets dropped
    #      by _subpage_list_to_toctree above.
    text = re.sub(r"^@cond\s+\S+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^@endcond\s*$", "", text, flags=re.MULTILINE)

    text = re.sub(r"</?center>", "", text, flags=re.IGNORECASE)

    # 11d. Escape C++ template <Type> in paragraph text; skip fenced code blocks.
    _cpp_tpl_re = re.compile(r'\b(\w+)<([A-Za-z_][\w:, *&]*?)>')
    _lines_out: list[str] = []
    _in_fence = False
    for _ln in text.splitlines(keepends=True):
        if re.match(r'^\s*```', _ln):
            _in_fence = not _in_fence
        if not _in_fence:
            _ln = _cpp_tpl_re.sub(lambda m: f'{m.group(1)}&lt;{m.group(2)}&gt;', _ln)
        _lines_out.append(_ln)
    text = "".join(_lines_out)

    # 11c. Wrap bare http(s) URLs in <> for CommonMark autolink.
    # Group 1 = full [text](url) link → pass through; else wrap bare URL.
    def _autolink_repl(m: re.Match) -> str:
        if m.group(1):
            return m.group(1)
        url = m.group(0)
        trail = ""
        while url and url[-1] in ".,;:!?)":
            trail = url[-1] + trail
            url = url[:-1]
        return f"<{url}>{trail}" if url else m.group(0)
    text = re.sub(
        r'(\[[^\]]*\]\([^)]*\))|(?<!\]\()(?<![<"])https?://\S+',
        _autolink_repl, text)

    # Depth-relative prefix for contrib_modules/ URLs (html_extra_path output).
    _depth = docname.count("/") if docname else 0
    _contrib_url_prefix = ("../" * _depth) + "contrib_modules/"

    def _emit_contrib_img(rel_url: str, alt: str) -> str:
        src = _contrib_url_prefix + rel_url
        return f'<img src="{src}" alt="{alt}"/>'

    # 12. Image paths "images/foo.png" — resolve like Doxygen's flat IMAGE_PATH:
    #     prefer the doc's own "images/" sibling, then fall back to a global
    #     basename lookup across every tutorial `images/` folder. As a final
    #     fallback, point at the consolidated `tutorials/others/images/` dir
    #     (where modules like `photo` store their assets).
    def _img_repl(m: re.Match) -> str:
        alt, rel = m.group("alt"), m.group("rel")
        if docname:
            local = DOC_ROOT / pathlib.Path(docname).parent / "images" / rel
            if local.is_file():
                return m.group(0)
        hit = _IMAGE_INDEX.get(pathlib.Path(rel).name)
        if hit:
            if hit.startswith("contrib_modules/"):
                return _emit_contrib_img(hit[len("contrib_modules/"):], alt)
            return f'![{alt}](/{hit})'
        if docname and docname.startswith("js_tutorials/"):
            return m.group(0)
        return f'![{alt}](/tutorials/others/images/{rel})'
    text = re.sub(
        r'!\[(?P<alt>[^\]]*)\]\(images/(?P<rel>[^)]+)\)',
        _img_repl, text)

    # 12a2. "pics/foo.png" — contrib modules use pics/ instead of images/.
    def _pics_img_repl(m: re.Match) -> str:
        alt = m.group("alt")
        hit = _IMAGE_INDEX.get(pathlib.Path(m.group("rel")).name)
        if hit:
            if hit.startswith("contrib_modules/"):
                return _emit_contrib_img(hit[len("contrib_modules/"):], alt)
            return f'![{alt}](/{hit})'
        return m.group(0)
    text = re.sub(
        r'!\[(?P<alt>[^\]]*)\]\(pics/(?P<rel>[^)]+)\)',
        _pics_img_repl, text)

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

    # 13a. Malformed Markdown link repair: `[text](URL]` where the
    #      closing paren was typo'd as a bracket.  Seen in
    #      text/install_tesseract.markdown line 4:
    #      `[this tutorials](http://.../gitbash_build]` — MyST renders
    #      the whole thing as literal text because Markdown can't close
    #      the link.  Rewrite to a proper `[text](URL)`.  Targeted at
    #      http(s) URLs only — keeps the regex tight enough to avoid
    #      misfiring on other `[...]` contexts (footnotes, ref-style
    #      links, etc.).
    text = re.sub(
        r"\[(?P<text>[^\]\n]+)\]\((?P<url>https?://[^\s\]\)]+)\]",
        lambda m: f"[{m.group('text')}]({m.group('url')})",
        text)

    # 13b. Auto-linkify bare URLs (Doxygen default; alternative is the
    #      linkify-it-py package). Skip code blocks/spans, existing
    #      markdown links, existing autolinks, and HTML attributes.
    #      Trailing sentence punctuation is left outside the autolink.
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
        # Split on inline `code` so URLs inside backticks stay untouched.
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

    # 14. Restore @verbatim stash (see step 0). Placeholder keys are private-
    #     use-area-safe strings so this is a literal replace.
    for _vk, _vv in _verbatim_stash.items():
        text = text.replace(_vk, _vv)

    return text


def _source_read(app, docname, source):
    if not (docname.startswith("tutorials/") or docname.startswith("js_tutorials/")
            or docname.startswith("py_tutorials/") or docname.startswith("tutorials_contrib/")
            or docname.startswith("api/")):
        return
    source[0] = _translate(source[0], docname)
    if docname == "tutorials/tutorials" and DOC_JS_MODULES:
        source[0] += (
            "\n\n```{toctree}\n:maxdepth: 1\n:caption: JavaScript Tutorials\n\n"
            "/js_tutorials/js_tutorials\n```\n"
        )
    if docname == "tutorials/tutorials" and DOC_PY_MODULES:
        source[0] += (
            "\n\n```{toctree}\n:maxdepth: 1\n:caption: Python Tutorials\n\n"
            "/py_tutorials/py_tutorials\n```\n"
        )
    if docname == "tutorials/tutorials" and CONTRIB_MODULES:
        source[0] += (
            "\n\n```{toctree}\n:maxdepth: 1\n:caption: Contrib Tutorials\n\n"
            "/tutorials_contrib/contrib_root\n```\n"
        )
    if docname == "tutorials/tutorials" and API_MODULES \
            and (SPHINX_INPUT_ROOT / "api" / "library_root.rst").exists():
        source[0] += (
            "\n\n```{toctree}\n:maxdepth: 1\n:caption: API Reference\n\n"
            "/api/library_root\n```\n"
        )

