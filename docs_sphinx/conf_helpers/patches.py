"""Runtime patches for Sphinx's C++ domain and breathe warnings.

Importing this module applies them (each helper invokes itself at import):
guard the C++ xref resolver against an upstream assertion, and silence two
classes of benign breathe / cpp-domain warnings. conf.py imports it for effect.
"""
from __future__ import annotations


def _patch_cpp_xref_resolver():
    """Guard the C++ domain's xref resolver against an upstream assertion on
    template-class cross-references. Sphinx 8.1.x asserts `parentSymbol`
    inside `_resolve_xref_inner`; some breathe-emitted class-page xrefs
    (e.g. `cv::Affine3<T>`) trigger that path with no parent symbol.
    Treat it as an unresolved xref instead of crashing the whole build."""
    try:
        from sphinx.domains.cpp import CPPDomain
    except ImportError:
        return
    original = CPPDomain._resolve_xref_inner

    def guarded(self, env, fromdocname, builder, typ, target, node, contnode):
        try:
            return original(self, env, fromdocname, builder, typ, target,
                            node, contnode)
        except AssertionError:
            return None, None
    CPPDomain._resolve_xref_inner = guarded


_patch_cpp_xref_resolver()


def _silence_breathe_anon_enum_warning():
    """Suppress the docutils "Invalid C++ declaration: Expected identifier
    in nested name." warning that breathe triggers when rendering an
    *anonymous* nested enum inside a struct (e.g. `cv::MatShape` has an
    enum whose `<name>` element is empty — Doxygen XML allows it, but the
    Sphinx C++ domain parser rejects the resulting declaration).

    The render is otherwise fine (the enum values still appear); only the
    parse-time warning is noise. We filter it via a Python logging filter
    rather than monkey-patching the parser so the same fix survives Sphinx
    version bumps."""
    import logging
    class _AnonEnumFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not (
                ("Invalid C++ declaration" in msg
                 and "Expected identifier in nested name" in msg)
                or
                ("Duplicate C++ declaration" in msg
                 and "cpp:None::" in msg)
            )
    # docutils warning messages route through both 'sphinx' and 'docutils'
    # loggers depending on entry point; attach to both for coverage.
    for _name in ("sphinx", "docutils"):
        logging.getLogger(_name).addFilter(_AnonEnumFilter())


_silence_breathe_anon_enum_warning()


def _silence_cpp_duplicate_declaration_warning():
    """Suppress Sphinx's "Duplicate C++ declaration, also defined at …" warning.

    The generated API tree intentionally documents a symbol in more than one
    place — a free function appears both on its module *group* page and on its
    *namespace* page, exactly as Doxygen's own HTML does. Sphinx's C++ domain,
    however, keeps a single global symbol table and warns once per symbol that
    is declared on a second page. For the `cv` namespace that is thousands of
    benign warnings.

    `suppress_warnings = ['cpp.duplicate_declaration']` does NOT catch these:
    in Sphinx 8.1 the warning is logged untyped (see
    sphinx/domains/cpp/__init__.py — `logger.warning(msg, location=signode)`
    with no `type=`/`subtype=`), so the only reliable hook is a logging filter,
    matching the approach already used for the anonymous-enum warning above."""
    import logging
    class _DupDeclFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "Duplicate C++ declaration" not in record.getMessage()
    for _name in ("sphinx", "docutils"):
        logging.getLogger(_name).addFilter(_DupDeclFilter())


_silence_cpp_duplicate_declaration_warning()


