"""Doc-build engine for the OpenCV Sphinx wrapper.

conf.py stays a thin Sphinx-settings file; everything heavy lives here:

* ``state``      — shared config, paths, tag maps, anchor indexes, constants
* ``xml_render`` — Doxygen XML -> Markdown primitives
* ``stubs``      — API-reference stub writers (groups / namespaces / classes)
* ``translate``  — Doxygen-flavored .markdown -> MyST (the source-read engine)
* ``patches``    — Sphinx C++ domain / breathe warning patches
* ``build``      — import-time orchestration that populates the shared indexes
"""
