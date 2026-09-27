"""Runner scripts executed *inside* each adapter's environment.

These modules are invoked as ``python -m glmbench.runners.<x> <request.json>`` in
the adapter's own env (which may carry torch / transformers / evo2). They are **never
imported by core** — core only shells out to them via a runner backend. This is the
one place torch-family imports are allowed (the torch-free import gate).
"""
