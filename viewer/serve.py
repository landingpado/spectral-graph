"""Old entry point, kept so `python viewer/serve.py [--port 8080] [--no-open]` still works.

It now starts the whole explorer (this folder as a static site plus the /api endpoints) by calling
lawgraph.serve. The direct command is:

    python -m lawgraph.serve --db graph.duckdb --port 8080
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lawgraph.serve import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--db" not in argv:
        argv = ["--db", os.path.join(ROOT, "graph.duckdb"), *argv]
    main(argv)
