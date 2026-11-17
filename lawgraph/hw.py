"""
Hardware helpers shared by the precompute (viewer_build) and the API server (serve).

    cores(), ram_gb()          what the machine has
    gpu()                      "NVIDIA ... (97887 MiB)" from nvidia-smi, or None
    tune(con, threads, mem)    PRAGMA threads / memory_limit / temp_directory on one DuckDB connection

tune() only changes session settings; nothing is written to the database file, so it is safe on the
read-only connections every process in this project uses.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile


def cores() -> int:
    return os.cpu_count() or 1


def ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        return 16.0


def gpu() -> str | None:
    """Name and memory of the first NVIDIA GPU, or None if nvidia-smi is missing or reports nothing."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
    if not line:
        return None
    name, _, mem = line.partition(",")
    return f"{name.strip()} ({mem.strip()})"


def tune(con, threads: int | None = None, memory_gb: float | None = None):
    """Give one connection `threads` DuckDB threads (default: all cores) and a memory limit (default: half
    of RAM). Spill files go to the system temp dir rather than next to the database."""
    threads = max(1, int(threads or cores()))
    memory_gb = max(1, int(memory_gb if memory_gb else ram_gb() * 0.5))
    tmp = os.path.join(tempfile.gettempdir(), "lawgraph_duckdb_tmp")
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"SET memory_limit='{memory_gb}GB'")
    con.execute(f"SET temp_directory='{tmp}'")
    return con
