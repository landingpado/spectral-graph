"""Lenses: analyses that take a Graph (built from a Snapshot) and return arrays aligned with its ids."""
from .base import Graph, Lens, build, laplacian, select          # noqa: F401
from .spectral import Spectral                                    # noqa: F401
from .community import Leiden, lineage                            # noqa: F401
from .align import align, align_sequence, drift, principal_angles                 # noqa: F401
