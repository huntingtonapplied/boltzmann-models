"""
boltzmann-models — runnable inference package (compiled distribution, not source).

Public surface, two lines:

    from boltzmann_models import load
    model = load("<craft_id>")
    fields = model.predict(field_grids)   # physical grids in -> physical fields out

Design (see MODEL_INFERENCE_PACKAGE_STRATEGY.md):
  * The model forward + de-normalization are baked into an ONNX graph and run under
    `onnxruntime` — no torch, no PyG, CPU-only.
  * The featurization (grid -> mesh-graph node/edge features, per the craft's input
    contract) is the real engine code, shipped **compiled** as `_featurizer` (an opaque
    extension module), never as readable source.
  * What you provide is a physical specification of the scene as dense field grids
    (e.g. a doping profile + geometry). What you get back are physical output fields.
    These are fast surrogates, not the ground-truth solver — validate against the
    numerical engine where correctness is critical.

This module is intentionally thin: it wires public inputs to the compiled featurizer
and the ONNX session, and reshapes outputs back to physical grids. It carries no model
architecture, no normalization constants, and no training coupling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover - dependency is declared in pyproject
    raise ImportError(
        "onnxruntime is required to run inference. Install with `pip install onnxruntime`."
    ) from e

# The compiled featurizer (Nuitka/Cython extension). Exposes:
#   grid_to_graph(field_grids: dict[str, np.ndarray], contract) -> dict[str, np.ndarray]
#   get_contract(craft_id: str) -> contract object (with .node_feature_dim etc.)
# It is opaque by design; there is no readable-source fallback.
from . import _featurizer  # type: ignore

_ROOT = Path(__file__).resolve().parent
_MODELS_DIR = _ROOT / "models"


def _meta() -> dict:
    p = _ROOT / "_meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def list_models() -> List[str]:
    """Craft ids shipped in this package (those with published weights)."""
    if not _MODELS_DIR.exists():
        return []
    return sorted(d.name for d in _MODELS_DIR.iterdir() if (d / "model.onnx").exists())


class Model:
    """A single loaded craft: a physical -> physical surrogate."""

    def __init__(self, craft_id: str):
        cdir = _MODELS_DIR / craft_id
        onnx_path = cdir / "model.onnx"
        if not onnx_path.exists():
            avail = ", ".join(list_models()) or "(none published in this build)"
            raise FileNotFoundError(
                f"No model weights for '{craft_id}'. Available: {avail}."
            )
        self.craft_id = craft_id
        self.config: dict = json.loads((cdir / "public_config.json").read_text())
        self._contract = _featurizer.get_contract(craft_id)
        # Single-threaded, deterministic; CPU provider only.
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_names = [i.name for i in self._sess.get_inputs()]

    @property
    def output_channels(self) -> List[str]:
        return list((self.config.get("output") or {}).get("channels") or [])

    def predict(self, field_grids: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run the surrogate.

        Parameters
        ----------
        field_grids : mapping of name -> dense array, each shaped [Nx, Ny, Nz]
            The physical scene. Which fields a craft consumes is documented in its
            model card (`config["input"]`); the encoding into model features is handled
            inside the package. Fields the contract expects but you omit are zero-filled.

        Returns
        -------
        dict of channel-name -> array shaped [Nx, Ny, Nz], in physical units.
        """
        if not isinstance(field_grids, Mapping) or not field_grids:
            raise ValueError("field_grids must be a non-empty mapping of name -> [Nx,Ny,Nz] array")

        grids = {k: np.asarray(v, dtype=np.float32) for k, v in field_grids.items()}
        # Each field is [Nx,Ny,Nz] (scalar) or [Nx,Ny,Nz,C] (vector, e.g. normalized
        # position). All fields share the same [Nx,Ny,Nz] spatial grid.
        first = next(iter(grids.values()))
        if first.ndim < 3:
            raise ValueError(f"each field grid must be at least 3-D [Nx,Ny,Nz(,C)]; got {first.shape}")
        shape = tuple(first.shape[:3])
        for k, v in grids.items():
            if tuple(v.shape[:3]) != shape:
                raise ValueError(f"field grid '{k}' spatial shape {v.shape[:3]} != {shape} (all fields share [Nx,Ny,Nz])")

        graph = _featurizer.grid_to_graph(grids, self._contract)
        feeds = {
            "node_features": np.asarray(graph["node_features"], dtype=np.float32),
            "edge_index": np.asarray(graph["edge_index"], dtype=np.int64),
            "edge_features": np.asarray(graph["edge_features"], dtype=np.float32),
        }
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        out = self._sess.run(None, feeds)[0]  # [N, C], physical (de-norm baked in)

        n = int(np.prod(shape))
        if out.shape[0] != n:
            raise RuntimeError(
                f"model returned {out.shape[0]} nodes but the grid has {n} — grid/contract mismatch"
            )
        channels = self.output_channels or [f"c{i}" for i in range(out.shape[1])]
        return {
            ch: out[:, i].reshape(shape)
            for i, ch in enumerate(channels)
            if i < out.shape[1]
        }


def load(craft_id: str) -> Model:
    """Load a craft by id (e.g. ``load("boltzmann_carrier_density_v1")``)."""
    return Model(craft_id)


__all__ = ["load", "list_models", "Model"]
