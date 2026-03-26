"""Notebook / Colab helpers: dataset-scoped paths matching pipeline.py and phase scripts."""

from pathlib import Path
from typing import Any, Dict, Tuple, Union


def get_dataset_name(cfg: Dict[str, Any]) -> str:
    """Folder name under workspace/data_root (same logic as pipeline.py)."""
    d = cfg.get("dataset", {}) or {}
    if d.get("output_name"):
        return str(d["output_name"])
    source = d.get("source", "json")
    if source == "json":
        return "custom"
    if source == "huggingface":
        repo = d.get("hf_repo", "") or ""
        return repo.split("/")[-1] if repo else "hf"
    if source == "json_levels":
        return "levels"
    return "custom"


def phase_output_roots(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    """Return ``(data_root_dataset, workspace_dataset)`` as :class:`Path`.

    Aligns with ``pipeline.py`` layout: ``data_root / {dataset} / …`` and
    ``workspace / {dataset} / …`` (activations/answers/phase_* vs labels/plots/logs).

    Parameters
    ----------
    cfg : dict
        Must contain ``cfg["paths"]["workspace"]`` and ``cfg["paths"]["data_root"]``
        (e.g. loaded YAML after Colab overrides).
    """
    ws = Path(cfg["paths"]["workspace"]).expanduser()
    dr = Path(cfg["paths"]["data_root"]).expanduser()
    dset = get_dataset_name(cfg)
    return dr / dset, ws / dset


def phase_c_subspace_basis(
    cfg: Union[Dict[str, Any], Path, str],
    level_run_id: str,
    layer: int,
    which: str = "wrong",
) -> Path:
    """Path to ``correct_basis.npy`` or ``wrong_basis.npy`` for code-geometry Phase C."""
    if not isinstance(cfg, dict):
        import yaml

        p = Path(cfg)
        with open(p) as f:
            cfg = yaml.safe_load(f)
    dr_dset, _ = phase_output_roots(cfg)
    w = which if which.endswith("_basis") else f"{which}_basis"
    if not w.endswith(".npy"):
        w = f"{w}.npy"
    return dr_dset / "phase_c" / "subspaces" / level_run_id / f"layer_{layer}" / w
