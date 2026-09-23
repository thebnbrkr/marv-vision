"""marv-vision: MARV's edit-and-measure approach, pointed at vision-language models.

Status: scaffold. Run scripts/inspect_model.py before trusting any attribute name.
"""
__version__ = "0.0.1"

from .arch import TowerSpec, describe_model, tower_specs

__all__ = ["TowerSpec", "describe_model", "tower_specs"]
