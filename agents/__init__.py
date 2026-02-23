from .agent1_scout import ScoutAgent
from .agent2_radiologist import RadiologistAgent
from .agent3_archivist import ArchivistAgent
from .agent5_regressor import RegressorAgent
from .agent6_ensemble import EnsembleAgent

try:
    from .agent4_vlm_client import LMStudioAnthropologistAgent
except ImportError:
    LMStudioAnthropologistAgent = None

__all__ = [
    "ScoutAgent",
    "RadiologistAgent",
    "ArchivistAgent",
    "LMStudioAnthropologistAgent",
    "RegressorAgent",
    "EnsembleAgent",
]
