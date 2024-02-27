__version__ = "0.3.2"

from .managers.detection_manager import DetectionAnalysisManager
from .managers.classification_manager import ClassificationAnalysisManager

__all__ = ["DetectionAnalysisManager", "ClassificationAnalysisManager"]