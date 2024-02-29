__version__ = "0.4.0rc2646"

from .managers.detection_manager import DetectionAnalysisManager
from .managers.classification_manager import ClassificationAnalysisManager

__all__ = ["DetectionAnalysisManager", "ClassificationAnalysisManager"]