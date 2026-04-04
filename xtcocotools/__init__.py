"""Compatibility shim for environments where xtcocotools wheels are unavailable.

This re-exports pycocotools APIs under the xtcocotools namespace, which is
enough for mmpose runtime imports used in MuseTalk inference.
"""

from .coco import COCO  # noqa: F401
from .cocoeval import COCOeval  # noqa: F401
