from contextlib import asynccontextmanager
import logging
from threading import Lock
from pathlib import Path
import os

from ml.c3.component3_severity_bridge import Component3SeverityBridge

logger = logging.getLogger(__name__)
BUNDLE = Path(os.environ.get('C3_BUNDLE_PATH', Path(__file__).resolve().parents[3] / 'ml' / 'c3'))


class SurfaceAnalysis:
    def __init__(self):
        self.bridge = None
        self.lock = Lock()

    def load(self):
        self.bridge = Component3SeverityBridge(BUNDLE, device='cpu')

    def analyze(self, rgb, detections):
        if self.bridge is None:
            raise RuntimeError('C3 model unavailable')
        with self.lock:
            result = self.bridge.predict(rgb, detections, include_maps=True)
        result['bundle_version'] = self.bridge.bridge.service.backend.base.meta['bundle_version']
        result['map_coordinates'] = {'space': 'full_frame', 'width': 256, 'height': 192, 'units': 'RAW', 'physical_scale_verified': False}
        return result


surface_analysis = SurfaceAnalysis()


@asynccontextmanager
async def lifespan(app):
    try:
        surface_analysis.load()
    except Exception:
        logger.exception('C3 model failed to initialize')
    yield
