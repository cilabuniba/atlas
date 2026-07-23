from .planetoid_run import PlanetoidRun
from src.utils import ParameterKeys


class WikiCSRun(PlanetoidRun):
    def _init_loaders(self):
        super()._init_loaders()
        self.masks = {
            phase: mask[:, 0] if phase != ParameterKeys.TEST else mask
            for phase, mask in self.masks.items()
        }
