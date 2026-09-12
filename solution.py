"""Competition entrypoint. Artifacts are loaded relative to this file."""
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from connectome.runtime import load_predictor


class PredictionModel:
    def __init__(self):
        self._model = load_predictor(Path(__file__).resolve().parent / "submission_config.json")

    def predict(self, data_point):
        return self._model.predict(data_point)
