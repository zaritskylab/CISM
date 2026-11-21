import pickle
from pathlib import Path


class ArtifactState:
    """Read saved state without importing optional CISM analysis dependencies."""
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        self.args = args


class StateReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(('cism.', 'CISM.')):
            return ArtifactState
        return super().find_class(module, name)


def load_artifact(path):
    """A serialized CISM run: tutorial 02's <dataset>_cism_ready.pkl dict, or an older pickled CISM object.

    Every cism class comes back as an ArtifactState holding the saved attributes, so artifacts load whatever version
    of the cism code wrote them.
    """
    with Path(path).open('rb') as handle:
        return StateReader(handle).load()
