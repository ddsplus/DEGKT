from . import Model as _model

if hasattr(_model, "DGEKT"):
    DGEKT = _model.DGEKT
    __all__ = ["DGEKT"]
elif hasattr(_model, "DKT"):
    DKT = _model.DKT
    __all__ = ["DKT"]
else:
    __all__ = []
