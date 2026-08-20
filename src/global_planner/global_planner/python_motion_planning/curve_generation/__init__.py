from .polynomial_curve import Polynomial
from .bezier_curve import Bezier
from .bspline_curve import BSpline
from .dubins_curve import Dubins
from .reeds_shepp import ReedsShepp
from .cubic_spline import CubicSpline

__all__ = ["Polynomial", "Dubins", "ReedsShepp", "Bezier", "CubicSpline", "BSpline"]

# FemPosSmoother needs the optional `osqp` solver, which isn't part of this
# project's dependencies (this project only uses BSpline, above) -- imported
# lazily so the whole package doesn't fail to import for everyone else just
# because osqp isn't installed.
try:
    from .fem_pos_smooth import FemPosSmoother
    __all__.append("FemPosSmoother")
except ImportError:
    pass