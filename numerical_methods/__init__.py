import os 
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from .interpolation import (
    interpolate, 
    Gauss_interpolate,
    Lagrange_interpolate,
    Newton_divided_interpolate,
    Newton_finite_interpolate,
    Lagrange_interpolate_derivative
)

from .quadratures import *
from .nonlinear_solutions import *