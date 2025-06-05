import numpy as np

from typing import Callable
from interpolation import Lagrange_interpolate_derivative


def chord_method(function: Callable[[float], float],
                 a: float, b: float, eps: float = 1e-9) -> float:
    """Находит и возвращает корень заданной функции (f(x) = 0) на заданном интервале с помощью метода хорд (секущих).

    - Замечание первое: корень должен быть единственный на заданном интервале.
    - Замечание второе: корень должен быть нечётной кратности.

    Args:
        function (Callable[[float], float]): функция для которой необходимо найти корень f(x) = 0.
        a (float): начало интервала поиска корня.
        b (float): конец интервала поиска корня.
        eps (float): точность с которой необходимо найти корень (default 1e-9).

    Returns:
        float: абцисса корня (f(x) = 0).

    Raises:
        - Если в какой-то момент итерации алгоритма знак на концах интервала будет одинаков (смотрите замечание).
    """
    if function(a) * function(b) > 0:
        raise ValueError("Знаки на концах интервала одинаковы!")

    x = eps
    last_x = 0
    while abs(x - last_x) >= eps:
        last_x = x
        x = a - function(a) / (function(b) - function(a)) * (b - a)

        if function(a) * function(x) < 0:
            b = x
        elif function(b) * function(x) < 0:
            a = x
        else:
            raise ValueError("Знаки на концах интервала одинаковы!")

    return a - function(a) / (function(b) - function(a)) * (b - a)


def Newton_method(function: Callable[[float], float],
                  a: float, b: float, eps: float = 1e-9,
                  n: int = 10) -> float:
    """Находит и возвращает корень заданной функции (f(x) = 0) на заданном интервале
    с помощью метода Ньютона (касательных).

    - Замечание первое: корень должен быть единственный на заданном интервале.
    - Замечание второе: корень должен быть нечётной кратности.

    Args:
        function (Callable[[float], float]): функция для которой необходимо найти корень f(x) = 0.
        a (float): начало интервала поиска корня.
        b (float): конец интервала поиска корня.
        eps (float): точность с которой необходимо найти корень (default 1e-9).
        n (int): количество точек для поиска производной методом дифференцирования полинома Лагранжа (default 10).

    Returns:
        float: абцисса корня (f(x) = 0).

    Raises:
        - Если в какой-то момент итерации алгоритма знак на концах интервала будет одинаков (смотрите замечание).
    """
    if function(a) * function(b) > 0:
        raise ValueError("Знаки на концах интервала одинаковы!")

    grid = np.linspace(a, b, n)
    values = [function(item) for item in grid]

    x = a if function(a) * Lagrange_interpolate_derivative(a, grid, values, 2) > 0 else b
    last_x = x - eps - 1
    while abs(x - last_x) >= eps:
        last_x = x
        x -= function(x) / Lagrange_interpolate_derivative(x, grid, values)

    return x
