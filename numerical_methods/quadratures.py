import numpy as np

from typing import Callable, Literal


NEWTON_COTES_WEIGHTS = (
    (1/2, 1/2), 
    (1/6, 4/6, 1/6),
    (1/8, 3/8, 3/8, 1/8),
    (7/90, 32/90, 12/90, 32/90, 7/90),
    (19/288, 75/288, 50/288, 50/288, 75/288, 19/288),
    (41/840, 216/840, 27/840, 272/840, 27/840, 216/840, 41/840)    
)

GAUSS_WEIGTHS = (
    ((0), (2)),
    ((-0.57735, 0.57735), (1, 1)),
    ((-0.774597, 0, 0.774597), (5/9, 8/9, 5/9)),
    ((-0.861136, -0.339981, 0.339981, 0.861136), (0.347855, 0.652145, 0.652145, 0.347855)),
)


def calculate_quadrature(
        function: Callable[[float], float],
        a: float, b: float, 
        n: int = 1000,
        method: Literal["left rectangles", "right rectangles", 
                        "central rectangles", "trapezoid", "Simpson",
                        "Weddle", "Newton Cotes", "Gauss"] = "trapezoid") -> float:
    """
    Возвращает приближенное значение интеграла функции.
    
    Args:
        function (Callable[[float], float]): функция для которой необходимо вычислить интеграл.
        a (float): нижняя граница интегрирования.
        b (float): верхняя граница интегрирования.
        n (int): количество интервалов.
        method (str): Метод для вычисления интеграла. Допустимые значения:
            - "left rectangles"
            - "right rectangles"
            - "central rectangles"
            - "trapezoid" (default)
            - "Simpson"
            - "Weddle"
            - "Newton Cotes"
            - "Gauss"
    
    Returns:
        float: вычисленное значение интеграла. 
    
    Raises:
        ValueError: 
            - Если указан несуществующий метод.
    """
    match method:
        case "left rectangles":
            result = calculate_left_rectangles(function, a, b, n)
        case "right rectangles":
            result = calculate_right_rectangles(function, a, b, n)
        case "central rectangles":
            result = calculate_central_rectangles(function, a, b, n)
        case "trapezoid":
            result = calculate_trapezoid(function, a, b, n)
        case "Simpson":
            result = calculate_Simpson(function, a, b, n)
        case "Weddle":
            result = calculate_Weddle(function, a, b, n)
        case "Newton Cotes":
            result = calculate_Newton_Cotes(function, a, b, n)
        case "Gauss":
            result = calculate_Gauss(function, a, b, n)
        case _: 
            raise ValueError(f"Метода `{method}` не существует.")
            
    return result
             

def calculate_left_rectangles(
        function: Callable[[float], float],
        a: float, b: float, 
        n: int = 1000) -> float: 
    """ Возвращает приближенное значение интеграла функции с помощью формулы левых прямоугольников.

    Args:
        function (Callable[[float], float]): функция для которой необходимо вычислить интеграл.
        a (float): нижняя граница интегрирования.
        b (float): верхняя граница интегрирования.
        n (int): количество интервалов.

    Returns:
        float: вычисленное значение интеграла. 
    """
    h = (b - a) / n
    return sum(map(function, np.linspace(a, b - h, n - 1))) * h


def calculate_right_rectangles(
        function: Callable[[float], float],
        a: float, b: float, 
        n: int = 1000) -> float: 
    """ Возвращает приближенное значение интеграла функции с помощью формулы правых прямоугольников.

    Args:
        function (Callable[[float], float]): функция для которой необходимо вычислить интеграл.
        a (float): нижняя граница интегрирования.
        b (float): верхняя граница интегрирования.
        n (int): количество интервалов (default 1000).

    Returns:
        float: вычисленное значение интеграла. 
    """
    h = (b - a) / n
    return sum(map(function, np.linspace(a + h, b, n - 1))) * h


def calculate_central_rectangles(
        function: Callable[[float], float],
        a: float, b: float, 
        n: int = 1000) -> float: 
    """ Возвращает приближенное значение интеграла функции с помощью формулы центральных прямоугольников.

    Args:
        function (Callable[[float], float]): функция для которой необходимо вычислить интеграл.
        a (float): нижняя граница интегрирования.
        b (float): верхняя граница интегрирования.
        n (int): количество интервалов (default 1000).

    Returns:
        float: вычисленное значение интеграла. 
    """
    h = (b - a) / n
    return sum(map(lambda x: function(x + h/2), np.linspace(a, b, n))) * h


def calculate_trapezoid(
        function: Callable[[float], float],
        a: float, b: float, 
        n: int = 1000) -> float: 
    """ Возвращает приближенное значение интеграла функции с помощью формулы трапеции.

    Args:
        function (Callable[[float], float]): функция для которой необходимо вычислить интеграл.
        a (float): нижняя граница интегрирования.
        b (float): верхняя граница интегрирования.
        n (int): количество интервалов (default 1000).

    Returns:
        float: вычисленное значение интеграла. 
    """
    h = (b - a) / n
    return sum(map(lambda x: (function(x) + function(x + h)) / 2, np.linspace(a, b, n))) * h


def calculate_Simpson(
        function: Callable[[float], float],
        a: float, b: float, 
        n: int = 1000) -> float: 
    """ Возвращает приближенное значение интеграла функции с помощью формулы Симпсона.

    Args:
        function (Callable[[float], float]): функция для которой необходимо вычислить интеграл.
        a (float): нижняя граница интегрирования.
        b (float): верхняя граница интегрирования.
        n (int): количество интервалов / 2 (default 1000).

    Returns:
        float: вычисленное значение интеграла. 
    """
    h = (b - a) / (n * 2)
    return sum(map(lambda x: function(x) + 4 * function(x + h/2) + function(x + h), np.linspace(a, b - h, n - 1))) * h / 3


def calculate_Weddle(
        function: Callable[[float], float],
        a: float, b: float, 
        n: int = 1000) -> float: 
    """ Возвращает приближенное значение интеграла функции с помощью формулы Уэдлля.

    Args:
        function (Callable[[float], float]): функция для которой необходимо вычислить интеграл.
        a (float): нижняя граница интегрирования.
        b (float): верхняя граница интегрирования.
        n (int): количество интервалов / 6 (default 1000).

    Returns:
        float: вычисленное значение интеграла. 
    """
    h = (b - a) / (n * 6)
    return sum(map(lambda x: function(x) + 5 * function(x + h/6) + function(x + h/3) + 6 * function(x + h/2) \
                             + function(x + 2 * h/3) + 5 * function(x + 5 * h / 6) + function(x + h), np.linspace(a, b - h, n - 1))) * h * 0.3
    
    
def calculate_Newton_Cotes(
        function: Callable[[float], float],
        a: float, b: float, 
        n: int = 4) -> float: 
    """ Возвращает приближенное значение интеграла функции с помощью формулы Ньютона-Котеса.

    Args:
        function (Callable[[float], float]): функция для которой необходимо вычислить интеграл.
        a (float): нижняя граница интегрирования.
        b (float): верхняя граница интегрирования.
        n (int): количество интервалов (default 4).

    Returns:
        float: вычисленное значение интеграла. 
        
    Raises:
        - Если n больше 6 или меньше 1.
    """
    if n < 1 or n > 6:
        raise ValueError(f"Не поддерживается `n` = {n}")
    
    h = (b - a) / n
    weights = NEWTON_COTES_WEIGHTS[n - 1]
    return sum([function(a + ind * h) * weights[ind] for ind in range(n)]) * (b - a)
    
    
def calculate_Gauss(
        function: Callable[[float], float],
        a: float, b: float, 
        n: int = 4) -> float: 
    """ Возвращает приближенное значение интеграла функции с помощью формулы Гаусса.

    Args:
        function (Callable[[float], float]): функция для которой необходимо вычислить интеграл.
        a (float): нижняя граница интегрирования.
        b (float): верхняя граница интегрирования.
        n (int): количество интервалов (default 4).

    Returns:
        float: вычисленное значение интеграла. 
    
    Raises:
        - Если n больше 4 или меньше 1.
    """
    if n < 1 or n > 4:
        raise ValueError(f"Не поддерживается `n` = {n}")
    
    t, weights = GAUSS_WEIGTHS[n - 1]
    c = (b - a)/2
    return sum([function((b + a) / 2 + c * t[ind]) * weights[ind]  for ind in range(n)]) * c
