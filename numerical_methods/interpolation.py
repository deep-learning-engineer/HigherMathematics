import pandas as pd
import numpy as np

from typing import Callable, Literal
from collections.abc import Sequence
from itertools import combinations
from functools import reduce
from bisect import bisect
from math import sin, log, log10, exp, factorial
from utils import (
    get_divided_differences_table, 
    get_k_nearest_nodes,
    get_finite_differences_table
)


pd.set_option("display.width", 200)


def interpolate(
        x: float, 
        grid: Sequence[float],
        values: Sequence[float],
        is_uniform: bool = False,
        method: Literal["Lagrange", "Newton div", "Newton fin", "Gauss"] = "Lagrange") -> float:
    """
    Возвращает приближенное значение заданной функции в точке `x` с помощью интерполяционного многочлена.
    
    Args:
        x (float): точка в которой нужно вычислить значение функции.
        grid (Sequence[float]): сетка узлов.
        values (Sequence[float]): значение функции в узлах. 
        is_uniform (bool): True если сетка равномерная, иначе False (default False).
        method (str): Метод интерполяции. Допустимые значения:
            - "Lagrange" (default)
            - "Newton div"
            - "Newton fin"
            - "Gauss"
    
    Returns:
        float: вычисленной значение в точке `x`. 
    
    Raises:
        ValueError: 
            - Если указан несуществующий метод интерполяции.
            - Если выбран одиз из методов "Newton fin", "Gauss" с неравномерной сеткой `is_uniform`= False.
    """
    if not is_uniform and method in ("Newton fin", "Gauss"):
        raise ValueError(f"Нельзя использовать метод `{method}` с неравномерной сеткой. Используйте равномерную сетку и укажите `is_uniform` = True.")
    
    
    match method: 
        case "Lagrange":
            y_interpolate = Lagrange_interpolate(x, grid, values)
        case "Newton div":
            y_interpolate = Newton_divided_interpolate(x, grid, values) 
        case "Newton fin":
            y_interpolate = Newton_finite_interpolate(x, grid, values)
        case "Gauss":
            y_interpolate = Gauss_interpolate(x, grid, values)
        case _: 
            raise ValueError(f"Метода `{method}` не существует.")

    return y_interpolate


def Lagrange_interpolate(x: float, grid: Sequence[float], values: Sequence[float]) -> float:
    """Возвращает приближенное значение заданной функции в точке `x` с помощью интерполяционного многочлена Лагранжа.
    
    Args: 
        x (float): точка в которой нужно вычислить значение функции.
        grid (Sequence[float]): узлы сетки.
        values (Sequence[float]): значения функции в узлах сетки.
    
    Returns:
        float: вычисленной значение в точке `x`. 
        
    Raises:
        ValueError: Если `x` не находится между узлами сетки.
    """
    if x < grid[0] or x > grid[-1]:
        raise ValueError("`x` должен находиться между первым и последним узлом.")
    
    count = len(grid)
    y_interpolate = 0
        
    for i in range(count):
        prod = 1
        
        for j in range(count):
            if i == j: continue
            prod *= (x - grid[j]) / (grid[i] - grid[j])
        
        y_interpolate += prod * values[i]
    
    return y_interpolate 


def Newton_divided_interpolate(x: float, grid: Sequence[float], values: Sequence[float]): 
    """Возвращает приближенное значение заданной функции в точке `x` с помощью интерполяционного многочлена Ньютона с разделенными разностями.
    
    Args: 
        x (float): точка в которой нужно вычислить значение функции.
        grid (Sequence[float]): узлы сетки.
        values (Sequence[float]): значения функции в узлах сетки.
    
    Returns:
        float: вычисленной значение в точке `x`. 
        
    Raises:
        ValueError: Если `x` не находится между узлами сетки.
    """
    if x < grid[0] or x > grid[-1]:
        raise ValueError("`x` должен находиться между первым и последним узлом.")
    
    count = len(grid)
    coeffs = get_divided_differences_table(grid, values)
    y_interpolate, prod = 0, 1.0
        
    for i in range(count):
        y_interpolate += coeffs[i] * prod
        prod *= (x - grid[i]) 
    
    return y_interpolate


def Newton_finite_interpolate(x: float, grid: Sequence[float], values: Sequence[float]): 
    """Возвращает приближенное значение заданной функции в точке `x` с помощью интерполяционного многочлена Ньютона с конечными разностями.
    
    Args: 
        x (float): точка в которой нужно вычислить значение функции.
        grid (Sequence[float]): узлы сетки.
        values (Sequence[float]): значения функции в узлах сетки.
    
    Returns:
        float: вычисленной значение в точке `x`. 
        
    Raises:
        ValueError: Если `x` не находится между узлами сетки.
    """
    h = grid[1] - grid[0]
    if x < grid[0] or x > grid[-1]:
        raise ValueError("`x` должен находиться между первым и последним узлом.")
    elif x - grid[0] < grid[-1] - x:
        coeffs = get_finite_differences_table(values) 
        t = (x - grid[0]) / h
    else:
        coeffs = get_finite_differences_table(values[::-1])
        t = (grid[-1] - x) / h
   
    y_interpolate = 0
    prod = 1
    for ind in range(len(values)):
        y_interpolate += coeffs[ind][0] * prod
        prod *= (t - ind) / (ind + 1)
        
    return y_interpolate


def Gauss_interpolate(x: float, grid: Sequence[float], values: Sequence[float]):
    """Возвращает приближенное значение заданной функции в точке `x` с помощью интерполяционного многочлена Гаусса с конечными разностями.
    
    Args: 
        x (float): точка в которой нужно вычислить значение функции.
        grid (Sequence[float]): узлы сетки.
        values (Sequence[float]): значения функции в узлах сетки.
    
    Returns:
        float: вычисленной значение в точке `x`. 
        
    Raises:
        ValueError: Если `x` не находится между узлами сетки.
    """
    if x < grid[0] or x > grid[-1]:
        raise ValueError("`x` должен находиться между первым и последним узлом.")
    
    ind = bisect(grid, x)
    if x - grid[ind - 1] < grid[ind] - x:
        ind -= 1
        if len(grid[ind:]) < len(grid[:ind]):
            new_grid = values[ind - len(grid[ind:]) + 1 : ind] + values[ind:]
        else:
            new_grid = values[:ind] + values[ind: ind + len(grid[:ind]) + 2]
    else: 
        if len(grid[ind:]) < len(grid[:ind]):
            new_grid = values[ind - len(grid[ind:]) : ind] + values[ind:]
        else:
            new_grid = values[:ind] + values[ind: ind + len(grid[:ind]) + 1]
    

    coeffs = get_finite_differences_table(new_grid)
    t = (x - grid[ind]) / (grid[1] - grid[0])
    
    prod, cur_ind = t, len(new_grid) // 2 - (len(new_grid) % 2 == 0)
    y_interpolate = coeffs[0][cur_ind] + coeffs[1][cur_ind] * t
    
    l = 1 if x - grid[ind - 1] < grid[ind] - x else -1
    for ind in range(2, len(new_grid)):
        cur_ind -= int(ind % 2 == 0)
        prod *= (t - l * (1 if ind % 2 else -1)) / ind 
        y_interpolate += coeffs[ind][cur_ind] * prod

        if ind % 2:
            l += -1 if l < 0 else 1
    
    return y_interpolate


def Lagrange_interpolate_derivative(x: float, grid: Sequence[float], values: Sequence[float], k: int = 1):
    """Возвращает приближенное значение k-ой производной заданной функции в точке `x` с помощью интерполяционного дифференцирования многочлена Лагранжа.
    
    Args: 
        x (float): точка в которой нужно вычислить производную заданной функции.
        grid (Sequence[float]): узлы сетки.
        values (Sequence[float]): значения функции в узлах сетки.
        k (int): порядок производной (default 1). 
    
    Returns:
        float: вычисленное значение в точке `x`. 
        
    Raises:
        ValueError: Если `x` не находится между узлами сетки.
    """
    if x < grid[0] or x > grid[-1]:
        raise ValueError("`x` должен находиться между первым и последним узлом.") 
    
    count = len(grid)
    if count <= k + 1:
        return 0
    
    y_interpolate = 0
    
    comb = list(combinations([ind for ind in range(count)], count - k - 1)) 
    for i in range(count):
        prod = 1
        
        for j in range(count):
            if i == j: continue
            prod /= (grid[i] - grid[j])
            
        for item in comb:
            if i in item:
                continue 
        
            y_interpolate += reduce(lambda x, y: x * y, [x - grid[ind] for ind in item]) * prod * values[i]
    
    return y_interpolate * factorial(k)
    
        
def get_interpolation_error_bounds(function: Callable[[float], float], 
                                   derivative_function: Callable[[float], float],
                                   min_derivative: Callable[[float], float], max_derivative: Callable[[float], float],
                                   grid: Sequence[float], 
                                   x: float,
                                   n: int = 1) -> tuple[float]:
    """
    Возвращает нижнюю и верхнюю границы ошибки интерполирования и приближенное значение заданной функции в точке `x`.
    
    Args: 
        function (Callable[[float], float]): функция для которой необходимо вычислить интервал.
        derivative_function (Callable[[float], float]): производная n + 1 порядка исходной функции.
        min_derivative Callable[[float], float]: функция, которая определяет точку в которой производная имеет наименьшее значение.
        max_derivative Callable[[float], float]: функция, которая определяет точку в которой производная имеет наибольшее значение.
        grid (Sequence[float]): сетка интерполяции.
        x (float): точка в которой необходимо вычислить интервал.
        n (int): порядок интерполирующего полинома (default 1).
    
    Returns:
        tuple[float]: граница интервала ошибки интерполяции и приближенное значение заданной функции в точке `x`, посчитаное по Лагранжу и Ньютону.
    """
    nearest_nodes = get_k_nearest_nodes(grid, x, k=n+1)
    derivative_min, derivative_max = derivative_function(min_derivative(nearest_nodes)), derivative_function(max_derivative(nearest_nodes))
    
    values = []
    f = factorial(n + 1)
    prod = 1 / f
    
    try:
        for value in nearest_nodes:
            prod *= (x - value)
            values.append(function(value))
    except: 
        raise("Нельзя вычислить значение функции в заданной точке.")  

    derivative_min, derivative_max = sorted([derivative_min * prod, derivative_max * prod])
    return derivative_min, derivative_max, interpolate(x, nearest_nodes, values,  method="Lagrange"), \
           interpolate(x, nearest_nodes, values,  method="Newton_div")


def test_uniform_grid() -> None:
    """Тестирование интерполяции Лагранжа/ Ньютона с равномерной сеткой."""
    for ind, (function, interval, data) in enumerate(zip(functions, intervals, interpolation_values)):
        grid = np.linspace(*interval)
        
        try:
            values = [function(x) for x in grid]
        except: 
            print("Нельзя вычислить значение функции в заданной точке.")
            print("---")
            continue 
        
        print(f"Test: {ind + 1}")
        for x in data:
            print(f"Абсолютная ошибка Лагранжа в точке {x}:", abs_error(interpolate(x, grid, values, method="Lagrange"), function(x)))
            print(f"Абсолютная ошибка Ньютона в точке {x}:", abs_error(interpolate(x, grid, values, method="Newton_div"), function(x)))
            print("---")
        print()
        

def test_non_uniform_grid() -> None:
    """Тестирование интерполяции Лагранжа/ Ньютона с неравномерной сеткой."""
    grid = [0.005, 0.0123, 0.051, 0.545, 0.675, 0.782, 0.895, 0.931, 1.012, 1.213, 1.324, 1.451, 1.567, 1.892, 2.123]
    
    for ind, (function, data) in enumerate(zip(functions, interpolation_values)):
        try:
            values = [function(x) for x in grid]
        except: 
            print("Нельзя вычислить значение функции в заданной точке.")
            print("---")
            continue 
        
        print(f"Test: {ind + 1}")
        for x in data:
            print(f"Абсолютная ошибка Лагранжа в точке {x}:", abs_error(interpolate(x, grid, values, method="Lagrange"), function(x)))
            print(f"Абсолютная ошибка Ньютона в точке {x}:", abs_error(interpolate(x, grid, values, method="Newton_div"), function(x)))
            print("---")
        print()


def test_interpolation_evalution() -> None:
    """Тестирование ошибки интерполяции Лагранжа/ Ньютона."""
    low_bound, high_bound = [], []
    interpolate_error_lagrange = []
    interpolate_error_newton = []
    
    data = interpolation_values[3]
    degrees = []
    interval = intervals[3]
    function = functions[3]
    min_max_fun = [lambda x: max(x), lambda x: min(x)]
    
    for n, derivative_function, (a, b) in zip([1, 2], [lambda x: 2 - 0.5 * exp(x), lambda x: - 0.5 * exp(x)], [min_max_fun, min_max_fun]):
        for x in data:
            grid = np.linspace(*interval)        
            low, high, lagrange, newton = get_interpolation_error_bounds(function, derivative_function, a, b, grid, x, n)
            
            low_bound.append(low)
            high_bound.append(high)
            degrees.append(n)
            
            interpolate_error_lagrange.append(function(x) - lagrange)
            interpolate_error_newton.append(function(x) - newton)
    
    result = pd.DataFrame({"x": data * 2, 
                           "Степень полинома": degrees, 
                           "Нижняя граница": low_bound,
                           "Ошибка интерполяции Лагранжа": interpolate_error_lagrange,
                           "Ошибка интерполяции Ньютона": interpolate_error_newton,
                           "Верхняя граница": high_bound})
    print(result)
           

if __name__ == "__main__":
    functions = [
        lambda x: x**2 + log(x), 
        lambda x: x**2 - log10(x + 2), 
        lambda x: (x - 1)**2 - 0.5*exp(x), 
        lambda x: (x - 1)**2 - exp(-x), lambda x: x**3 - sin(x)
    ]

    interpolation_values = [
        [0.52, 0.42, 0.87, 0.67],
        [0.53, 0.52, 0.97, 0.73],
        [0.13, 0.12, 0.57, 0.33],
        [1.07, 1.02, 1.47, 1.27],
    ]

    intervals = [
        [0.4, 0.9, 11], 
        [0.5, 1.0, 11], 
        [0.1, 0.6, 11], 
        [1.0, 1.5, 11]
    ]

    abs_error = lambda x, y: abs(x - y) if x else "inf"
    
    print("Тестирование методов интерполяции с равномерной сеткой.", end='\n\n')
    test_uniform_grid()
    
    print("Тестирование методов интерполяции с неравномерной сеткой.", end='\n\n')
    test_non_uniform_grid()
    test_interpolation_evalution()
    