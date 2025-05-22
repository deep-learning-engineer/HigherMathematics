import numpy as np

from collections.abc import Sequence


def get_k_nearest_nodes(grid: Sequence[float], 
                        x: float, k: int = 1) -> list[float]:
    """
    Возвращает `k` ближайщих к `x` узлов в заданной сетке.
    
    Args:
        grid (Sequence[float]): сетка узлов.
        x (float): значение по отнощению к которому будут искаться ближайщие узлы.
        k (int): количество ближайщих узлов (default 1).
    
    Returns:
        list[float]: ближайщие к `x` узлы сетки `grid`.
    """
    k = max(1, k)
    if k > len(grid): 
        k = len(grid) - 1
        
    return sorted(grid, key=lambda y: abs(x - y))[:k]


def get_divided_differences_table(grid_nodes: Sequence[float],
                                  grid_values: Sequence[float]) -> list[float]:
    """
    Вычисляет и возвращает таблицу разделенных разностей.

    Args:
        grid_nodes (Sequence[float]): Узловые точки (x_i).
        grid_values (Sequence[float]): Значения функции в узловых точках (f(x_i)).

    Returns:
        list[float]: Список разделенных разностей.
    """
    n = len(grid_nodes)
    if n == 0:
        return []
    if n != len(grid_values):
        raise ValueError("grid_nodes и grid_values должны быть одной длины.")

    table = np.zeros((n, n))
    table[:, 0] = grid_values 

    for j in range(1, n): 
        for i in range(n - j):
            numerator = table[i+1, j-1] - table[i, j-1]
            denominator = grid_nodes[i+j] - grid_nodes[i]
            table[i, j] = numerator / denominator

    return table[0, :].tolist()


def get_finite_differences_table(grid_values: list[float]) -> list[list[float]]:
    """
    Вычисляет и возвращает таблицу конечных разностей.

    Args:.
        grid_values (list[float]): Значения функции в узловых точках (f(x_i)).

    Returns:
        list[float]: Список конечных разностей.
    """
    result_grid = [grid_values.copy()]
    
    while len(result_grid[-1]) != 1:
        result_grid.append([result_grid[-1][i + 1] - result_grid[-1][i] for i in range(len(result_grid[-1]) - 1)])
              
    return result_grid 


def print_finite_differences_table(finite_table: list[list[float]]) -> None: 
    """Выводит на экран таблицу конечных разностей."""
    prevs_full = [[0, 0]]
    prevs_fractional = [[1, 0]]

    for i in range(len(finite_table)):
        if i == 0:
            print('f(x)', end=' '*6)
        elif i == 1:
            print('Δf', end=' '*7)
        else: 
            print(f'Δ^{i}f', end=' '*6)

    print('\n')
    for i in range(len(finite_table)):
        if i % 2 == 0: 
            for item in prevs_full:
                print("{0:+.6f}".format(finite_table[item[0]][item[1]]), end=' '*10)
                item[1] += 1
            print()
            prevs_full.append([prevs_full[-1][0] + 2, 0])
        else: 
            print(' ' * 10, end='')
            for item in prevs_fractional:
                print("{0:+.6f}".format(finite_table[item[0]][item[1]]), end=' '*10)
                item[1] += 1
            print()
            prevs_fractional.append([prevs_fractional[-1][0] + 2, 0])


    if len(finite_table) % 2:
        prevs_full.pop()
    else:
        prevs_fractional.pop()
        
    for i in range(int(len(finite_table) % 2), len(finite_table)):
        if i % 2 == 0: 
            prevs_full.pop()
            for item in prevs_full:
                print("{0:+.6f}".format(finite_table[item[0]][item[1]]), end=' '*10)
                item[1] += 1
            print()
        else: 
            print(' ' * 10, end='')
            prevs_fractional.pop()
            for item in prevs_fractional:
                print("{0:+.6f}".format(finite_table[item[0]][item[1]]), end=' '*10)
                item[1] += 1
            print()
    