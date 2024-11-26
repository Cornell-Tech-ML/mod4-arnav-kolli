import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generates a list of N random points in 2D space.

    This function generates a list of N random points in 2D space, where each point is represented as a tuple of two floats. The points are randomly distributed between 0 and 1 on both the x and y axes.

    Args:
    ----
    N (int): The number of random points to generate.

    Returns:
    -------
    List[Tuple[float, float]]
        A list of N random points in 2D space.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generates a simple dataset for binary classification.

    This function generates a dataset for binary classification where the decision boundary is a horizontal line at x_1 = 0.5. Points above this line are labeled as 1, and points below are labeled as 0.

    N (int): The number of data points to generate.

    Returns
    -------
    Graph: A Graph object containing the generated dataset.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generates a dataset for binary classification where the decision boundary is a diagonal line from (0,0) to (1,1). Points below this line are labeled as 1, and points above are labeled as 0.

    N (int): The number of data points to generate.

    Returns
    -------
    Graph: A Graph object containing the generated dataset.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates a dataset for binary classification where the decision boundary is two vertical lines at x_1 = 0.2 and x_1 = 0.8. Points between these lines are labeled as 1, and points outside are labeled as 0.

    N (int): The number of data points to generate.

    Returns
    -------
    Graph: A Graph object containing the generated dataset.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates a dataset for binary classification where the decision boundary is an XOR gate. Points in the top-left and bottom-right quadrants are labeled as 1, and points in the top-right and bottom-left quadrants are labeled as 0.

    N (int): The number of data points to generate.

    Returns
    -------
    Graph: A Graph object containing the generated dataset.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generates a dataset for binary classification where the decision boundary is a circle centered at (0.5, 0.5) with a radius of 0.1. Points inside the circle are labeled as 1, and points outside are labeled as 0.

    Args:
    ----
    N (int): The number of data points to generate.

    Returns:
    -------
    Graph: A Graph object containing the generated dataset.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates a dataset for binary classification where the decision boundary is a spiral. Points on the spiral are labeled as 1, and points outside are labeled as 0.

    Args:
    ----
    N : The number of data points to generate.

    Returns:
    -------
    Graph: A Graph object containing the generated dataset.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
