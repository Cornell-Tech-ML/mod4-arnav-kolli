"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y


def id(x: float) -> float:
    """Returns the identity of a number."""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negates a number."""
    return float(-x)


def lt(x: float, y: float) -> float:
    """Returns True if x is less than y, False otherwise."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Returns True if x is equal to y, False otherwise."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Returns True if x is close to y, False otherwise."""
    return 1.0 if abs(x - y) <= 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Returns the sigmoid of a number."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Returns the ReLU of a number."""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Returns the logarithm of a number."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Returns the exponential of a number."""
    return float(math.exp(x))


def log_back(x: float, d: float) -> float:
    """Returns the derivative of the logarithm function."""
    return d / (x + EPS)


def inv(x: float) -> float:
    """Returns the inverse of a number."""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Returns the derivative of the inverse function."""
    return -(d / x**2)


def relu_back(x: float, d: float) -> float:
    """Returns the derivative of the ReLU function."""
    return d if x >= 0.0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(f: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply function of two arguments cumulatively to the items of a list, from left to right.

    Args:
    ----
        f (Callable[[float, float], float]): A function that takes two floats and returns a float.

    Returns:
    -------
        A function that takes a list, applies the function to each element, and returns a list.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        return [f(x) for x in ls]

    return _map


def zipWith(
    f: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Apply function of two arguments cumulatively to the items of two lists, from left to right.

    Args:
    ----
        f (Callable[[float, float], float]): A function that takes two floats and returns a float.


    Returns:
    -------
        Function that takes two equally sized lists ls1 and ls2 and produces a new list with the result of f(x, y) for each x,y in the lists.

    """
    # it1, it2 = iter(ls1), iter(ls2)
    # result = []
    # while True:
    #     try:
    #         x = next(it1)
    #         y = next(it2)
    #         result.append(f(x, y))
    #     except StopIteration:
    #         break

    def _zipwith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(f(x, y))
        return ret

    return _zipwith


def reduce(
    f: Callable[[float, float], float], initial: float
) -> Callable[[Iterable[float]], float]:
    """Apply function of two arguments cumulatively to the items of a list, from left to right, so as to reduce the list to a single value.

    Args:
    ----
        f (Callable[[float, float], float]): A function that takes two floats and returns a float.
        initial (float): The initial value to use in the reduction.

    Returns:
    -------
        function that takes a list ls of elements and computes the reduction of ls using f and initial.

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = initial
        for x in ls:
            val = f(val, x)
        return val

    return _reduce
    # if not ls:
    #     return 0.0
    # for x in ls:
    #     initial = f(initial, x)
    # return initial


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negates a list of floats.

    Args:
    ----
        ls (Iterable[float]): The iterable of floats to negate.

    Returns:
    -------
        Iterable[float]: The negated iterable of floats.

    """
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Adds two lists of floats.

    Args:
    ----
        ls1 (Iterable[float]): The first iterable of floats.
        ls2 (Iterable[float]): The second iterable of floats.

    Returns:
    -------
        Iterable[float]: The sum of the two iterables of floats.

    """
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sums a list of floats.

    Args:
    ----
        ls (Iterable[float]): The iterable of floats to sum.

    Returns:
    -------
        float: The sum of the list of floats.

    """
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Takes the product of a list of floats.

    Args:
    ----
        ls (Iterable[float]): The iterable of floats to take the product of.

    Returns:
    -------
        float: The product of the list of floats.

    """
    return reduce(mul, 1.0)(ls)
