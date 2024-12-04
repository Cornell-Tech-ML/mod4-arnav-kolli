from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    lst = list(vals)
    lst[arg] = lst[arg] + epsilon
    fplus = f(*lst)

    lst = list(vals)
    lst[arg] = lst[arg] - epsilon
    fminus = f(*lst)

    return (fplus - fminus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivatives"""

    ...

    @property
    def unique_id(self) -> int:
        """Return the unique id"""
        ...

    def is_leaf(self) -> bool:
        """Returns if it is a leaf"""
        ...

    def is_constant(self) -> bool:
        """Return if it is a constant"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parents of variable"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the derivatives of the output with respect to the inputs."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    topological_order = []

    def dfs(v: Variable) -> None:
        """Helper function to perform depth-first search."""
        if v.unique_id in visited or v.is_constant():
            return
        visited.add(v.unique_id)

        for parent in v.parents:
            dfs(parent)

        topological_order.append(v)

    dfs(variable)
    return reversed(topological_order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    sorted_vars = topological_sort(variable)
    derivs = {variable.unique_id: deriv}

    for var in sorted_vars:
        if var.is_leaf():
            var.accumulate_derivative(derivs[var.unique_id])
        else:
            for parent, l_deriv in var.chain_rule(derivs[var.unique_id]):
                if parent.unique_id in derivs:
                    derivs[parent.unique_id] += l_deriv
                else:
                    derivs[parent.unique_id] = l_deriv
    return None


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Store tensors"""
        return self.saved_values
