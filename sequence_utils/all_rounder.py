"""
sequence_utils/all_rounder.py

Dynamically calls any built-in sequence method by name.
Supports lists, strings, and other sequence types.

Usage:
    python all_rounder.py
"""

import ast


def safe_parse(param: str):
    """
    Parse a string into a Python object safely.
    Uses ast.literal_eval instead of eval() to prevent
    arbitrary code execution.
    """
    try:
        return ast.literal_eval(param)
    except (ValueError, SyntaxError):
        return param


def all_rounder(seq, method: str, param):
    """
    Call a built-in sequence method dynamically.

    Parameters
    ----------
    seq : list or str or any sequence
        The object to call the method on.
    method : str
        Method name e.g. 'append', 'join', 'count'.
    param : any
        Argument to pass. Strings are safely parsed first.

    Returns
    -------
    Result of the method call, the modified sequence
    for in-place methods, or an error message string
    if the method does not exist.

    Examples
    --------
    >>> all_rounder([1, 5], 'append', [3, 4])
    [1, 5, [3, 4]]

    >>> all_rounder('#', 'join', ('Jack', 'Mahmut'))
    'Jack#Mahmut'

    >>> all_rounder([1, 2], 'nonexistent', 5)
    "Method 'nonexistent' does not exist for this sequence type."
    """
    if isinstance(param, str):
        param = safe_parse(param)

    if not hasattr(seq, method):
        return f"Method '{method}' does not exist for this sequence type."

    func = getattr(seq, method)

    if method == "join" and isinstance(seq, str):
        return func(param)

    result = func(*param) if isinstance(param, tuple) else func(param)

    # in-place methods like append return None
    # return the modified sequence instead
    return seq if result is None else result


if __name__ == "__main__":
    assert all_rounder([1, 5], "append", [3, 4, 2, 104]) == [1, 5, [3, 4, 2, 104]]
    assert all_rounder([1, 5], "append", "[3, 4, 2, 104]") == [1, 5, [3, 4, 2, 104]]
    assert all_rounder("#", "join", "('Jack', 'Mahmut')") == "Jack#Mahmut"
    assert all_rounder("#", "join", ("Jack", "Mahmut")) == "Jack#Mahmut"
    assert "does not exist" in all_rounder([1, 2, 3], "nonexistent", 5)
    print("All tests passed.")
