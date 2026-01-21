import functools


def swap_args(func):
    """Returns a new function that swaps the first two arguments."""

    @functools.wraps(func)
    def wrapper(a, b, *args, **kwargs):
        return func(b, a, *args, **kwargs)

    return wrapper
