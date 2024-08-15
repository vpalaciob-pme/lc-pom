from importlib import import_module
from platform import python_version


def _version_as_tuple(ver_str):
    return tuple(int(i) for i in ver_str.split(".") if i.isdigit())


_python_version_tuple = _version_as_tuple(python_version())


if _python_version_tuple >= (3, 10):
    pairwise = import_module("itertools").pairwise
else:

    # pairwise was added in python 3.10
    def pairwise(iterable):
        """
        Return successive overlapping pairs taken from the input iterable.

        The number of 2-tuples in the output iterator will be one fewer than
        the number of inputs. It will be empty if the input iterable has fewer
        than two values.
        """
        iterator = iter(iterable)
        a = next(iterator, None)
        for b in iterator:
            yield a, b
            a = b
