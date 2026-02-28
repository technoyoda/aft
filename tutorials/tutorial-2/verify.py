"""Verify that buggy.py was correctly fixed.

Imports the fixed module and tests each function with the inputs
that would have crashed in the original. Returns exit code 0 only
if all checks pass.
"""

import importlib.util
import sys


def load_module(path):
    spec = importlib.util.spec_from_file_location("buggy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify(path):
    errors = []

    try:
        mod = load_module(path)
    except Exception as e:
        return [f"Failed to import: {e}"]

    # 1. divide(10, 0) should not raise
    try:
        result = mod.divide(10, 0)
        # Should return something sensible, not crash
    except ZeroDivisionError:
        errors.append("divide(10, 0) still raises ZeroDivisionError")
    except Exception as e:
        errors.append(f"divide(10, 0) raises {type(e).__name__}: {e}")

    # 2. average([]) should not raise
    try:
        result = mod.average([])
    except ZeroDivisionError:
        errors.append("average([]) still raises ZeroDivisionError")
    except Exception as e:
        errors.append(f"average([]) raises {type(e).__name__}: {e}")

    # 3. first_element([]) should not raise
    try:
        result = mod.first_element([])
    except IndexError:
        errors.append("first_element([]) still raises IndexError")
    except Exception as e:
        errors.append(f"first_element([]) raises {type(e).__name__}: {e}")

    # 4. Normal cases should still work
    try:
        assert mod.divide(10, 2) == 5.0, f"divide(10, 2) returned {mod.divide(10, 2)}"
    except Exception as e:
        errors.append(f"divide(10, 2) broken: {e}")

    try:
        assert mod.average([1, 2, 3]) == 2.0, f"average([1,2,3]) returned {mod.average([1, 2, 3])}"
    except Exception as e:
        errors.append(f"average([1,2,3]) broken: {e}")

    try:
        assert mod.first_element([42]) == 42, f"first_element([42]) returned {mod.first_element([42])}"
    except Exception as e:
        errors.append(f"first_element([42]) broken: {e}")

    return errors


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify.py <path-to-buggy.py>")
        sys.exit(2)

    errors = verify(sys.argv[1])
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)
