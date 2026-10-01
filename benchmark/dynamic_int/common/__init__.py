# Helpers shared by the dynamic_int benchmarks. Each benchmark's model.py (or run script) puts
# this directory's parent on sys.path and imports `from common import ...`.


def str2bool(x):
    """argparse type for the True/False options."""
    return str(x).lower() in ("true", "1", "yes")
