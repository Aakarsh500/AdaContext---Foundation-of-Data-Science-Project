import sys

from _bootstrap import ROOT  # noqa: F401
from adacontext.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["eda", *sys.argv[1:]]))

