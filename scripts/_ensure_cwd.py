import os
from pathlib import Path
import sys


def ensure_cwd():
    if Path.cwd().stem == "scripts":
        PROJECT_DIR = Path.cwd().parent
        os.chdir("..")
    else:
        PROJECT_DIR = Path.cwd()

    if Path(os.path.realpath(__file__)).parent.stem == "scripts":
        sys.path.append("../inz")
    return PROJECT_DIR
