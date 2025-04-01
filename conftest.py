import numpy as np
import qutip
import scqubits
from sybil import Sybil
from sybil.parsers.doctest import DocTestParser
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser

import floquet


def sybil_setup(namespace):  # noqa ARG001
    namespace["np"] = np
    namespace["scq"] = scqubits
    namespace["ft"] = floquet
    namespace["qt"] = qutip


# sybil configuration
pytest_collect_file = Sybil(
    parsers=[DocTestParser(), PythonCodeBlockParser(), SkipParser()],
    patterns=["*.md"],
    setup=sybil_setup,
).pytest()
