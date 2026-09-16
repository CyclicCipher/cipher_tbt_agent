import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


@pytest.fixture(scope="session")
def env():
    from env.families import load_env
    return load_env()
