import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import AppContext  # noqa: E402
from tests.make_fixtures import DATE, offline_settings  # noqa: E402

FIX = Path(__file__).parent / "fixtures"
_CACHE = {}


def _load(name):
    if name not in _CACHE:
        _CACHE[name] = json.loads((FIX / name).read_text())
    return copy.deepcopy(_CACHE[name])


@pytest.fixture
def ctx():
    """Hors-ligne : données synthétiques + LLM mock ; Store neuf à chaque test."""
    return AppContext.build(offline_settings())


@pytest.fixture
def date_analyse():
    return DATE


@pytest.fixture
def etat_metriques():
    return _load("etat_metriques.json")


@pytest.fixture
def notes():
    return _load("notes.json")


@pytest.fixture
def etat_selection():
    return _load("etat_selection.json")
