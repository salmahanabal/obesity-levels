import pytest

from obesity.data import clean, load_raw


@pytest.fixture(scope="session")
def df():
    return clean(load_raw())
