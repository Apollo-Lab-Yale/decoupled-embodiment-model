import os

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if os.environ.get("DEM_NETWORK_TESTS"):
        return
    skip = pytest.mark.skip(reason="set DEM_NETWORK_TESTS=1 to download weights from the Hub")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
