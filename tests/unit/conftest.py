"""The ``network`` marker: tests that reach the Hugging Face Hub are opt-in.

A few adapter tests build their tiny stand-in model from the real upstream config, tokenizer
and ``trust_remote_code`` modeling files, so they need the Hub (or a warm HF cache). Such a
test's fixture requests ``hub_access``; every test that depends on it, directly or through
another fixture, is marked ``network`` and skipped unless ``GLMBENCH_TEST_NETWORK=1``. Select
them with ``-m network``, or leave them out with ``-m "not network"``.
"""

import os

import pytest

NETWORK_ENV = "GLMBENCH_TEST_NETWORK"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"network: needs the Hugging Face Hub (downloads, remote code); runs only with "
        f"{NETWORK_ENV}=1",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "hub_access" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.network)


@pytest.fixture(scope="session")
def hub_access():
    """Request this from any fixture that fetches from the Hub."""
    if os.environ.get(NETWORK_ENV) != "1":
        pytest.skip(f"needs the Hugging Face Hub; set {NETWORK_ENV}=1 to run")
