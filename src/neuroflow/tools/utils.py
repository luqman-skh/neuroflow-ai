"""Utility functions for deterministic mock analysis tools."""

import hashlib
import json
from typing import Any


def get_deterministic_seed(*args: Any, **kwargs: Any) -> int:
    """Generate a reproducible integer seed from input arguments.
    
    Guarantees that identical arguments always produce the exact same seed,
    while different parameters produce different seeds.
    """
    payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


MOCK_DISCLAIMER_LIMITATION = (
    "DEMO/MOCK OUTPUT: This tool produces deterministic synthetic data for interface "
    "and orchestration prototyping. No real participant records or genuine statistical "
    "tests were computed."
)
