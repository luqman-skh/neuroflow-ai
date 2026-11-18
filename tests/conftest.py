"""Global test configuration and fixtures for NeuroFlow AI.

Ensures that test suites run hermetically in deterministic fake mode
regardless of local developer environment settings.
"""

import os
import pytest

# Force fake provider for deterministic testing
os.environ["LLM_PROVIDER"] = "fake"
os.environ["LLM_MODEL"] = "mock-model"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
