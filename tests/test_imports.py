"""
Phase 1 import tests.

Verifies all required packages are available and
core module structure is properly set up.
"""

import sys
import pytest


def test_python_version():
    """Verify Python version is 3.10 or higher."""
    assert sys.version_info >= (3, 10), "Python 3.10+ required"


def test_numpy_import():
    """Test numpy is available."""
    import numpy as np
    assert np.__version__
    print(f"✓ numpy {np.__version__}")


def test_pandas_import():
    """Test pandas is available."""
    import pandas as pd
    assert pd.__version__
    print(f"✓ pandas {pd.__version__}")


def test_sklearn_import():
    """Test scikit-learn is available."""
    from sklearn import __version__
    assert __version__
    print(f"✓ scikit-learn {__version__}")


def test_sentinel_package():
    """Test sentinel package structure."""
    import src
    assert src.__version__
    assert "Phase 1" in src.__phase__
    print(f"✓ sentinel {src.__version__} ({src.__phase__})")


def test_sentinel_submodules():
    """Test sentinel submodules exist."""
    from src import data, features, models, evaluation
    print("✓ All submodules available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
