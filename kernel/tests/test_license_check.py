import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.ci.license_check import check_requirements, check_cargo

def test_license_check_clean():
    # Test with clean requirements file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
        f.write("fastapi\nuvicorn\npydantic\n")
        f.flush()
        temp_path = f.name

    try:
        assert check_requirements(temp_path) is True
    finally:
        os.remove(temp_path)

def test_license_check_agpl():
    # Test with AGPL requirement
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
        f.write("fastapi\nlicense-agpl-v3\n")
        f.flush()
        temp_path = f.name

    try:
        assert check_requirements(temp_path) is False
    finally:
        os.remove(temp_path)

def test_license_check_bsl():
    # Test with BSL requirement
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
        f.write("fastapi\nlicense-bsl-1.1\n")
        f.flush()
        temp_path = f.name

    try:
        assert check_requirements(temp_path) is False
    finally:
        os.remove(temp_path)
